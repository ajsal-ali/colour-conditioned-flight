#!/usr/bin/env python3
"""Stage 3: train the memory module with a frozen SeVAE.

The objective is the paper's Eq. (2): from z_t (plus the state vector x_t),
reconstruct several images at once -- `I_{t-T}`, `I_t`, `I_{t+T}` -- all through
the same frozen decoder, with `lambda_i in {0,1}` selecting which. Note this is
the *only* multi-frame stage: the VAE itself (Eq. (1)) is trained per-frame with
no time index and no recurrence, which is why stage 2 uses `FrameDataset` and
this stage uses `SequenceDataset`.

`--aux-segments` is that lambda configuration. Default `past,current` = (1,1,0);
`past,current,future` reproduces the full three-way head.

Remember what T is. It is the *prediction offset*, not a memory window -- an
LSTM's hidden state has no horizon, and T=20 is a supervision target that
pressures it into retaining ~20 steps. For the attention backbone the window K
genuinely is a cap, which is why K=16 < T=20: if the window still contained step
t-T the loss would be satisfiable by copying rather than remembering, and only
the LSTM would face a real memory task.

Validation is split **by episode**, not by window. With stride=4 and seq_len=48
consecutive windows share 44 of their 48 frames, so a window-level split would
put near-duplicates of every training window into val and report a val loss that
means nothing. Both views are built with the same `--split-seed`; changing it
between runs reshuffles which episodes are held out and makes the numbers
incomparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mavrl.config import AUX_SEGMENTS, AUX_T, LATENT_DIM
from mavrl.dataset import SequenceDataset
from mavrl.memory import AuxReconstructionHead, aux_valid_range, build_memory
from mavrl.sevae import SeVAE


def run_epoch(ds, batch_size, rng, sevae, memory, aux, segments, offsets,
              lo, hi, device, opt=None):
    """One pass over `ds`. `opt=None` means evaluation: no grad, no shuffle.

    Returns (summed per-segment losses, n_batches). The caller divides.
    """
    train = opt is not None
    memory.train(train)
    aux.train(train)

    agg = {s: 0.0 for s in segments} | {"total": 0.0}
    n = 0

    def decode(latent):
        rgb, depth, _ = sevae.decoder(latent.reshape(-1, LATENT_DIM))
        return torch.cat([rgb, depth], dim=1)

    with torch.enable_grad() if train else torch.no_grad():
        # Evaluation keeps every shard: a small val split can easily have no
        # shard holding `batch_size` windows, and dropping them all reports a
        # flawless 0.0.
        for batch in ds.iter_batches(batch_size, rng, shuffle=train,
                                     drop_last=train):
            img = torch.as_tensor(batch["image"], device=device)
            gt = torch.as_tensor(batch["image_gt"], device=device)
            proprio = torch.as_tensor(batch["proprio"], device=device)
            b, t = img.shape[:2]

            with torch.no_grad():
                z, _, _ = sevae.encode(
                    img.permute(0, 1, 4, 2, 3).reshape(b * t, 4, *img.shape[2:4]),
                    sample=False)
                z = z.view(b, t, -1)

            z_seq, _ = memory(z, None)                       # (B,T,256)

            # Only timesteps whose every target lies inside the window count.
            idx = torch.arange(lo, min(hi, t), device=device)
            pred = aux(z_seq[:, idx], proprio[:, idx])       # (B,T',n_seg,latent)

            tgt = gt.permute(0, 1, 4, 2, 3).float() / 255.0
            per_seg = {}
            loss = 0.0
            for i, (name, off) in enumerate(zip(segments, offsets)):
                target = tgt[:, idx + off].reshape(-1, 4, *img.shape[2:4])
                l = F.mse_loss(decode(pred[..., i, :]), target)
                per_seg[name] = l
                loss = loss + l

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(memory.parameters()) + list(aux.parameters()), 1.0)
                opt.step()

            for name, l in per_seg.items():
                agg[name] += l.item()
            agg["total"] += loss.item()
            n += 1

    return agg, n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data_all"))
    p.add_argument("--sevae", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention"))
    p.add_argument("--mem-tokens", type=int, default=None,
                   help="attention only; 0 = plain windowed attention")
    p.add_argument("--T", type=int, default=AUX_T)
    p.add_argument("--aux-segments", default=",".join(AUX_SEGMENTS),
                   help="paper's lambda_i, by name: comma-separated subset of "
                        "past,current,future")
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of EPISODES held out (0 disables validation)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="keep fixed across runs or val numbers are incomparable")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--samples", type=Path, default=None,
                   help="where to write the past/current reconstruction grid "
                        "(default: alongside --out)")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    segments = tuple(s.strip() for s in args.aux_segments.split(",") if s.strip())
    aux = AuxReconstructionHead(segments=segments)
    offsets = aux.offsets(args.T)
    lo, hi = aux_valid_range(args.seq_len, offsets)
    if hi <= lo:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) is too short for segments "
            f"{list(segments)} at T={args.T}: offsets {list(offsets)} leave no "
            f"timestep whose targets all lie inside the window "
            f"(need seq_len > {max(offsets) - min(offsets)})")

    tag = args.memory_type + ("" if args.mem_tokens is None
                              else f"_m{args.mem_tokens}")
    if segments != AUX_SEGMENTS:
        tag += "_" + "".join(s[0] for s in segments)
    out = args.out or Path(f"ckpt/memory_{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    sevae = SeVAE().to(device)
    sevae.load_state_dict(torch.load(args.sevae, map_location=device)["model"])
    sevae.eval()
    for prm in sevae.parameters():
        prm.requires_grad = False

    mem_kwargs = {} if args.mem_tokens is None else {"n_mem": args.mem_tokens}
    memory = build_memory(args.memory_type, **mem_kwargs).to(device)
    aux = aux.to(device)
    opt = torch.optim.Adam(
        list(memory.parameters()) + list(aux.parameters()), lr=args.lr)

    ds_kwargs = dict(seq_len=args.seq_len,
                     keys=("image", "image_gt", "proprio"))
    use_val = args.val_frac > 0.0
    if use_val:
        train_ds = SequenceDataset(args.data, split="train",
                                   val_frac=args.val_frac,
                                   split_seed=args.split_seed, **ds_kwargs)
        val_ds = SequenceDataset(args.data, split="val",
                                 val_frac=args.val_frac,
                                 split_seed=args.split_seed, **ds_kwargs)
    else:
        train_ds, val_ds = SequenceDataset(args.data, **ds_kwargs), None

    if len(train_ds) == 0:
        raise SystemExit(
            f"no windows of {args.seq_len} frames in {args.data} -- every "
            f"episode is shorter than that. Lower --seq-len, or collect longer "
            f"episodes.")
    if use_val and len(val_ds) == 0:
        raise SystemExit(
            f"the val split has 0 windows ({val_ds.n_episodes()} episodes, all "
            f"shorter than --seq-len {args.seq_len}). Raise --val-frac, lower "
            f"--seq-len, or pass --val-frac 0.")

    if use_val:
        # These two counts must differ. Identical counts mean both views got the
        # same window list, which is the bug this split was written to kill.
        print(f"episodes: {train_ds.n_episodes()} train / "
              f"{val_ds.n_episodes()} val (split by episode, "
              f"no window overlap across the split)")
        print(f"{len(train_ds)} train windows | {len(val_ds)} val windows | "
              f"{args.seq_len} frames | memory={tag} T={args.T}")
    else:
        print(f"{len(train_ds)} windows of {args.seq_len} frames | "
              f"memory={tag} T={args.T} | validation disabled")
    print(f"segments={list(segments)} offsets={list(offsets)} "
          f"supervised timesteps [{lo},{hi}) of {args.seq_len}")

    # `past` is the selection criterion; `current` is near-free for any backbone
    # and would rank the checkpoints by nothing at all.
    key_metric = "past" if "past" in segments else "total"
    best = float("inf")
    history = []

    def save(path, epoch, val_row):
        torch.save({"memory": memory.state_dict(), "aux": aux.state_dict(),
                    "memory_type": args.memory_type,
                    "mem_tokens": args.mem_tokens, "T": args.T,
                    "aux_segments": list(segments), "epoch": epoch,
                    "val": val_row}, path)

    for epoch in range(args.epochs):
        tr, ntr = run_epoch(train_ds, args.batch_size, rng, sevae, memory, aux,
                            segments, offsets, lo, hi, device, opt=opt)
        # `iter_batches` skips any shard holding fewer than batch_size windows,
        # so an epoch can silently do nothing and report a loss of 0.0.
        if ntr == 0:
            raise SystemExit(
                f"epoch {epoch} ran zero batches: no shard holds "
                f"{args.batch_size} windows of {args.seq_len} frames. Lower "
                f"--batch-size or --seq-len.")
        row = {f"train_{k}": v / ntr for k, v in tr.items()} | {"epoch": epoch}
        line = "  ".join(f"{s}={row['train_' + s]:.5f}" for s in segments)

        val_row = None
        if use_val and (epoch % args.val_every == 0
                        or epoch == args.epochs - 1):
            va, nva = run_epoch(val_ds, args.batch_size, rng, sevae, memory,
                                aux, segments, offsets, lo, hi, device,
                                opt=None)
            if nva == 0:
                raise SystemExit(
                    f"validation ran zero batches over {len(val_ds)} windows. "
                    f"This should be unreachable with drop_last=False -- the "
                    f"val split is probably empty.")
            val_row = {k: v / nva for k, v in va.items()}
            row |= {f"val_{k}": v for k, v in val_row.items()}
            line += "  |  " + "  ".join(
                f"val_{s}={val_row[s]:.5f}" for s in segments)
            if val_row[key_metric] < best:
                best = val_row[key_metric]
                save(out.with_name(out.stem + "_best" + out.suffix),
                     epoch, val_row)
                line += f"  * best val_{key_metric}"

        history.append(row)
        print(f"epoch {epoch:3d}  {line}", flush=True)
        save(out, epoch, val_row)

    # A `past` MSE is not a picture. Render one -- it is the only way to see the
    # difference between remembering a bar and learning the mean corridor. Draw
    # it from val: held-out frames are the ones that distinguish the two.
    if not args.no_samples:
        from mavrl.visualize import save_memory_samples
        sample_ds = val_ds if use_val else train_ds
        sample_batch = next(sample_ds.iter_batches(
            min(4, args.batch_size), rng, shuffle=False, drop_last=False), None)
        if sample_batch is not None:
            save_memory_samples(
                sevae, memory, aux, sample_batch,
                args.samples or out.parent / f"memory_{tag}_samples.png",
                T=args.T, n=4)

    (out.parent / f"memory_{tag}_history.json").write_text(
        json.dumps(history, indent=2))
    print("saved", out)
    if use_val:
        print(f"best val_{key_metric}={best:.5f} -> "
              f"{out.with_name(out.stem + '_best' + out.suffix)}")
    print("\nNote: `past` is the number that matters. It is the one the "
          "attention window cannot satisfy by copying when K < T. `current` is "
          "near-free for any backbone, and `future` is partly unlearnable -- "
          "the paper's Fig. 2(b) shows it as the blurriest of the three.")
    if use_val:
        print("Note: with only a handful of held-out episodes the val curve is "
              "noisy; treat a small val_past gap as inconclusive rather than "
              "as evidence the memory generalises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())