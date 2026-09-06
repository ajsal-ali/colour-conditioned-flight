#!/usr/bin/env python3
"""Figures for the stages whose numbers are hard to read.

A SeVAE loss of 0.026 tells you nothing about whether the bars survived. A memory
`past` loss of 0.08 tells you nothing about whether the model remembers a bar or
has just learned the average brightness of a corridor. Both stages therefore
write a PNG at the end of training, and both can be re-run standalone:

    python -m mavrl.visualize --data data --sevae ckpt/sevae.pt \
        --memory ckpt/memory_lstm.pt --out samples

`save_training_curves` is the third: PPO's `log.jsonl` plotted as return, success,
AGV and the loss terms against timesteps. `train_course` writes it as it goes, and
it re-runs on any log -- including one pulled off Modal mid-flight:

    python -m mavrl.visualize --log runs/course/log.jsonl --out runs/course

What to look for is printed under each figure and repeated in mavrl/README.md.

matplotlib is imported lazily and on the Agg backend: these run on headless
machines, and a missing matplotlib must never take a training run down with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mavrl.config import AUX_T, IMG_RES, LATENT_DIM
from mavrl.course_world import N_SEM_CLASSES
from mavrl.memory import aux_valid_range


def _plt():
    """Agg-backend pyplot, or None if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:                      # pragma: no cover - env dependent
        print(f"[visualize] matplotlib unavailable ({exc}); skipping samples")
        return None


def _grid(plt, rows, path: Path, title: str = "", scale: float = 1.9):
    """rows = [(label, [image, ...], cmap_or_None)], one row per label."""
    n = len(rows[0][1])
    fig, ax = plt.subplots(len(rows), n,
                           figsize=(scale * n, scale * len(rows)),
                           squeeze=False)
    for r, (label, images, cmap) in enumerate(rows):
        for c, im in enumerate(images):
            kw = {"cmap": cmap} if cmap else {}
            if cmap == "tab10":
                kw |= {"vmin": 0, "vmax": max(9, N_SEM_CLASSES - 1)}
            ax[r][c].imshow(np.asarray(im), **kw)
            ax[r][c].axis("off")
        # Row label goes on the first panel's title: axis("off") kills ylabels.
        ax[r][0].set_title(label, fontsize=8, loc="left")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _chw_rgb(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


@torch.no_grad()
def save_sevae_samples(model, batch: dict, path, n: int = 6) -> None:
    """Input / clean target / reconstruction, for RGB, depth and segmentation.

    The row that matters is `recon RGB`: near geometry should be visibly sharper
    than far geometry. That asymmetry *is* the proximity weight -- if near and far
    are equally blurry the weight is not reaching the loss.
    """
    plt = _plt()
    if plt is None:
        return
    model.eval()
    device = next(model.parameters()).device
    n = min(n, len(batch["image"]))
    img = torch.as_tensor(batch["image"][:n], device=device).permute(0, 3, 1, 2)
    out = model(img, sample=False)

    rows = [
        ("input RGB (noisy)", [batch["image"][i, ..., :3] for i in range(n)], None),
        ("target RGB (clean)", [batch["image_gt"][i, ..., :3] for i in range(n)], None),
        ("recon RGB", [_chw_rgb(out.rgb[i]) for i in range(n)], None),
        ("target depth", [batch["image_gt"][i, ..., 3] for i in range(n)], "viridis"),
        ("recon depth", [out.depth[i, 0].cpu().numpy() for i in range(n)], "viridis"),
    ]
    if "seg" in batch:
        rows += [
            ("true seg", [batch["seg"][i] for i in range(n)], "tab10"),
            ("pred seg", [out.seg_logits[i].argmax(0).cpu().numpy()
                          for i in range(n)], "tab10"),
        ]
    _grid(plt, rows, Path(path), "SeVAE reconstruction")
    print("  look for: near geometry sharper than far (the proximity weight), "
          "and bars present in `pred seg` -- a bar is 1-4% of the frame, so a "
          "model that never predicts one still scores well on pixel accuracy.")


@torch.no_grad()
def save_memory_samples(sevae, memory, aux, batch: dict, path, T: int = AUX_T,
                        n: int = 4, t_index: int | None = None) -> None:
    """Ground truth vs reconstruction for every supervised aux segment.

    This is the figure that answers "is the memory actually working". The `past`
    pair is the test: `Î_{t-T}` is reconstructed from `z_t` alone, so anything
    recognizable in it -- a bar, a window edge -- is information the recurrent
    state carried forward after the frame left the field of view. Compare it
    against the `current` pair, which any backbone gets nearly for free.
    """
    plt = _plt()
    if plt is None:
        return
    sevae.eval()
    memory.eval()
    aux.eval()
    device = next(memory.parameters()).device

    n = min(n, len(batch["image"]))
    img = torch.as_tensor(batch["image"][:n], device=device)
    gt = torch.as_tensor(batch["image_gt"][:n], device=device)
    proprio = torch.as_tensor(batch["proprio"][:n], device=device)
    b, seq = img.shape[:2]

    offsets = aux.offsets(T)
    lo, hi = aux_valid_range(seq, offsets)
    if hi <= lo:
        print(f"[visualize] window of {seq} too short for offsets {list(offsets)}")
        return
    # Latest fully-valid step: the memory has had the most context to accumulate.
    t = hi - 1 if t_index is None else int(np.clip(t_index, lo, hi - 1))

    z, _, _ = sevae.encode(                       # encode() normalizes uint8 itself
        img.permute(0, 1, 4, 2, 3).reshape(b * seq, 4, IMG_RES, IMG_RES),
        sample=False)
    z_seq, _ = memory(z.view(b, seq, -1), None)
    pred = aux(z_seq[:, t], proprio[:, t])                  # (B, n_seg, latent)

    rows = []
    for i, (name, off) in enumerate(zip(aux.segments, offsets)):
        rgb, _, _ = sevae.decoder(pred[:, i].reshape(-1, LATENT_DIM))
        truth = gt[:, t + off, ..., :3].cpu().numpy()
        rows.append((f"{name}: true I(t{off:+d})",
                     [truth[k] for k in range(n)], None))
        rows.append((f"{name}: recon",
                     [_chw_rgb(rgb[k]) for k in range(n)], None))

    _grid(plt, rows, Path(path),
          f"memory reconstruction from z_t  (t={t}, T={T}, window={seq})")
    print("  look for: the `past` recon showing something the drone can no longer "
          "see at t. If `past` is a featureless smear while `current` is sharp, "
          "the backbone is describing, not remembering.")


# -- PPO training curves ------------------------------------------------------

#: (log key, panel title). Order is the reading order: what the policy achieved
#: first, then what the optimizer did to get there.
CURVE_PANELS = (
    ("return", "episode return"),
    ("success", "success rate"),
    ("agv", "AGV  [m/s]"),
    ("ep_len", "episode length  [policy steps]"),
    ("gates", "gates cleared  [fraction]"),
    ("heading_err", "|heading error|  [deg]"),
    ("pg", "policy-gradient loss"),
    ("vf", "value loss"),
    ("ent", "entropy"),
    ("kl", "approx KL"),
    ("sil", "self-imitation loss"),
    ("expert_frac", "envs flown by the pilot"),
)


def read_log(path) -> list[dict]:
    """Parse a `log.jsonl`, tolerating a torn final line.

    A run killed mid-write leaves a partial row; refusing to plot the other
    4 000 because of it would be perverse.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _column(rows, key) -> np.ndarray:
    """One log field as a float column, NaN where the row predates the field."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[i] = float(v)
    return out


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Trailing mean over k points, ignoring NaNs."""
    if k <= 1:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        if w.size:
            out[i] = w.mean()
    return out


def save_training_curves(log, path, smooth: int = 15) -> None:
    """Plot a `train_course` log to a single PNG.

    Episode-level series (return, success, AGV) are already means over the last
    200 episodes; the per-rollout loss terms are not, so every panel draws the
    raw trace faintly under a trailing mean. Stage changes are dashed verticals
    across all panels -- on a curriculum run the success drop at each one is the
    signal, and without the marks it reads as the policy collapsing.
    """
    plt = _plt()
    if plt is None:
        return
    rows = read_log(log)
    if len(rows) < 2:
        print(f"[visualize] {log}: {len(rows)} rows, nothing to plot yet")
        return

    x = _column(rows, "timesteps")
    if np.all(np.isnan(x)):
        x = np.arange(len(rows), dtype=float)
    columns = [(k, t, _column(rows, k)) for k, t in CURVE_PANELS]
    columns = [(k, t, y) for k, t, y in columns
               if not np.all(np.isnan(y) | (y == 0.0))]
    if not columns:
        print(f"[visualize] {log}: no plottable columns")
        return

    stage = _column(rows, "stage")
    marks = [x[i] for i in range(1, len(rows))
             if not np.isnan(stage[i]) and stage[i] != stage[i - 1]]

    ncol = 3
    nrow = -(-len(columns) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow),
                           squeeze=False, sharex=True)
    for a in ax.ravel()[len(columns):]:
        a.axis("off")

    for i, (key, title, y) in enumerate(columns):
        a = ax[i // ncol][i % ncol]
        a.plot(x, y, lw=0.7, alpha=0.3, color="C0")
        a.plot(x, _smooth(y, smooth), lw=1.7, color="C0")
        for m in marks:
            a.axvline(m, color="0.6", ls="--", lw=0.8, zorder=0)
        a.set_title(title, fontsize=9)
        a.grid(alpha=0.25, lw=0.5)
        if key in ("success", "gates"):
            a.set_ylim(-0.02, 1.02)
        if i // ncol == nrow - 1 or i + ncol >= len(columns):
            a.set_xlabel("timesteps", fontsize=8)
        a.tick_params(labelsize=7)

    last = rows[-1]
    fig.suptitle(
        f"{Path(log).parent.name}  --  {int(last.get('timesteps', 0)):,} steps, "
        f"stage {last.get('stage', '?')}, "
        f"success {last.get('success', float('nan')):.2f}, "
        f"AGV {last.get('agv', float('nan')):.2f} m/s",
        fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def _load_batch(data, seq_len: int, batch: int, keys, seed: int = 0):
    from mavrl.dataset import FrameDataset, SequenceDataset
    rng = np.random.default_rng(seed)
    ds = (FrameDataset(data, keys=keys) if seq_len <= 1
          else SequenceDataset(data, seq_len=seq_len, keys=keys))
    return next(ds.iter_batches(batch, rng))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--sevae", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--memory", type=Path, default=None,
                   help="a stage-3 checkpoint; omit to render the SeVAE only")
    p.add_argument("--out", type=Path, default=Path("samples"))
    p.add_argument("--log", type=Path, default=None,
                   help="a train_course log.jsonl; plots the curves and exits")
    p.add_argument("--smooth", type=int, default=15,
                   help="trailing-mean width, in rollouts, for --log")
    p.add_argument("--T", type=int, default=AUX_T)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    # Curves need neither a checkpoint nor a dataset -- do them before anything
    # that would demand one.
    if args.log:
        save_training_curves(args.log, args.out / "curves.png", args.smooth)
        return 0

    from mavrl.memory import AuxReconstructionHead, build_memory
    from mavrl.sevae import SeVAE

    device = torch.device(args.device)
    sevae = SeVAE().to(device)
    sevae.load_state_dict(torch.load(args.sevae, map_location=device)["model"])

    batch = _load_batch(args.data, 1, args.n,
                        ("image", "image_gt", "seg", "depth_m"), args.seed)
    save_sevae_samples(sevae, batch, args.out / "sevae.png", args.n)

    if args.memory:
        ckpt = torch.load(args.memory, map_location=device)
        mem_kwargs = ({} if ckpt.get("mem_tokens") is None
                      else {"n_mem": ckpt["mem_tokens"]})
        memory = build_memory(ckpt["memory_type"], **mem_kwargs).to(device)
        memory.load_state_dict(ckpt["memory"])
        aux = AuxReconstructionHead(
            segments=ckpt.get("aux_segments", ("past", "current"))).to(device)
        aux.load_state_dict(ckpt["aux"])

        seq = _load_batch(args.data, args.seq_len, min(args.n, 4),
                          ("image", "image_gt", "proprio"), args.seed)
        save_memory_samples(sevae, memory, aux, seq,
                            args.out / f"memory_{args.memory.stem}.png",
                            T=ckpt.get("T", args.T), n=min(args.n, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
