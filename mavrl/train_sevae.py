#!/usr/bin/env python3
"""Stage 2: train the SeVAE on collected RGB-D + segmentation.

Input is the noise-corrupted frame, target is the clean one -- a denoising VAE.
Reconstruction is proximity-weighted; the segmentation cross-entropy is not
(see mavrl.sevae for why).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mavrl.config import BETA_KL, LAMBDA_SEG
from mavrl.course_world import N_SEM_CLASSES, SemClass
from mavrl.dataset import FrameDataset
from mavrl.sevae import SeVAE, seg_class_weights, sevae_loss


def to_tensors(batch: dict, device):
    img = torch.as_tensor(batch["image"], device=device).permute(0, 3, 1, 2)
    gt = torch.as_tensor(batch["image_gt"], device=device).permute(0, 3, 1, 2)
    depth = torch.as_tensor(batch["depth_m"].astype(np.float32),
                            device=device).unsqueeze(1)
    seg = torch.as_tensor(batch["seg"].astype(np.int64), device=device)
    return img, gt, depth, seg


@torch.no_grad()
def seg_iou(logits: torch.Tensor, target: torch.Tensor, n_classes: int):
    """Per-class IoU. Bar classes are the ones that will actually be hard --
    a bar is 1-4% of the frame, so a model that never predicts one still scores
    well on pixel accuracy."""
    pred = logits.argmax(1)
    ious = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        ious.append((p & t).sum().item() / union if union else float("nan"))
    return ious


def split_batches(ds, batch_size, rng, val_every: int):
    """Yield (batch, is_val) pairs. Every val_every-th batch is held out.

    This is a batch-level split, not a frame-level one: FrameDataset reshuffles
    each epoch, so a held-out batch is not guaranteed disjoint from training
    frames across epochs. It catches gross overfitting; it is not a clean
    generalisation estimate. For that, split the shards on disk.
    """
    for i, batch in enumerate(ds.iter_batches(batch_size, rng)):
        yield batch, (val_every > 0 and i % val_every == 0)


@torch.no_grad()
def evaluate(model, batches, device, class_w, args):
    """Mean loss parts over held-out batches, plus per-class IoU."""
    model.eval()
    agg, n = {}, 0
    inter = np.zeros(N_SEM_CLASSES)
    union = np.zeros(N_SEM_CLASSES)

    for batch in batches:
        img, gt, depth, seg = to_tensors(batch, device)
        out = model(img, sample=False)
        _, parts = sevae_loss(
            out, gt, depth, seg, class_weights=class_w,
            beta=args.beta, lambda_seg=args.lambda_seg,
            weight_seg_by_proximity=(args.seg_weight == "proximity"))
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1

        pred = out.seg_logits.argmax(1)
        for c in range(N_SEM_CLASSES):
            pc, tc = pred == c, seg == c
            inter[c] += (pc & tc).sum().item()
            union[c] += (pc | tc).sum().item()

    if n == 0:
        return {}, [float("nan")] * N_SEM_CLASSES

    means = {k: v / n for k, v in agg.items()}
    ious = [inter[c] / union[c] if union[c] else float("nan")
            for c in range(N_SEM_CLASSES)]
    return means, ious


def save_history_plot(history, path: Path):
    """Train vs val curves. Written every epoch so a killed run still has one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    ep = [r["epoch"] for r in history]
    keys = ["recon", "seg", "kl", "total"]

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        tr = [r.get(k) for r in history]
        va = [r.get(f"val_{k}") for r in history]
        if any(v is not None for v in tr):
            ax.plot(ep, tr, label="train", lw=1.4)
        if any(v is not None for v in va):
            ax.plot(ep, va, label="val", lw=1.4, ls="--")
        ax.set_title(k)
        ax.set_xlabel("epoch")
        if k != "kl":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--resume", type=Path, default=None,
                   help="checkpoint to resume from; restores model, optimizer "
                        "and epoch counter. --epochs is the total target, not "
                        "the number of additional epochs.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=BETA_KL)
    p.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG)
    p.add_argument("--seg-weight", choices=("uniform", "proximity"),
                   default="uniform")
    p.add_argument("--val-every", type=int, default=8,
                   help="hold out every Nth batch for validation "
                        "(0 disables validation)")
    p.add_argument("--iou-every", type=int, default=5,
                   help="report per-class val IoU every N epochs (0 disables)")
    p.add_argument("--samples", type=Path, default=None,
                   help="where to write the reconstruction grid "
                        "(default: alongside --out)")
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ds = FrameDataset(args.data)
    counts = ds.class_counts(N_SEM_CLASSES)
    class_w = seg_class_weights(torch.as_tensor(counts)).to(device)
    print("class pixel share:",
          {SemClass(i).name: f"{c / counts.sum():.4f}"
           for i, c in enumerate(counts)})
    print("class weights   :",
          {SemClass(i).name: round(w, 2)
           for i, w in enumerate(class_w.tolist())})

    model = SeVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    hist_path = args.out.parent / "sevae_history.json"
    plot_path = args.out.parent / "sevae_history.png"
    best_path = args.out.with_name(args.out.stem + "_best.pt")

    history: list[dict] = []
    start_epoch = 0
    best_val = float("inf")

    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"--resume: no such checkpoint: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            opt.load_state_dict(ck["optim"])
        else:
            print("warning: checkpoint has no optimizer state; "
                  "Adam moments restart from zero (expect a brief loss bump)")
        start_epoch = ck.get("epoch", -1) + 1
        best_val = ck.get("best_val", float("inf"))

        src_hist = args.resume.parent / "sevae_history.json"
        if src_hist.exists():
            try:
                history = [r for r in json.loads(src_hist.read_text())
                           if r.get("epoch", -1) < start_epoch]
            except json.JSONDecodeError:
                print(f"warning: could not parse {src_hist}; "
                      "history restarts empty")
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    if start_epoch >= args.epochs:
        raise SystemExit(
            f"checkpoint is already at epoch {start_epoch} but --epochs is "
            f"{args.epochs}. Raise --epochs to train further.")

    last_val_batch = None

    for epoch in range(start_epoch, args.epochs):
        model.train()
        agg, n = {}, 0
        val_batches = []

        for batch, is_val in split_batches(ds, args.batch_size, rng,
                                           args.val_every):
            if is_val:
                val_batches.append(batch)
                continue

            img, gt, depth, seg = to_tensors(batch, device)
            out = model(img)
            loss, parts = sevae_loss(
                out, gt, depth, seg, class_weights=class_w,
                beta=args.beta, lambda_seg=args.lambda_seg,
                weight_seg_by_proximity=(args.seg_weight == "proximity"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

        if n == 0:
            raise SystemExit(
                f"epoch {epoch} ran zero training batches: no shard holds "
                f"{args.batch_size} frames, or --val-every is too small. "
                f"Lower --batch-size.")

        row = {k: v / n for k, v in agg.items()} | {"epoch": epoch}

        val_means, val_ious = evaluate(model, val_batches, device, class_w, args)
        row |= {f"val_{k}": v for k, v in val_means.items()}
        if val_batches:
            last_val_batch = val_batches[-1]

        history.append(row)
        line = "  ".join(f"{k}={v:.4f}" for k, v in row.items()
                         if k != "epoch" and not k.startswith("val_"))
        if val_means:
            line += "  |  " + "  ".join(
                f"val_{k}={v:.4f}" for k, v in val_means.items())
        print(f"epoch {epoch:3d}  {line}", flush=True)

        if args.iou_every > 0 and (epoch + 1) % args.iou_every == 0 and val_batches:
            print("          val IoU:",
                  {SemClass(i).name: (None if np.isnan(v) else round(v, 3))
                   for i, v in enumerate(val_ious)}, flush=True)

        ck = {"model": model.state_dict(),
              "optim": opt.state_dict(),
              "epoch": epoch,
              "best_val": best_val}
        torch.save(ck, args.out)

        # Track best-validating weights separately. The last epoch is not
        # necessarily the one you want.
        cur_val = val_means.get("total")
        if cur_val is not None and cur_val < best_val:
            best_val = cur_val
            ck["best_val"] = best_val
            torch.save(ck, best_path)

        hist_path.write_text(json.dumps(history, indent=2))
        if not args.no_plot:
            save_history_plot(history, plot_path)

    # Final per-class IoU on held-out data if available, else a fresh batch.
    model.eval()
    if last_val_batch is not None:
        batch, tag = last_val_batch, "val"
    else:
        batch, tag = next(ds.iter_batches(args.batch_size, rng)), "train (no val split)"
    img, gt, depth, seg = to_tensors(batch, device)
    out = model(img, sample=False)
    ious = seg_iou(out.seg_logits, seg, N_SEM_CLASSES)
    print(f"\nfinal seg IoU [{tag}]:",
          {SemClass(i).name: (None if np.isnan(v) else round(v, 3))
           for i, v in enumerate(ious)})

    # The losses above cannot tell you whether the bars survived. Look at the
    # picture.
    if not args.no_samples:
        from mavrl.visualize import save_sevae_samples
        save_sevae_samples(model, batch,
                           args.samples or args.out.parent / "sevae_samples.png")

    hist_path.write_text(json.dumps(history, indent=2))
    if not args.no_plot:
        save_history_plot(history, plot_path)
        print("saved", plot_path)
    if best_val < float("inf"):
        print(f"best val total {best_val:.4f} ->", best_path)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())