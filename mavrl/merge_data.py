#!/usr/bin/env python3
"""Combine several shard directories into one trainable dataset.

    python -m mavrl.merge_data --out data_all data data_manual

Every loader in `mavrl.dataset` globs `*.npz` under one root, so merging really
is just gathering files -- but doing it by hand invites the two failure modes
this guards against:

* **Filename collisions.** Both collectors write `<prefix>_00000.npz`; copying
  them into one directory silently drops one. Shards are renamed
  `<source-dir>__<original>` here, which also keeps provenance readable.
* **Incompatible columns.** A shard written before a format change, or at a
  different `IMG_RES`, loads fine on its own and then explodes mid-epoch when
  `np.stack` hits a mismatched frame. Every shard is checked against the first
  one's key set and per-frame shapes before anything is written.

Default is to copy. `--link` hard-links instead (same filesystem only, no extra
disk); `--move` relocates. Nothing is ever deleted from the sources on a copy.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from mavrl.dataset import summarize

#: Per-frame arrays, whose trailing shape must agree across every shard.
FRAME_ARRAYS = ("image", "image_gt", "seg", "depth_m", "proprio", "action",
                "gate_flags")


def shard_signature(path: Path) -> dict:
    """Key set plus per-frame shape/dtype -- everything a merge can break."""
    with np.load(path) as z:
        keys = set(z.files)
        shapes = {k: (tuple(z[k].shape[1:]), str(z[k].dtype))
                  for k in FRAME_ARRAYS if k in keys}
        n_frames = len(z["image"])
        n_eps = len(z["ep_len"])
    return {"keys": keys, "shapes": shapes, "frames": n_frames,
            "episodes": n_eps}


def describe_mismatch(ref: dict, sig: dict) -> str:
    missing = ref["keys"] - sig["keys"]
    extra = sig["keys"] - ref["keys"]
    parts = []
    if missing:
        parts.append(f"missing {sorted(missing)}")
    if extra:
        parts.append(f"unexpected {sorted(extra)}")
    for k, want in ref["shapes"].items():
        got = sig["shapes"].get(k)
        if got is not None and got != want:
            parts.append(f"{k}: {got} != {want}")
    return "; ".join(parts) or "unknown difference"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", type=Path,
                   help="shard directories to combine")
    p.add_argument("--out", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--link", action="store_true",
                      help="hard-link instead of copying (same filesystem only)")
    mode.add_argument("--move", action="store_true",
                      help="move the shards out of the sources")
    p.add_argument("--force", action="store_true",
                   help="merge even if a shard's columns disagree (it will be "
                        "skipped, not coerced)")
    args = p.parse_args(argv)

    shards = []
    for src in args.sources:
        if not src.is_dir():
            raise SystemExit(f"not a directory: {src}")
        found = sorted(src.glob("*.npz"))
        if not found:
            raise SystemExit(f"no .npz shards under {src}")
        shards += [(src, f) for f in found]
    print(f"{len(shards)} shards from {len(args.sources)} sources")

    ref = ref_path = None
    accepted, skipped = [], []
    for src, path in shards:
        sig = shard_signature(path)
        if ref is None:
            ref, ref_path = sig, path
        elif sig["keys"] != ref["keys"] or sig["shapes"] != ref["shapes"]:
            reason = describe_mismatch(ref, sig)
            skipped.append((path, reason))
            if not args.force:
                raise SystemExit(
                    f"{path} is not compatible with {ref_path}: {reason}\n"
                    f"Re-collect it, or pass --force to skip incompatible "
                    f"shards instead of failing.")
            continue
        accepted.append((src, path, sig))

    args.out.mkdir(parents=True, exist_ok=True)
    total_frames = total_eps = 0
    for src, path, sig in accepted:
        dest = args.out / f"{src.name}__{path.name}"
        if dest.resolve() == path.resolve():
            continue                      # merging a directory into itself
        if args.move:
            shutil.move(str(path), dest)
        elif args.link:
            dest.unlink(missing_ok=True)
            os.link(path, dest)
        else:
            shutil.copy2(path, dest)
        total_frames += sig["frames"]
        total_eps += sig["episodes"]

    verb = "moved" if args.move else ("linked" if args.link else "copied")
    print(f"{verb} {len(accepted)} shards -> {args.out} "
          f"({total_eps} episodes, {total_frames} frames)")
    for path, reason in skipped:
        print(f"  SKIPPED {path}: {reason}")

    report = summarize(args.out)
    print("\nmerged dataset:", report)
    by_src = report.get("frames_by_source", {})
    if by_src:
        total = sum(by_src.values()) or 1
        print("frame mix:", {k: f"{v} ({v / total:.0%})"
                             for k, v in sorted(by_src.items())})
    if report["problems"]:
        print("PROBLEMS:", *report["problems"], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
