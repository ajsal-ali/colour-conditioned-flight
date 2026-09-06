#!/usr/bin/env python3
"""Sharded sequence dataset for SeVAE, memory and BC training.

The existing rl/ dataset (`demos.npz`) is 8042 flat i.i.d. frames with no
episode boundaries, which cannot train a sequence model at all. This format
keeps them:

    image      (N,128,128,4) uint8    encoder input (noise-corrupted)
    image_gt   (N,128,128,4) uint8    CLEAN reconstruction target
    seg        (N,128,128)   uint8    semantic class ids
    depth_m    (N,128,128)   float16  metres, drives the proximity weight
    proprio    (N,16)        float32
    action     (N,4)         float32
    ep_start   (M,)          int64    episode boundaries into N
    ep_len     (M,)          int64
    ep_layout  (M,)          <U64     layout descriptor, for stratification
    ep_source  (M,)          <U16     "scripted" | "manual" -- survives a merge
    gate_flags (N,)          uint8    gates cleared as of this frame

Sharded per batch of episodes: at 128^2 with RGB-D, clean RGB-D, seg and depth,
one frame is ~115 kB, so 2000 episodes lands in the tens of GB if written as a
single file.

Window indices are **shard-local**. `ep_start` restarts at 0 in every shard, so
a window start only means something paired with the shard it came from. The
train/val split therefore partitions *episodes* while preserving that pairing;
flattening the windows into one global list is exactly what produced

    ValueError: all input arrays must have the same shape

-- a start from shard 3 sliced against shard 0's arrays runs off the end, numpy
silently returns a short slice, and `np.stack` is the first thing to notice.
`_shard_windows` now asserts every window lies inside its shard so that the
failure, if it ever recurs, names the episode instead of a stack call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from mavrl.config import IMG_RES


class ShardWriter:
    """Accumulates episodes and flushes a shard every `episodes_per_shard`."""

    def __init__(self, out_dir, episodes_per_shard: int = 50, prefix: str = "shard"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_shard = episodes_per_shard
        self.prefix = prefix
        self.shard_idx = 0
        self._reset()

    def _reset(self) -> None:
        self.buf = {k: [] for k in
                    ("image", "image_gt", "seg", "depth_m", "proprio",
                     "action", "gate_flags")}
        self.ep_len: List[int] = []
        self.ep_layout: List[str] = []
        self.ep_source: List[str] = []
        self.ep_success: List[bool] = []

    def add_episode(self, frames: dict, layout_desc: str,
                    source: str = "scripted", success: bool = True) -> None:
        """`success` is whether the pilot actually flew the whole course.

        It has to be stored, not just printed, because the collector
        deliberately keeps flying after a missed gate (see the module docstring)
        so the SeVAE gets failure frames. That makes every shard a mix of clean
        traversals and botched ones, and BC has no way to tell them apart
        without this flag -- it would clone the crashes as readily as the
        good runs.
        """
        n = len(frames["image"])
        if n == 0:
            return
        for k in self.buf:
            self.buf[k].append(np.asarray(frames[k]))
        self.ep_len.append(n)
        self.ep_layout.append(layout_desc)
        self.ep_source.append(source)
        self.ep_success.append(bool(success))
        if len(self.ep_len) >= self.episodes_per_shard:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self.ep_len:
            return None
        ep_len = np.asarray(self.ep_len, dtype=np.int64)
        ep_start = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
        path = self.dir / f"{self.prefix}_{self.shard_idx:05d}.npz"
        np.savez_compressed(
            path,
            ep_start=ep_start, ep_len=ep_len,
            ep_layout=np.asarray(self.ep_layout, dtype="<U64"),
            ep_source=np.asarray(self.ep_source, dtype="<U16"),
            ep_success=np.asarray(self.ep_success, dtype=bool),
            **{k: np.concatenate(v, axis=0) for k, v in self.buf.items()})
        self.shard_idx += 1
        self._reset()
        return path

    def close(self) -> Optional[Path]:
        return self.flush()


class SequenceDataset:
    """Yields fixed-length windows that never cross an episode boundary.

    Iterates **shard at a time**, decompressing each shard exactly once per
    epoch. The obvious per-window `np.load` is catastrophically slow -- a
    compressed npz is fully decompressed on every access, so a 2-shard, 1621
    frame set took longer to iterate than it took to generate.

    `split` partitions by episode, never by window: windows overlap by
    `seq_len - stride` frames, so a window-level split leaks almost the whole
    training set into val. Build the train and val views with the *same*
    `split_seed` or the two halves will overlap.
    """

    def __init__(self, root, seq_len: int = 32,
                 keys: Optional[Sequence[str]] = None, stride: int = 4,
                 split: Optional[str] = None, val_frac: float = 0.2,
                 split_seed: int = 0, stratify_by_layout: bool = True,
                 only_success: bool = False):
        self.root = Path(root)
        self.shards = sorted(self.root.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {self.root}")
        if split not in (None, "train", "val"):
            raise ValueError(f"split must be None, 'train' or 'val', got {split!r}")
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.split = split
        self.only_success = only_success
        self.n_dropped = 0
        self.keys = tuple(keys) if keys else (
            "image", "image_gt", "seg", "depth_m", "proprio", "action")
        self.episodes = self._select_episodes(
            split, val_frac, split_seed, stratify_by_layout)
        self._windows = self._build_index()

    # -- episode-level bookkeeping -------------------------------------------

    def _read_headers(self):
        """(ep_start, ep_len, ep_layout, n_frames, ep_success) per shard."""
        headers = []
        for path in self.shards:
            with np.load(path) as z:
                n_eps = len(z["ep_len"])
                layout = (z["ep_layout"].tolist() if "ep_layout" in z.files
                          else [""] * n_eps)
                # Shards written before ep_success existed cannot be filtered.
                # Marking them all True keeps them usable rather than silently
                # dropping the whole set; only_success reports the shortfall.
                success = (z["ep_success"].astype(bool) if "ep_success" in z.files
                           else np.ones(n_eps, dtype=bool))
                headers.append((z["ep_start"].astype(np.int64),
                                z["ep_len"].astype(np.int64),
                                layout,
                                int(len(z["image"])),
                                success))
        return headers

    def _select_episodes(self, split, val_frac, seed, stratify):
        """List of (shard_idx, ep_idx) kept by this view."""
        self._headers = self._read_headers()
        all_eps = [(si, ei)
                   for si, (_, ep_len, _, _, _) in enumerate(self._headers)
                   for ei in range(len(ep_len))]
        if self.only_success:
            n_before = len(all_eps)
            all_eps = [(si, ei) for si, ei in all_eps
                       if bool(self._headers[si][4][ei])]
            if not all_eps:
                raise ValueError(
                    f"only_success=True kept 0 of {n_before} episodes under "
                    f"{self.root}. Either the pilot never completed a course "
                    f"or these shards predate ep_success -- recollect.")
            self.n_dropped = n_before - len(all_eps)
        if split is None:
            return all_eps
        if not 0.0 < val_frac < 1.0:
            raise ValueError(f"val_frac must be in (0,1), got {val_frac}")

        rng = np.random.default_rng(seed)
        # Stratify so val is not one layout's worth of episodes. With a handful
        # of episodes the val number is noisy either way -- this only stops it
        # being systematically noisy.
        groups: dict = {}
        for si, ei in all_eps:
            key = self._headers[si][2][ei] if stratify else ""
            groups.setdefault(key, []).append((si, ei))

        val, train = [], []
        for key in sorted(groups):
            eps = groups[key]
            perm = rng.permutation(len(eps))
            n_val = int(round(val_frac * len(eps)))
            n_val = min(max(n_val, 1), len(eps) - 1) if len(eps) > 1 else 0
            val.extend(eps[i] for i in perm[:n_val])
            train.extend(eps[i] for i in perm[n_val:])
        chosen = sorted(val if split == "val" else train)
        if not chosen:
            raise ValueError(
                f"split={split!r} is empty: {len(all_eps)} episode(s) across "
                f"{len(self.shards)} shard(s) at val_frac={val_frac}. Collect "
                f"more episodes or set split=None.")
        return chosen

    def _shard_windows(self, ep_start, ep_len, n_frames, ep_ids=None):
        """Window start offsets that stay inside a single episode.

        Offsets are shard-local. The bound check is the guard against ever
        again mixing an index from one shard with another shard's arrays.
        """
        out = []
        ids = ep_ids if ep_ids is not None else range(len(ep_len))
        for ei, s, ln in zip(ids, ep_start, ep_len):
            s, ln = int(s), int(ln)
            if s + ln > n_frames:
                raise ValueError(
                    f"episode {ei} spans [{s},{s + ln}) but its shard holds "
                    f"only {n_frames} frames -- shard header is inconsistent, "
                    f"or a global index was passed where a shard-local one "
                    f"was expected")
            if ln < self.seq_len:
                continue
            out.extend(range(s, s + ln - self.seq_len + 1, self.stride))
        return out

    def _build_index(self):
        """Per-shard window lists for the episodes this view kept."""
        windows = [[] for _ in self.shards]
        for si, ei in self.episodes:
            ep_start, ep_len, _, n_frames, _ = self._headers[si]
            windows[si].extend(self._shard_windows(
                ep_start[ei:ei + 1], ep_len[ei:ei + 1], n_frames, ep_ids=(ei,)))
        return windows

    # -- iteration ------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(w) for w in self._windows)

    def n_episodes(self) -> int:
        return len(self.episodes)

    def iter_batches(self, batch_size: int, rng: np.random.Generator,
                     shuffle: bool = True,
                     drop_last: bool = True) -> Iterator[dict]:
        """Yield (B, seq_len, ...) batches.

        `drop_last=False` also keeps shards holding fewer than `batch_size`
        windows. The default drops them, which for a small val split can throw
        away every shard and report a loss of 0.0 -- pass False for evaluation.
        """
        order = (rng.permutation(len(self.shards)) if shuffle
                 else np.arange(len(self.shards)))
        for si in order:
            starts = list(self._windows[si])
            if not starts or (drop_last and len(starts) < batch_size):
                continue
            with np.load(self.shards[si]) as z:
                data = {k: z[k] for k in self.keys}
            if shuffle:
                rng.shuffle(starts)
            stop = (len(starts) - batch_size + 1 if drop_last else len(starts))
            for b in range(0, stop, batch_size):
                idx = starts[b:b + batch_size]
                yield {k: np.stack([v[s:s + self.seq_len] for s in idx])
                       for k, v in data.items()}


class FrameDataset:
    """Flat frame view over the same shards, for the SeVAE (no time axis)."""

    def __init__(self, root, keys: Optional[Sequence[str]] = None):
        self.root = Path(root)
        self.shards = sorted(self.root.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards under {self.root}")
        self.keys = tuple(keys) if keys else (
            "image", "image_gt", "seg", "depth_m")

    def iter_batches(self, batch_size: int, rng: np.random.Generator,
                     shuffle: bool = True) -> Iterator[dict]:
        for path in (rng.permutation(self.shards) if shuffle else self.shards):
            with np.load(path) as z:
                data = {k: z[k] for k in self.keys}
            n = len(next(iter(data.values())))
            order = rng.permutation(n) if shuffle else np.arange(n)
            for s in range(0, n - batch_size + 1, batch_size):
                idx = order[s:s + batch_size]
                yield {k: v[idx] for k, v in data.items()}

    def class_counts(self, n_classes: int) -> np.ndarray:
        counts = np.zeros(n_classes, dtype=np.int64)
        for path in self.shards:
            with np.load(path) as z:
                counts += np.bincount(z["seg"].ravel(), minlength=n_classes)
        return counts


def summarize(root) -> dict:
    """Shape/consistency report -- the assertions the notebook runs after collect."""
    root = Path(root)
    shards = sorted(root.glob("*.npz"))
    total_frames = total_eps = total_ok = 0
    layouts, bad = [], []
    by_source: dict = {}
    for path in shards:
        with np.load(path) as z:
            n = len(z["image"])
            ep_start, ep_len = z["ep_start"], z["ep_len"]
            if int(ep_len.sum()) != n:
                bad.append(f"{path.name}: ep_len sums to {ep_len.sum()}, N={n}")
            expected = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
            if not np.array_equal(ep_start, expected):
                bad.append(f"{path.name}: ep_start does not match cumsum(ep_len)")
            total_frames += n
            total_eps += len(ep_len)
            layouts.extend(z["ep_layout"].tolist())
            # Pre-merge shards predate ep_source; treat them as scripted.
            src = (z["ep_source"].tolist() if "ep_source" in z.files
                   else ["scripted"] * len(ep_len))
            for s_, ln in zip(src, ep_len):
                by_source[s_] = by_source.get(s_, 0) + int(ln)
            if "ep_success" in z.files:
                total_ok += int(z["ep_success"].astype(bool).sum())
    # The completion rate is the one number that says whether the pilot is
    # worth cloning at all, and it used to be printed by collect.py and thrown
    # away. BC trains on this subset, so a low rate here caps everything
    # downstream -- it belongs in the report, not in a scrollback buffer.
    if total_eps and total_ok == 0:
        bad.append("no episode is marked ep_success: either the pilot never "
                   "completed a course, or these shards predate the flag")
    return {"shards": len(shards), "frames": total_frames, "episodes": total_eps,
            "completed": total_ok,
            "completed_frac": round(total_ok / max(1, total_eps), 3),
            "unique_layouts": len(set(layouts)), "frames_by_source": by_source,
            "problems": bad}