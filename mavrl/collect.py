#!/usr/bin/env python3
"""Data collection with a scripted privileged pilot.

Two behaviours this needs that ordinary rollout collection does not:

* **Episodes do not end on a missed gate.** The SeVAE has to see failure states
  -- approaching a bar on the wrong side, drifting toward the decoy window -- or
  the latent space only ever models clean traversals. Only collisions and the
  time limit end an episode here.
* **A different layout every episode**, drawn across all curriculum stages and
  every bar height (including the eval-only ones, since the encoder is allowed
  to have seen a height the *policy* has not).

Acceleration labels are produced **natively** by a cascaded controller
(P on position error -> velocity target, P on velocity error -> acceleration).
Differentiating a velocity command to get acceleration would amplify controller
noise straight into the BC targets.

Headless collection parallelises with `--workers N`: N processes, each with its
own MuJoCo/GL context and its own `ShardWriter`, all writing into the same
directory. Frame data never crosses a process boundary -- at ~115 kB a frame,
piping episodes back to a parent would cost more than rendering them.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")   # headless GL; export to override
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_aviary import CourseAviary                     # noqa: E402
from mavrl.course_gates import (                                 # noqa: E402
    TRAVEL_SIGN, BarGate, BarSide, WallGate,
)
from mavrl.course_world import (                                 # noqa: E402
    RED_Z_HI, RED_Z_LO, STAGE_STATIONS, YAW_DOWN_COURSE, sample_layout,
)
# The pilot itself lives in mavrl.pilot -- the env imports it too, and collect.py
# imports the env, so it cannot live here. Re-exported so existing callers and
# `python -m mavrl.collect` keep working unchanged.
from mavrl.pilot import APPROACH_DIST, CascadedPilot, gate_targets  # noqa: E402,F401
from mavrl.dataset import ShardWriter, summarize                 # noqa: E402
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402


FRAME_KEYS = ("image", "image_gt", "seg", "depth_m", "proprio", "action",
              "gate_flags")


def new_frames() -> dict:
    """Empty per-episode buffer. Shared with mavrl.teleop so the two collectors
    cannot drift into writing different columns."""
    return {k: [] for k in FRAME_KEYS}


def record_frame(frames: dict, obs: dict, action, env) -> None:
    frames["image"].append(obs["image"])
    frames["image_gt"].append(obs["image_gt"])
    frames["seg"].append(obs["seg"])
    frames["depth_m"].append(obs["depth_m"].astype(np.float16))
    frames["proprio"].append(obs["proprio"])
    frames["action"].append(np.asarray(action, dtype=np.float32))
    frames["gate_flags"].append(np.uint8(env.gates_cleared))


def stack_frames(frames: dict) -> dict:
    return {k: np.asarray(v) for k, v in frames.items()}


def run_episode(env, pilot, rng, noise_std: float, max_steps: int,
                on_step=None):
    frames = new_frames()
    obs, info = env.reset()
    for _ in range(max_steps):
        clean = pilot(env)
        record_frame(frames, obs, clean, env)

        # DAgger-style: execute a perturbed action, record the clean label, so
        # the dataset covers states a slightly-wrong policy would reach.
        noisy = np.clip(clean + rng.normal(0.0, noise_std, 4), -1.0, 1.0)
        obs, _, term, trunc, info = env.step(noisy.astype(np.float32))
        if on_step is not None and on_step(env, obs, info) is False:
            break
        if term or trunc:
            break
    return stack_frames(frames), info


def collect_worker(args, worker_id: int, n_episodes: int,
                   progress=None) -> None:
    """One process's share of the collection. Writes its own shards.

    Deliberately no IPC of frame data: one frame is ~115 kB, so piping episodes
    back to a parent would spend more time pickling than rendering. Each worker
    owns a `ShardWriter` with its own filename prefix and writes straight to the
    shared output directory; `mavrl.dataset` globs `*.npz` so the result reads
    back as one dataset with no merge step.
    """
    seed = args.seed + 1_000 * worker_id
    rng = np.random.default_rng(seed)
    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())

    layout = sample_layout(rng, args.max_stations, args.split)
    env = CourseAviary(layout=layout, seed=seed, noise=noise, collect_mode=True)
    pilot = CascadedPilot()
    prefix = (args.prefix if args.workers == 1
              else f"{args.prefix}_w{worker_id:02d}")
    writer = ShardWriter(args.out, args.episodes_per_shard, prefix=prefix)

    viewer = feed = on_step = None
    if args.gui:
        from mavrl.gui import FeedWindow, ViewerWindow, hud_lines
        viewer, feed = ViewerWindow(env), FeedWindow("mavrl -- scripted pilot")

        def on_step(env, obs, info):
            feed.poll()
            feed.draw(obs, hud_lines(env, "scripted pilot (read-only)"))
            feed.tick()
            viewer.sync(env)
            return feed.alive and viewer.running

    tag = "" if args.workers == 1 else f"[w{worker_id}] "
    kept = completed = 0
    for ep in range(n_episodes):
        n_stations = int(rng.integers(0, args.max_stations + 1))
        layout = sample_layout(rng, n_stations, args.split)
        env.set_layout(layout)
        if viewer is not None:
            viewer.attach(env)          # set_layout built a new MjModel

        max_steps = int(C.t_max(n_stations) * C.POLICY_FREQ) + 5
        frames, info = run_episode(env, pilot, rng, args.noise, max_steps,
                                   on_step=on_step)
        if len(frames["image"]) == 0:
            continue

        success = bool(info.get("is_success", False))
        writer.add_episode(frames, layout.describe(), source="scripted",
                           success=success)
        kept += 1
        completed += int(success)
        if progress is not None:
            progress.put(1)
        if (ep + 1) % 50 == 0 or args.gui:
            print(f"{tag}[{ep + 1}/{n_episodes}] kept={kept} "
                  f"completed={completed} ({completed / max(1, kept):.0%}) "
                  f"last={layout.describe()}", flush=True)
        if args.gui and not (feed.alive and viewer.running):
            print("window closed -- stopping and saving")
            break

    writer.close()
    if viewer is not None:
        viewer.close()
        feed.close()
    env.close()
    print(f"{tag}done: kept {kept}, completed {completed} "
          f"({completed / max(1, kept):.0%})", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel collector processes. Each renders in its own "
                        "MuJoCo/GL context and writes its own shards, so this "
                        "scales until the GPU saturates. Incompatible with "
                        "--gui.")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.06,
                   help="DAgger action noise std")
    p.add_argument("--sensor-noise", type=float, default=1.0,
                   help="sensor noise scale (0 disables)")
    p.add_argument("--episodes-per-shard", type=int, default=50)
    p.add_argument("--split", default="all",
                   choices=("train", "eval", "all"),
                   help="bar-height split; 'all' lets the encoder see held-out "
                        "heights the policy never trains on")
    p.add_argument("--max-stations", type=int, default=max(STAGE_STATIONS))
    p.add_argument("--gui", action="store_true",
                   help="watch it fly: 3-D viewer + live RGB/depth/seg feed. "
                        "Slower, and it needs a display -- use MUJOCO_GL=glfw.")
    p.add_argument("--prefix", default="shard",
                   help="shard filename prefix; keep distinct per source so "
                        "merged directories stay traceable")
    args = p.parse_args(argv)

    if args.workers > 1 and args.gui:
        raise SystemExit("--gui shows one drone; use --workers 1 with it")
    args.workers = max(1, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.workers == 1:
        collect_worker(args, 0, args.episodes)
    else:
        # spawn, not fork: a forked child inherits the parent's GL/EGL state and
        # MuJoCo renderers do not survive that. spawn is also the only option on
        # Windows, so this is one code path rather than two.
        ctx = mp.get_context("spawn")
        share = args.episodes // args.workers
        extra = args.episodes % args.workers
        procs = []
        for w in range(args.workers):
            n = share + (1 if w < extra else 0)
            if n == 0:
                continue
            pr = ctx.Process(target=collect_worker, args=(args, w, n),
                             daemon=False)
            pr.start()
            procs.append(pr)
        print(f"{len(procs)} workers x ~{share} episodes", flush=True)
        for pr in procs:
            pr.join()
        failed = [pr.exitcode for pr in procs if pr.exitcode != 0]
        if failed:
            print(f"WARNING: {len(failed)} worker(s) exited non-zero: {failed}")

    dt = time.time() - t0
    report = summarize(args.out)
    print(f"\ncollected in {dt / 60:.1f} min "
          f"({report['episodes'] / max(dt, 1e-9) * 60:.1f} episodes/min)")
    print("dataset:", report)
    if report["problems"]:
        print("PROBLEMS:", *report["problems"], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
