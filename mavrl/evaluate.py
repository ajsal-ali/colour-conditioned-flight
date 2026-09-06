#!/usr/bin/env python3
"""Evaluation: success rate and AGV per stage, plus held-out-height generalization.

The held-out split is the point of this file. Training only ever sees red bars at
{1.20, 1.98} and blue at {0.40, 1.20}; evaluating on red 1.60 / blue 0.80 asks
whether the policy learned "red means above" or just memorized four heights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_aviary import CourseAviary                     # noqa: E402
from mavrl.course_world import STAGE_STATIONS, sample_layout     # noqa: E402
from mavrl.policy import (                                       # noqa: E402
    MavrlActorCritic, load_policy_state, obs_to_tensors,
)
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402


@torch.no_grad()
def rollout(env, policy, device, deterministic: bool = True,
            record: bool = False):
    obs, _ = env.reset()
    state = policy.initial_state(1, device)
    frames = []
    while True:
        batched = {k: np.asarray(v)[None] for k, v in obs.items()
                   if k in ("image", "proprio")}
        image, proprio = obs_to_tensors(batched, device)
        action, _, _, state = policy.act(image, proprio, state,
                                         deterministic=deterministic)
        if record:
            frames.append(obs["image"][..., :3].copy())
        obs, _, term, trunc, info = env.step(
            action.cpu().numpy()[0].astype(np.float32))
        if term or trunc:
            return info, frames


def evaluate(policy, device, n_episodes: int = 25, split: str = "train",
             stages=STAGE_STATIONS, seed: int = 0, noise=None):
    rng = np.random.default_rng(seed)
    noise = noise if noise is not None else NoiseConfig()
    results = {}
    env = CourseAviary(layout=sample_layout(rng, 0, split), seed=seed,
                       noise=noise)
    try:
        for stage, n_stations in enumerate(stages):
            succ, agv, reasons = [], [], {}
            for _ in range(n_episodes):
                env.set_layout(sample_layout(rng, n_stations, split))
                info, _ = rollout(env, policy, device)
                succ.append(bool(info["is_success"]))
                agv.append(float(info["agv"]))
                r = info.get("terminal_reason") or "timeout"
                reasons[r] = reasons.get(r, 0) + 1
            results[f"stage{stage}"] = {
                "n_stations": n_stations,
                "success": float(np.mean(succ)),
                "agv": float(np.mean(agv)),
                "reasons": reasons,
            }
    finally:
        env.close()
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention", "none"))
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    memory_type = "lstm" if args.memory_type == "none" else args.memory_type
    policy = MavrlActorCritic(memory_type=memory_type,
                              mem_tokens=args.mem_tokens).to(device)
    ckpt = torch.load(args.model, map_location=device)
    load_policy_state(policy, ckpt.get("policy", ckpt.get("actor")),
                      ckpt.get("aux_segments"))
    policy.eval()

    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())

    report = {}
    for split in ("train", "eval"):
        report[split] = evaluate(policy, device, args.episodes, split,
                                 seed=args.seed, noise=noise)
        print(f"\n--- {split} heights ---")
        for stage, row in report[split].items():
            print(f"{stage} ({row['n_stations']} bars): "
                  f"success={row['success']:.0%} agv={row['agv']:.2f} m/s "
                  f"{row['reasons']}")

    gaps = [report["train"][k]["success"] - report["eval"][k]["success"]
            for k in report["train"]]
    print(f"\nmean train-vs-held-out success gap: {np.mean(gaps):+.1%}")
    print("A large positive gap means the heights were memorized, not the rule.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
