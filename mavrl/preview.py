#!/usr/bin/env python3
"""Fly a checkpoint and write video of it, third-person plus what it sees.

`evaluate.py` answers "what fraction succeeded"; this answers "what is it doing
wrong". Numbers cannot distinguish a policy that dives under a red bar from one
that never saw the bar at all -- the video can, because the onboard panels sit
next to the third-person view frame for frame.

Works on any checkpoint that stores a `policy` state dict: `ckpt/bc_init.pt`
straight out of behaviour cloning, or `runs/course/final.pt` after PPO.

    python -m mavrl.preview --model ckpt/bc_init.pt \
        --memory-type attention --mem-tokens 4 --episodes 4 --stations 2

Headless by design -- it writes mp4 files rather than opening a window, so it
runs on the same box that trains. Set MUJOCO_GL=osmesa if EGL is unavailable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_aviary import CourseAviary                     # noqa: E402
from mavrl.course_world import sample_layout                     # noqa: E402
from mavrl.gui import depth_rgb, seg_rgb                         # noqa: E402
from mavrl.policy import (                                       # noqa: E402
    MavrlActorCritic, load_policy_state, obs_to_tensors,
)
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402

#: All divisible by 16, so ffmpeg encodes without silently rescaling.
TP_W, TP_H = 960, 544        # third-person view
PANEL = 320                  # each onboard panel; three of them span TP_W


def _nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize. Deliberately not interpolating: the seg panel
    holds class ids, and a blended id is a different class."""
    ys = np.arange(h) * img.shape[0] // h
    xs = np.arange(w) * img.shape[1] // w
    return img[ys][:, xs]


def _hud(frame: np.ndarray, lines, color=(255, 255, 255)) -> np.ndarray:
    """Burn text into the top-left corner, if Pillow is around."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        draw.text((10, 8 + 15 * i), line, fill=color)
    return np.asarray(im)


def _border(frame: np.ndarray, color, width: int = 6) -> np.ndarray:
    frame[:width] = frame[-width:] = color
    frame[:, :width] = frame[:, -width:] = color
    return frame


def compose(third: np.ndarray, obs: dict, lines) -> np.ndarray:
    """Third-person on top, the policy's own RGB / depth / seg underneath."""
    rgb = obs["image"][..., :3]
    if "depth_m" in obs:
        dep = depth_rgb(obs["depth_m"])
    else:
        # No collect_mode: recover metres from the packed uint8 channel.
        dep = depth_rgb(obs["image"][..., 3].astype(np.float32)
                        / 255.0 * C.DEPTH_MAX)
    panels = [rgb, dep]
    panels.append(seg_rgb(obs["seg"]) if "seg" in obs
                  else np.zeros_like(rgb))
    row = np.concatenate([_nearest(p, PANEL, PANEL) for p in panels], axis=1)
    if row.shape[1] != TP_W:                      # keep the mosaic rectangular
        row = _nearest(row, PANEL, TP_W)
    return _hud(np.concatenate([_nearest(third, TP_H, TP_W), row], axis=0),
                lines)


@torch.no_grad()
def fly(env, policy, device, renderer, camera, mujoco,
        deterministic: bool = True, max_steps: int = 2000):
    """One episode; returns (frames, info)."""
    obs, _ = env.reset()
    state = policy.initial_state(1, device)
    frames = []
    for step in range(max_steps):
        batched = {k: np.asarray(v)[None] for k, v in obs.items()
                   if k in ("image", "proprio")}
        image, proprio = obs_to_tensors(batched, device)
        action, _, _, state = policy.act(image, proprio, state,
                                         deterministic=deterministic)
        a = action.cpu().numpy()[0].astype(np.float32)

        pos = env.pos[0]
        # Look ahead of the drone, exactly as gui.ViewerWindow does, so the gate
        # being approached stays in frame instead of the one just cleared.
        camera.lookat[:] = [0.0, pos[1] + 1.5, max(1.2, pos[2])]
        renderer.update_scene(env.data, camera=camera)
        third = renderer.render()

        speed = float(np.linalg.norm(env.vel[0]))
        frames.append(compose(third, obs, [
            f"step {step:3d}   x={pos[0]:+.2f} y={pos[1]:+.2f} z={pos[2]:.2f}",
            f"speed {speed:.2f} m/s   gates {env.gates_cleared}/{env.gates.n_gates}",
            f"action a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) yaw={a[3]:+.2f}",
        ]))

        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return frames, info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("ckpt/bc_init.pt"))
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention", "none"))
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--stations", type=int, default=2,
                   help="bars in the course; 0 is entry->exit only")
    p.add_argument("--split", default="train", choices=("train", "eval", "all"),
                   help="'eval' uses the held-out bar heights")
    p.add_argument("--out", type=Path, default=Path("preview"))
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    import imageio.v2 as imageio
    import mujoco

    device = torch.device(args.device)
    memory_type = "lstm" if args.memory_type == "none" else args.memory_type
    policy = MavrlActorCritic(memory_type=memory_type,
                              mem_tokens=args.mem_tokens).to(device)
    ckpt = torch.load(args.model, map_location=device)
    load_policy_state(policy, ckpt.get("policy", ckpt.get("actor")),
                      ckpt.get("aux_segments"))
    policy.eval()
    if args.memory_type == "none":
        policy.memory_type = "none"
    print(f"loaded {args.model} "
          f"({ckpt.get('memory_type', '?')}, aux={ckpt.get('aux_segments')})")

    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # collect_mode gives clean seg and metric depth for the panels; the encoder
    # still receives the noisy `image`, so this does not flatter the policy.
    env = CourseAviary(layout=sample_layout(rng, args.stations, args.split),
                       seed=args.seed, noise=noise, collect_mode=True)
    outcomes = []
    try:
        for ep in range(args.episodes):
            env.set_layout(sample_layout(rng, args.stations, args.split))
            # set_layout rebuilds MjModel, so the renderer must be rebuilt too.
            renderer = mujoco.Renderer(env.model, height=TP_H, width=TP_W)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance, camera.elevation, camera.azimuth = 6.5, -14.0, 90.0

            frames, info = fly(env, policy, device, renderer, camera, mujoco,
                               deterministic=not args.stochastic)
            renderer.close()

            ok = bool(info["is_success"])
            reason = info.get("terminal_reason") or "timeout"
            tint = (60, 200, 90) if ok else (220, 70, 70)
            frames = [_border(f, tint) for f in frames]

            path = args.out / f"ep{ep:02d}_{'success' if ok else reason}.mp4"
            imageio.mimsave(path, frames, fps=C.POLICY_FREQ, macro_block_size=1)
            outcomes.append((ok, reason, info))
            print(f"ep{ep:02d}  {'PASS' if ok else 'FAIL':4s}  {reason:16s} "
                  f"gates {info['gates_cleared']}/{info['n_gates']}  "
                  f"agv {info['agv']:.2f} m/s  {len(frames)} frames -> {path}")
    finally:
        env.close()

    n_ok = sum(ok for ok, _, _ in outcomes)
    print(f"\n{n_ok}/{len(outcomes)} succeeded on {args.stations} stations "
          f"({args.split} heights). Videos in {args.out}/")
    print("Watch the depth panel as a bar comes up: if the policy climbs before "
          "the bar is resolvable there, it is using memory, not reaction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
