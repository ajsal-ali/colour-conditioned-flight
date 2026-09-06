#!/usr/bin/env python3
"""Fly the course yourself and record it.

    MUJOCO_GL=glfw python -m mavrl.teleop --episodes 30 --out data_manual

Two windows open: a third-person MuJoCo viewer, and a feed window showing the
RGB / depth / segmentation the encoder actually receives. **Keys only register
while the feed window is focused.**

    W / S            forward / back        (body +x / -x)
    A / D            left / right          (body +y / -y)
    SPACE / L-SHIFT  up / down
    Q / E            yaw left / right      (only with --free-yaw)
    [ / ]            speed scale down / up
    ENTER            end this episode and KEEP it
    BACKSPACE        end this episode and THROW IT AWAY
    P                pause
    ESC              quit (everything kept so far is saved)

Hold-to-fly, as asked: while a key is held that axis commands +-v_max * scale,
and the instant it is released that axis's target goes to zero. That is a
velocity command, not a force -- the drone decelerates to a stop rather than
drifting.

Why the recorded label is still an acceleration
-----------------------------------------------
The env's action space is body acceleration + yaw rate, and the scripted
collector records exactly that. If teleop recorded raw velocities the two
datasets could not be concatenated -- the `action` column would mean different
things in different rows, and BC would fit a blend of the two. So the keyboard
sets a *velocity target* and this converts it the same way `CascadedPilot` does:

    a_body = (v_target_body - env.v_cmd_body) / POLICY_DT

which is the acceleration that lands the env's velocity integrator exactly on
the target. Identical semantics, one dataset.

Worth knowing before you rely on this for BC
--------------------------------------------
rl/pilot_record.py's docstring is still right: hold-to-fly demos are not a clean
function of the camera image (the action stays constant while a key is held,
regardless of what comes into view), so BC on human data tends to regress toward
the mean. For the **SeVAE** the opposite holds -- human flying visits stranger
states than any scripted controller and that is exactly what the encoder wants.
Mix accordingly; `--bc-safe` is not a thing this offers, but `ep_source` is
recorded on every episode so you can filter later.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")     # teleop needs a display anyway
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.collect import new_frames, record_frame, stack_frames  # noqa: E402
from mavrl.course_aviary import CourseAviary                     # noqa: E402
from mavrl.course_world import (                                 # noqa: E402
    STAGE_STATIONS, YAW_DOWN_COURSE, sample_layout,
)
from mavrl.dataset import ShardWriter, summarize                 # noqa: E402
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402


class KeyboardPilot:
    """Held keys -> body velocity target -> acceleration action."""

    def __init__(self, pg, speed: float = 0.6, lock_yaw: bool = True,
                 kp_yaw: float = 2.0):
        self.pg = pg
        self.speed = float(np.clip(speed, 0.05, 1.0))
        self.lock_yaw = lock_yaw
        self.kp_yaw = kp_yaw
        # body x forward, body y left, body z up
        self.axes = (
            (0, pg.K_w, pg.K_s),
            (1, pg.K_a, pg.K_d),
            (2, pg.K_SPACE, pg.K_LSHIFT),
        )

    def velocity_target(self, held) -> np.ndarray:
        """Non-zero only on axes whose key is down *right now*."""
        v = np.zeros(3)
        v_max = np.array(C.V_MAX)
        for axis, pos_key, neg_key in self.axes:
            d = float(held[pos_key]) - float(held[neg_key])
            v[axis] = d * v_max[axis] * self.speed
        return v

    def __call__(self, env, held) -> np.ndarray:
        v_des_body = self.velocity_target(held)

        # Same conversion as CascadedPilot: ask for the acceleration that puts
        # the env's velocity integrator on the target in one policy step.
        a_body = (v_des_body - env.v_cmd_body) / C.POLICY_DT

        action = np.empty(4, dtype=np.float32)
        action[:3] = np.clip(a_body / np.array(C.A_MAX), -1.0, 1.0)

        if self.lock_yaw:
            yaw = env.rpy[0, 2]
            err = math.atan2(math.sin(YAW_DOWN_COURSE - yaw),
                             math.cos(YAW_DOWN_COURSE - yaw))
            action[3] = np.clip(self.kp_yaw * err / C.YAW_RATE_MAX, -1.0, 1.0)
        else:
            action[3] = float(held[self.pg.K_q]) - float(held[self.pg.K_e])
        return action


def _status(pilot, paused: bool) -> str:
    yaw = "locked" if pilot.lock_yaw else "manual (Q/E)"
    return (f"{'PAUSED  ' if paused else ''}speed {pilot.speed:.2f}  "
            f"yaw {yaw}   [ENTER keep] [BKSP discard] [P pause] [ESC quit]")


def fly_episode(env, pilot, feed, viewer, max_steps: int):
    """One manual episode. Returns (frames, info, verdict).

    verdict: "keep" | "discard" | "quit"
    """
    pg = feed.pygame
    frames = new_frames()
    obs, info = env.reset()
    paused = False
    verdict = "keep"

    for _ in range(max_steps):
        held, tapped = feed.poll()
        if not feed.alive or not viewer.running or pg.K_ESCAPE in tapped:
            return stack_frames(frames), info, "quit"
        if pg.K_RETURN in tapped or pg.K_KP_ENTER in tapped:
            break
        if pg.K_BACKSPACE in tapped:
            verdict = "discard"
            break
        if pg.K_p in tapped:
            paused = not paused
        if pg.K_LEFTBRACKET in tapped:
            pilot.speed = float(np.clip(pilot.speed - 0.1, 0.05, 1.0))
        if pg.K_RIGHTBRACKET in tapped:
            pilot.speed = float(np.clip(pilot.speed + 0.1, 0.05, 1.0))

        if paused:
            feed.draw(obs, _hud(env, pilot, paused, len(frames["image"])))
            feed.tick()
            viewer.sync(env)
            continue

        action = pilot(env, held)
        record_frame(frames, obs, action, env)
        obs, _, term, trunc, info = env.step(action)

        feed.draw(obs, _hud(env, pilot, paused, len(frames["image"])))
        feed.tick()
        viewer.sync(env)
        if term or trunc:
            break

    return stack_frames(frames), info, verdict


def _hud(env, pilot, paused: bool, n: int):
    from mavrl.gui import hud_lines
    lines = hud_lines(env, _status(pilot, paused))
    v = env.v_cmd_body
    lines.append(f"v_cmd body ({v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}) m/s   "
                 f"frames {n}")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("data_manual"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=0.6,
                   help="fraction of V_MAX a held key commands")
    p.add_argument("--free-yaw", action="store_true",
                   help="Q/E control yaw. Off by default so manual data matches "
                        "the scripted set, which always faces down-corridor")
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--episodes-per-shard", type=int, default=10)
    p.add_argument("--split", default="all", choices=("train", "eval", "all"))
    p.add_argument("--stations", type=int, default=None,
                   help="fix the bar count instead of sampling it")
    p.add_argument("--max-stations", type=int, default=max(STAGE_STATIONS))
    p.add_argument("--prefix", default="manual")
    p.add_argument("--no-seg", action="store_true",
                   help="hide the segmentation panel")
    args = p.parse_args(argv)

    from mavrl.gui import FeedWindow, ViewerWindow

    rng = np.random.default_rng(args.seed)
    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())

    layout = sample_layout(rng, args.max_stations, args.split)
    env = CourseAviary(layout=layout, seed=args.seed, noise=noise,
                       collect_mode=True)
    feed = FeedWindow("mavrl -- manual teleop", show_seg=not args.no_seg)
    viewer = ViewerWindow(env)
    pilot = KeyboardPilot(feed.pygame, args.speed, lock_yaw=not args.free_yaw)
    writer = ShardWriter(args.out, args.episodes_per_shard, prefix=args.prefix)

    print(__doc__.split("Why the recorded label")[0])

    kept = discarded = completed = 0
    for ep in range(args.episodes):
        n_stations = (args.stations if args.stations is not None
                      else int(rng.integers(0, args.max_stations + 1)))
        layout = sample_layout(rng, n_stations, args.split)
        env.set_layout(layout)
        viewer.attach(env)               # set_layout rebuilt the model
        if not viewer.running or not feed.alive:
            break

        print(f"\n[{ep + 1}/{args.episodes}] {layout.describe()}")
        max_steps = int(C.t_max(n_stations) * C.POLICY_FREQ) + 5
        frames, info, verdict = fly_episode(env, pilot, feed, viewer, max_steps)

        if verdict == "discard" or len(frames["image"]) == 0:
            discarded += 1
            print("  discarded")
        else:
            writer.add_episode(frames, layout.describe(), source="manual")
            kept += 1
            completed += int(info.get("is_success", False))
            print(f"  kept {len(frames['image'])} frames | "
                  f"gates {info['gates_cleared']}/{info['n_gates']} | "
                  f"{info.get('terminal_reason') or 'timeout'}")
        if verdict == "quit":
            print("quit requested")
            break

    writer.close()
    viewer.close()
    feed.close()
    env.close()

    print(f"\nkept {kept} episodes, discarded {discarded}, "
          f"completed {completed}")
    report = summarize(args.out)
    print("dataset:", report)
    if report["problems"]:
        print("PROBLEMS:", *report["problems"], sep="\n  ")
        return 1
    print(f"\nMerge with the scripted set:\n"
          f"  python -m mavrl.merge_data --out data_all data {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
