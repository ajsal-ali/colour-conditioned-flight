#!/usr/bin/env python3
"""Stages 1 and 4: PPO on the course.

Stage 1  --stage 0 --frozen-encoder     random frozen encoder, entry->exit only.
                                        Produces the policy that collects data.
Stage 4  --curriculum --init ckpt/...   full curriculum, warm-started from BC.

The curriculum gives **each render worker its own layout**, swapped at a rollout
boundary. Switching mid-rollout would invalidate the value estimates already
collected against the old geometry; switching all of them to the *same* course
was what let the policy memorise one colour order instead of learning the rule
(see mavrl.curriculum).

A decaying fraction of the envs is flown by the scripted pilot rather than the
policy, with those transitions going into the same rollout buffer under a
self-imitation objective -- `--expert-frac`, see mavrl.ppo.

Two artefacts land in `--out` as the run goes: `log.jsonl`, one row per rollout,
and `curves.png`, that log plotted. The PNG is rewritten every `--plot-every`
rollouts and again on exit, so a run on a remote box can be watched by pulling
one file -- no TensorBoard process to keep alive alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")   # headless GL; export to override
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_world import STAGE_STATIONS                    # noqa: E402
from mavrl.curriculum import CourseSampler, broadcast_layout     # noqa: E402
from mavrl.policy import MavrlActorCritic, load_policy_state     # noqa: E402
from mavrl.ppo import PPOConfig, RecurrentPPO                    # noqa: E402
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402
from mavrl.vecenv import build_venv, layout_slots                # noqa: E402
from mavrl.visualize import save_training_curves                 # noqa: E402


class WarmStartFreeze:
    """Freeze the actor while the critic catches up.

    Ported from rl/train_window.py:95. After BC the actor is good and the value
    function is random; letting them train together immediately lets garbage
    advantages destroy a policy that already works.
    """

    def __init__(self, policy: MavrlActorCritic, steps: int):
        self.policy = policy
        self.steps = steps
        self.released = steps <= 0
        if not self.released:
            self._set(False)

    def _set(self, on: bool) -> None:
        for module in (self.policy.pi,):
            for prm in module.parameters():
                prm.requires_grad = on
        self.policy.log_std.requires_grad = on

    def maybe_release(self, num_timesteps: int) -> bool:
        if not self.released and num_timesteps >= self.steps:
            self._set(True)
            self.released = True
            return True
        return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs2/course"))
    p.add_argument("--timesteps", type=int, default=500_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention", "none"))
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--stage", type=int, default=None,
                   help="fix the curriculum stage (no advancement)")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--split", default="train", choices=("train", "eval", "all"),
                   help="which bar heights to train on. 'train' holds red "
                        "3.520 / blue 1.760 back so evaluate.py can ask whether "
                        "the colour rule generalized; 'all' trains on all three "
                        "per colour and gives that up")
    p.add_argument("--frozen-encoder", action="store_true")
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--critic-warmup", type=int, default=30_000)
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--envs-per-batch", type=int, default=None,
                   help="minibatch size in envs (default n_envs//2). Each "
                        "minibatch decodes envs_per_batch * n_steps frames at "
                        "once, so this is the VRAM knob, not --n-envs")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005,
                   help="entropy bonus. Was hardcoded to 0, and the first long "
                        "run decayed monotonically 1.68 -> 0.21 with no "
                        "plateau, including through 200 rollouts of identical "
                        "failure -- by the time the layout changed there was no "
                        "exploration left to re-adapt with")
    p.add_argument("--expert-frac", type=float, default=0.25,
                   help="fraction of envs flown by the scripted pilot at step "
                        "0, decaying linearly to 0 over --expert-decay steps. "
                        "0 disables demonstrations entirely")
    p.add_argument("--expert-decay", type=int, default=1_000_000,
                   help="timesteps over which --expert-frac reaches zero")
    p.add_argument("--sil-coef", type=float, default=0.1,
                   help="weight on the self-imitation term. Watch `sil` "
                        "against `pg` in log.jsonl and scale it so the "
                        "demonstrations lead early without swamping PPO")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-subproc", action="store_true")
    p.add_argument("--render-workers", type=int, default=None,
                   help="M: processes sharing one GL context each "
                        "(default ceil(n_envs/4)). This is the VRAM knob for "
                        "rendering, as --envs-per-batch is for the backward pass")
    p.add_argument("--no-shared-render", action="store_true",
                   help="one GL context per env (SubprocVecEnv), the old path")
    p.add_argument("--plot-every", type=int, default=10,
                   help="rollouts between redraws of curves.png; 0 = only at the end")
    p.add_argument("--plot-smooth", type=int, default=15,
                   help="trailing-mean width, in rollouts, for the curves")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())

    start_stage = args.stage if args.stage is not None else 0
    # One layout per render worker: as many distinct courses inside a single
    # gradient update as the shared renderer allows. Has to be known before the
    # vec env exists, because the sampler draws the world the envs are built
    # with -- hence layout_slots() rather than reading it off `venv.groups`.
    n_layouts = layout_slots(args.n_envs, subproc=not args.no_subproc,
                             shared_render=not args.no_shared_render,
                             render_workers=args.render_workers)
    sampler = CourseSampler(seed=args.seed, split=args.split,
                            start_stage=start_stage, n_layouts=n_layouts)
    venv = build_venv(sampler.layout, args.n_envs, args.seed, noise,
                      subproc=not args.no_subproc,
                      shared_render=not args.no_shared_render,
                      render_workers=args.render_workers)
    # Every env was built with sampler.layout; hand each group its own now.
    if n_layouts > 1:
        broadcast_layout(venv, sampler.layouts)
    print(f"{args.n_envs} envs over "
          f"{getattr(venv, 'n_workers', args.n_envs)} render context(s), "
          f"{n_layouts} layout(s) at a time", flush=True)

    # "none" is the no-memory ablation: an LSTM with a 1-step window would still
    # be recurrent, so instead we keep the LSTM module but zero its state every
    # step, making the policy genuinely Markovian.
    memory_type = "lstm" if args.memory_type == "none" else args.memory_type
    policy = MavrlActorCritic(memory_type=memory_type,
                              mem_tokens=args.mem_tokens,
                              freeze_encoder=args.frozen_encoder)
    if args.memory_type == "none":
        policy.memory_type = "none"

    if args.init and args.init.exists():
        ckpt = torch.load(args.init, map_location="cpu")
        # load_policy_state is strict=False, so an attention checkpoint loaded
        # into an LSTM policy raises nothing: the 30 memory tensors land in
        # `unexpected`, the 4 LSTM ones in `missing`, and the run starts from a
        # random memory with a trained encoder and head. That is a day of V100
        # time producing a curve nobody can interpret, so refuse it here.
        saved = ckpt.get("memory_type")
        if saved is not None and saved != memory_type:
            raise SystemExit(
                f"{args.init} was trained with --memory-type {saved}, not "
                f"{args.memory_type}. The load would silently succeed with a "
                f"randomly initialized memory.")
        load_policy_state(policy, ckpt["policy"], ckpt.get("aux_segments"))
        print(f"warm-started from {args.init} "
              f"({ckpt.get('num_timesteps', 0)} steps, {saved})")

    per_batch = (args.envs_per_batch if args.envs_per_batch
                 else max(1, args.n_envs // 2))
    cfg = PPOConfig(n_steps=args.n_steps, learning_rate=args.lr,
                    envs_per_batch=min(per_batch, args.n_envs),
                    ent_coef=args.ent_coef,
                    expert_frac=args.expert_frac,
                    expert_decay=args.expert_decay,
                    sil_coef=args.sil_coef)
    algo = RecurrentPPO(policy, venv, cfg, device=args.device)

    warmup = WarmStartFreeze(policy, args.critic_warmup if args.init else 0)
    log_path = args.out / "log.jsonl"
    plot_path = args.out / "curves.png"
    # 200 episodes is ~25 rollouts at 8 envs: long enough that the success rate
    # is not quantized into eighths, short enough to still move within a stage.
    recent = {k: deque(maxlen=200)
              for k in ("success", "return", "agv", "ep_len", "gates",
                        "heading_err")}

    def draw_curves() -> None:
        """A broken plot must never take down a run that is otherwise fine."""
        try:
            save_training_curves(log_path, plot_path, args.plot_smooth)
        except Exception as exc:                      # pragma: no cover
            print(f"[plot] skipped: {exc}", flush=True)

    def on_rollout_end(algo: RecurrentPPO, rollout: int, stats: dict) -> None:
        infos = algo.episode_infos
        algo.episode_infos = []
        for info in infos:
            recent["success"].append(float(info.get("is_success", False)))
            recent["return"].append(float(info.get("ep_return", 0.0)))
            recent["agv"].append(float(info.get("agv", 0.0)))
            recent["ep_len"].append(float(info.get("ep_len", 0)))
            # Partial credit. Success is all-or-nothing, so early in a stage it
            # sits at zero while the policy is in fact learning gate 1 of 3.
            n_gates = max(1, int(info.get("n_gates", 1)))
            recent["gates"].append(float(info.get("gates_cleared", 0)) / n_gates)
            # abs: the signed error averages to ~0 across episodes that drift
            # both ways, which would read as "no drift" when there is plenty.
            recent["heading_err"].append(
                abs(float(info.get("heading_err", 0.0))))
        sampler.record_episodes(infos)

        # Why the episode ended, counted over this rollout. Success rate says
        # how often it worked; this says how it failed, which is the difference
        # between "raise R_CRASH" and "the policy is too cautious".
        reasons = Counter(info.get("terminal_reason") or "timeout"
                          for info in infos)

        if warmup.maybe_release(algo.num_timesteps):
            algo.optimizer = torch.optim.Adam(
                [q for q in policy.parameters() if q.requires_grad], lr=args.lr)
            print(f"[{algo.num_timesteps}] actor released")

        row = {
            "rollout": rollout,
            "timesteps": algo.num_timesteps,
            "stage": sampler.stage,
            "episodes": len(infos),
            **{k: float(np.mean(v)) if v else 0.0 for k, v in recent.items()},
            "reasons": dict(reasons),
            **stats,
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        if rollout % 5 == 0:
            top = " ".join(f"{k}={n}" for k, n in reasons.most_common(3))
            print(f"[{algo.num_timesteps:>9}] {sampler.describe()} "
                  f"ret={row['return']:+.2f} agv={row['agv']:.2f} "
                  f"yaw={row['heading_err']:.0f}deg "
                  f"pg={stats['pg']:+.4f} "
                  f"vf={stats['vf']:.4f} kl={stats['kl']:.4f} "
                  f"sil={stats['sil']:+.4f}@{stats['expert_frac']:.2f}\n"
                  f"{'':>11} why: {top}", flush=True)

        if args.plot_every and rollout % args.plot_every == 0:
            draw_curves()

        if args.curriculum:
            new_layouts = sampler.on_rollout_end()
            if new_layouts is not None:
                broadcast_layout(algo.venv, new_layouts)

        if rollout % 50 == 0:
            algo.save(args.out / f"ckpt_{algo.num_timesteps}.pt")

    try:
        algo.learn(args.timesteps, on_rollout_end=on_rollout_end)
    finally:
        algo.save(args.out / "final.pt")
        draw_curves()          # also covers Ctrl-C and a crash mid-run
        venv.close()
    print("saved", args.out / "final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
