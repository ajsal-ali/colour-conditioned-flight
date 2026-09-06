#!/usr/bin/env python3
"""Actor-Learner Distillation.

Premise: train with the expensive sequence model, deploy the cheap one.

  * the **actor** (LSTM) collects all experience and is what you would fly;
  * the **learner** (attention + memory tokens) does the RL, on the actor's
    trajectories, off-policy-corrected by PPO's own importance ratio -- the
    stored behaviour log-probs come from the actor, so `exp(logp_learner -
    logp_behaviour)` is already the correct ratio;
  * the actor is then pulled toward the learner by `KL(pi_learner || pi_actor)`.

Run this only *after* train_course has shown attention actually beats LSTM on
this task. If it does not, there is nothing worth distilling and the extra loop
is pure cost.

Caveat worth stating: with RMT memory tokens the learner is itself recurrent, so
the deployed actor's advantage is compute, not architecture. The gap should be
smaller than the original ALD paper's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.curriculum import CourseSampler, broadcast_layout     # noqa: E402
from mavrl.policy import MavrlActorCritic, load_policy_state     # noqa: E402
from mavrl.ppo import PPOConfig, RecurrentPPO                    # noqa: E402
from mavrl.sensor_noise import NoiseConfig                       # noqa: E402
from mavrl.vecenv import build_venv                              # noqa: E402


def gaussian_kl(p_mean, p_logstd, q_mean, q_logstd):
    """KL(p || q) for diagonal Gaussians, summed over action dims."""
    p_var = torch.exp(2 * p_logstd)
    q_var = torch.exp(2 * q_logstd)
    return (q_logstd - p_logstd
            + (p_var + (p_mean - q_mean) ** 2) / (2 * q_var) - 0.5).sum(-1)


class ALD:
    def __init__(self, actor: MavrlActorCritic, learner: MavrlActorCritic,
                 venv, cfg: PPOConfig, device: str, distill_coef: float = 1.0,
                 actor_lr: float = 3e-4):
        # The PPO driver collects with the ACTOR -- that is the whole point.
        self.ppo = RecurrentPPO(actor, venv, cfg, device=device)
        self.actor = self.ppo.policy
        self.learner = learner.to(self.ppo.device)
        self.device = self.ppo.device
        self.distill_coef = distill_coef

        self.learner_opt = torch.optim.Adam(
            [p for p in self.learner.parameters() if p.requires_grad],
            lr=cfg.learning_rate)
        self.actor_opt = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=actor_lr)

    def _learner_update(self, cfg: PPOConfig) -> dict:
        stats = {"pg": [], "vf": []}
        for _ in range(cfg.n_epochs):
            order = np.random.permutation(self.ppo.n_envs)
            for s in range(0, self.ppo.n_envs, cfg.envs_per_batch):
                idx = order[s:s + cfg.envs_per_batch]
                if not len(idx):
                    continue
                b = self.ppo.buffer.env_batch(idx)
                images = b["images"].permute(0, 1, 4, 2, 3).contiguous()
                state = self.learner.initial_state(len(idx), self.device)

                logp, values, _, _, _ = self.learner.evaluate(
                    images, b["proprio"], state, b["actions"], b["ep_starts"])

                adv = b["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                # Behaviour log-probs are the ACTOR's, so this ratio is the
                # off-policy correction, not a same-policy no-op.
                ratio = torch.exp(logp - b["logprobs"])
                pg = -torch.min(
                    adv * ratio,
                    adv * torch.clamp(ratio, 1 - cfg.clip_range,
                                      1 + cfg.clip_range)).mean()
                vf = nn.functional.mse_loss(values, b["returns"])
                loss = pg + cfg.vf_coef * vf

                self.learner_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.learner.parameters(),
                                         cfg.max_grad_norm)
                self.learner_opt.step()
                stats["pg"].append(float(pg.detach()))
                stats["vf"].append(float(vf.detach()))
        return {f"learner_{k}": float(np.mean(v)) if v else 0.0
                for k, v in stats.items()}

    def _distill(self, cfg: PPOConfig) -> dict:
        kls = []
        order = np.random.permutation(self.ppo.n_envs)
        for s in range(0, self.ppo.n_envs, cfg.envs_per_batch):
            idx = order[s:s + cfg.envs_per_batch]
            if not len(idx):
                continue
            b = self.ppo.buffer.env_batch(idx)
            images = b["images"].permute(0, 1, 4, 2, 3).contiguous()

            with torch.no_grad():
                l_state = self.learner.initial_state(len(idx), self.device)
                l_dist, _, _, _ = self.learner.sequence(
                    images, b["proprio"], l_state, b["ep_starts"])
                l_mean = l_dist.mean
                l_logstd = self.learner.log_std.expand_as(l_mean)

            a_state = self.actor.initial_state(len(idx), self.device)
            a_dist, _, _, _ = self.actor.sequence(
                images, b["proprio"], a_state, b["ep_starts"])
            a_logstd = self.actor.log_std.expand_as(a_dist.mean)

            kl = gaussian_kl(l_mean, l_logstd, a_dist.mean, a_logstd).mean()
            loss = self.distill_coef * kl

            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_opt.step()
            kls.append(float(kl.detach()))
        return {"distill_kl": float(np.mean(kls)) if kls else 0.0}

    def learn(self, total_timesteps: int, on_rollout_end=None) -> "ALD":
        cfg = self.ppo.cfg
        rollout = 0
        while self.ppo.num_timesteps < total_timesteps:
            self.ppo.collect()
            stats = self._learner_update(cfg)
            stats |= self._distill(cfg)
            rollout += 1
            if on_rollout_end is not None:
                on_rollout_end(self, rollout, stats)
        return self

    def save(self, path) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "learner": self.learner.state_dict(),
                    "aux_segments": list(self.actor.aux.segments),
                    "num_timesteps": self.ppo.num_timesteps}, path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/ald"))
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--actor", default="lstm")
    p.add_argument("--learner", default="attention")
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--actor-init", type=Path, default=None)
    p.add_argument("--learner-init", type=Path, default=None)
    p.add_argument("--distill-coef", type=float, default=1.0)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--sensor-noise", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-subproc", action="store_true")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    noise = (NoiseConfig().scaled(args.sensor_noise)
             if args.sensor_noise > 0 else NoiseConfig.disabled())
    sampler = CourseSampler(seed=args.seed, split="train")
    venv = build_venv(sampler.layout, args.n_envs, args.seed, noise,
                      subproc=not args.no_subproc)

    actor = MavrlActorCritic(memory_type=args.actor)
    learner = MavrlActorCritic(memory_type=args.learner,
                               mem_tokens=args.mem_tokens)
    for path, net, name in ((args.actor_init, actor, "actor"),
                            (args.learner_init, learner, "learner")):
        if path and Path(path).exists():
            ckpt = torch.load(path, map_location="cpu")
            load_policy_state(net, ckpt["policy"], ckpt.get("aux_segments"))
            print(f"warm-started {name} from {path}")

    cfg = PPOConfig(n_steps=args.n_steps,
                    envs_per_batch=max(1, args.n_envs // 2))
    ald = ALD(actor, learner, venv, cfg, args.device, args.distill_coef)

    log_path = args.out / "log.jsonl"
    recent = deque(maxlen=200)

    def on_rollout_end(ald: "ALD", rollout: int, stats: dict) -> None:
        infos = ald.ppo.episode_infos
        ald.ppo.episode_infos = []
        for info in infos:
            recent.append(bool(info.get("is_success", False)))
        sampler.record_episodes(infos)

        row = {"rollout": rollout, "timesteps": ald.ppo.num_timesteps,
               "stage": sampler.stage,
               "success": float(np.mean(recent)) if recent else 0.0, **stats}
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if rollout % 5 == 0:
            print(f"[{ald.ppo.num_timesteps:>9}] {sampler.describe()} "
                  f"kl={stats['distill_kl']:.4f}", flush=True)

        if args.curriculum:
            new_layout = sampler.on_rollout_end()
            if new_layout is not None:
                broadcast_layout(ald.ppo.venv, new_layout)
        if rollout % 50 == 0:
            ald.save(args.out / f"ckpt_{ald.ppo.num_timesteps}.pt")

    try:
        ald.learn(args.timesteps, on_rollout_end=on_rollout_end)
    finally:
        ald.save(args.out / "final.pt")
        venv.close()
    print("saved", args.out / "final.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
