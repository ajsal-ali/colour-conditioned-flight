#!/usr/bin/env python3
"""Recurrent PPO with truncated BPTT.

Written rather than taken from sb3-contrib because RecurrentPPO sizes its state
buffers from `lstm.hidden_size` and feeds that same number into the MLP
extractor. The attention backbone needs a 1024-wide state (a 16x64 ring buffer
plus 4x256 memory tokens) but a 256-wide output, and those cannot both be
`hidden_size` without patching SB3 internals. ALD needs a custom two-network
loop regardless, so one loop serves both.

Minibatching is over **envs, not timesteps**: each minibatch takes a subset of
environments and their full n_steps of history, so BPTT runs over an unbroken
sequence and the recurrent state stays valid. Episode boundaries inside a
segment are handled by resetting state mid-sequence (see
MavrlActorCritic.sequence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from mavrl.policy import MavrlActorCritic, obs_to_tensors


@dataclass
class PPOConfig:
    n_steps: int = 128           # per env, per rollout
    n_epochs: int = 4
    envs_per_batch: int = 4      # minibatch size, in envs
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    target_kl: Optional[float] = 0.03
    normalize_advantage: bool = True
    #: Running return normalization, the cheap half of VecNormalize. Matters
    #: here because the curriculum changes episode length and gate count, so raw
    #: return scale drifts between stages even with a constant per-course total.
    normalize_returns: bool = True

    # -- demonstrations in the rollout buffer -------------------------------
    #
    # A fraction of the envs is flown by the scripted pilot instead of the
    # policy. Their transitions go into the *same* buffer, with real rewards,
    # the same gamma, the same GAE and the same running return normalizer -- so
    # their advantages are directly comparable with the policy's own, which is
    # the whole reason to generate demonstrations online instead of replaying
    # stored ones. (The shards carry no reward at all: mavrl.dataset stores
    # image/seg/depth/proprio/action, and the reward needs absolute position,
    # which proprio does not have.)
    #
    # What those transitions may NOT do is enter the clipped surrogate. The
    # importance ratio there is pi_theta / pi_behaviour, and the pilot is
    # deterministic -- its density is a delta, so the ratio either explodes or
    # saturates the clip and contributes exactly zero gradient. They get the
    # self-imitation objective below instead.

    #: Fraction of envs flown by the pilot at step 0. 0 disables the whole path.
    expert_frac: float = 0.0
    #: Timesteps over which that fraction decays linearly to zero.
    expert_decay: int = 1_000_000
    #: Weight on the self-imitation term, -E[(A)+ * log pi(a_demo|s)].
    #:
    #: Oh et al. (2018). The (A)+ gate makes it self-limiting: once the critic
    #: values a state above what the demonstration achieved there, the weight is
    #: zero and the term stops pulling. That is a decay the critic earns rather
    #: than one imposed by a schedule -- but it only works if the demonstration
    #: advantage is on the critic's scale, which is what generating it in-buffer
    #: guarantees.
    #:
    #: The advantage weights are self-normalized, so this is a scale-free
    #: mixing fraction: `sil` is a weighted mean of -log pi over the
    #: demonstrations that beat the critic, directly comparable to `pg` in the
    #: log regardless of the reward scale.
    sil_coef: float = 0.1


class RunningNorm:
    """Welford running mean/variance of the discounted return."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self._ret = None

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        if self._ret is None:
            self._ret = np.zeros(rewards.shape[-1], dtype=np.float64)
        for r, d in zip(rewards, dones):
            self._ret = self._ret * self.gamma + r
            self._ret[d.astype(bool)] = 0.0
            batch = self._ret
            bm, bv, bc = batch.mean(), batch.var(), batch.size
            delta = bm - self.mean
            tot = self.count + bc
            self.mean += delta * bc / tot
            m_a = self.var * self.count
            m_b = bv * bc
            self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
            self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)


class RolloutBuffer:
    """(T, N, ...) storage for one rollout, plus the state each env started in."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape, n_proprio: int,
                 n_actions: int, device, state_batch_dim: int = 0):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        # Which axis of a state tensor indexes the environment. Declared by the
        # memory module, never inferred -- see MavrlActorCritic._reset_state.
        self.state_batch_dim = state_batch_dim
        self.images = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.uint8)
        self.proprio = np.zeros((n_steps, n_envs, n_proprio), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, n_actions), dtype=np.float32)
        self.logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.ep_starts = np.zeros((n_steps, n_envs), dtype=np.float32)
        #: 1.0 where the stored action came from the scripted pilot rather than
        #: from the policy. The two need different objectives, not different
        #: buffers -- everything else about the transition is identical.
        self.expert = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.initial_state = None
        self.ptr = 0

    def reset(self, initial_state) -> None:
        self.ptr = 0
        self.expert[:] = 0.0
        self.initial_state = tuple(s.detach().clone() for s in initial_state)

    def add(self, image, proprio, action, logprob, value, reward, done,
            ep_start, expert=None):
        i = self.ptr
        self.images[i] = image
        self.proprio[i] = proprio
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = done
        self.ep_starts[i] = ep_start
        if expert is not None:
            self.expert[i] = expert
        self.ptr += 1

    def compute_returns(self, last_values: np.ndarray, last_dones: np.ndarray,
                        gamma: float, gae_lambda: float, ret_std: float = 1.0):
        adv = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        rewards = self.rewards / ret_std
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_nonterminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (rewards[t] + gamma * next_values * next_nonterminal
                     - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        self.advantages = adv
        self.returns = adv + self.values

    def env_batch(self, idx: np.ndarray):
        """(B, T, ...) tensors for a subset of envs -- time axis intact."""
        def swap(a):
            return torch.as_tensor(a[:, idx], device=self.device).transpose(0, 1)
        return {
            "images": swap(self.images),
            "proprio": swap(self.proprio),
            "actions": swap(self.actions),
            "logprobs": swap(self.logprobs),
            "advantages": swap(self.advantages),
            "returns": swap(self.returns),
            "ep_starts": swap(self.ep_starts),
            "expert": swap(self.expert),
            "state": tuple(
                s[:, idx] if self.state_batch_dim == 1 else s[idx]
                for s in self.initial_state),
        }


class RecurrentPPO:
    def __init__(self, policy: MavrlActorCritic, venv, cfg: PPOConfig,
                 device: str = "cuda", aux_loss_fn: Optional[Callable] = None):
        self.policy = policy.to(device)
        self.venv = venv
        self.cfg = cfg
        self.device = torch.device(device)
        self.aux_loss_fn = aux_loss_fn

        self.n_envs = venv.num_envs
        img_space = venv.observation_space["image"]
        # Stored NHWC as the env produces it; permuted to NCHW at use.
        self.obs_shape = img_space.shape
        self.buffer = RolloutBuffer(
            cfg.n_steps, self.n_envs, self.obs_shape,
            venv.observation_space["proprio"].shape[0],
            venv.action_space.shape[0], self.device,
            state_batch_dim=self.policy.state_batch_dim)

        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=cfg.learning_rate)
        self.ret_norm = RunningNorm(cfg.gamma) if cfg.normalize_returns else None

        self._obs = venv.reset()
        self._state = self.policy.initial_state(self.n_envs, self.device)
        self._ep_start = np.ones(self.n_envs, dtype=np.float32)
        self.num_timesteps = 0
        self.episode_infos = []
        # Per-env accumulators for the finished-episode return/length. The env's
        # info dict cannot carry these -- an episode outlives a rollout, so only
        # the loop that owns the step sequence can sum it.
        self._ep_return = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)

        # Demonstration bookkeeping. `_expert_a` holds the action the pilot
        # would take from the state `self._obs` describes -- the envs compute it
        # and ship it in `info`, because the trainer needs it *before* it acts
        # and a per-step env_method to every worker would cost more than the
        # pilot itself does.
        self.n_actions = venv.action_space.shape[0]
        self._expert_a = np.zeros((self.n_envs, self.n_actions), dtype=np.float32)
        self._expert_ok = np.zeros(self.n_envs, dtype=bool)
        self._expert_ptr = 0

        # Declared last, and the set_attr goes with it: an earlier `_emitting =
        # True` would be clobbered by this block's initialization.
        self._emitting = cfg.expert_frac > 0.0
        if self._emitting:
            self.venv.set_attr("emit_expert", True)

    # -- rollout -------------------------------------------------------------

    # -- demonstrations ------------------------------------------------------

    def expert_fraction(self) -> float:
        """Fraction of envs the pilot should fly right now: linear to zero."""
        cfg = self.cfg
        if cfg.expert_frac <= 0.0:
            return 0.0
        if cfg.expert_decay <= 0:
            return float(cfg.expert_frac)
        left = 1.0 - self.num_timesteps / float(cfg.expert_decay)
        return float(cfg.expert_frac * max(0.0, left))

    def _select_expert_envs(self) -> np.ndarray:
        """Which envs the pilot flies this rollout, rotating through them.

        A fixed subset would keep the same envs on demonstrations forever, and
        minibatching is over envs (`envs_per_batch`) -- so those envs' whole BPTT
        sequences would never contain a policy action. Rotating spreads the
        demonstrations over every env and every render worker instead.

        The set is held for a whole rollout, not resampled per step, so an env's
        (B, T) sequence is all pilot or all policy rather than interleaved.
        """
        n = int(round(self.expert_fraction() * self.n_envs))
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        n = min(n, self.n_envs)
        idx = (self._expert_ptr + np.arange(n)) % self.n_envs
        self._expert_ptr = int((self._expert_ptr + n) % self.n_envs)
        return idx.astype(np.int64)

    def _harvest_expert(self, infos) -> None:
        """Read next step's pilot action out of the step infos.

        On a step that ended an episode the vec env has already auto-reset, so
        `self._obs` is the *new* episode's first observation and the matching
        pilot action is the one reset produced -- `info["reset_info"]`, not the
        terminal step's own info.
        """
        for i, info in enumerate(infos):
            src = info.get("reset_info") or info
            a = src.get("expert_action")
            if a is None:
                self._expert_ok[i] = False
            else:
                self._expert_a[i] = a
                self._expert_ok[i] = True

    # -- rollout -------------------------------------------------------------

    @torch.no_grad()
    def collect(self) -> None:
        self.buffer.reset(self._state)
        expert_envs = self._select_expert_envs()
        if self._emitting and expert_envs.size == 0:
            # The fraction has decayed to zero. Stop the envs computing a pilot
            # action nobody reads for the rest of the run.
            self.venv.set_attr("emit_expert", False)
            self._emitting = False
        for _ in range(self.cfg.n_steps):
            image, proprio = obs_to_tensors(self._obs, self.device)
            # Not policy.act(): the sampled action is overridden for the pilot's
            # envs, and the stored log-prob has to be log pi(a_stored | s) for
            # whichever action actually went to the env.
            dist, value, self._state = self.policy.step(
                image, proprio, self._state)
            action = dist.sample()

            expert_mask = np.zeros(self.n_envs, dtype=np.float32)
            if expert_envs.size:
                sel = expert_envs[self._expert_ok[expert_envs]]
                if sel.size:
                    action[sel] = torch.as_tensor(
                        self._expert_a[sel], dtype=action.dtype,
                        device=action.device)
                    expert_mask[sel] = 1.0
            logp = dist.log_prob(action).sum(-1)

            a = action.cpu().numpy()
            obs, reward, done, infos = self.venv.step(np.clip(a, -1.0, 1.0))
            if self._emitting:
                self._harvest_expert(infos)

            self.buffer.add(
                self._obs["image"], self._obs["proprio"], a,
                logp.cpu().numpy(), value.cpu().numpy(),
                reward, done.astype(np.float32), self._ep_start,
                expert_mask)

            self._ep_return += reward
            self._ep_len += 1
            for i, d in enumerate(done):
                if d:
                    self.episode_infos.append(
                        dict(infos[i], ep_return=float(self._ep_return[i]),
                             ep_len=int(self._ep_len[i])))
                    self._ep_return[i] = 0.0
                    self._ep_len[i] = 0
            self._ep_start = done.astype(np.float32)
            self._obs = obs
            self.num_timesteps += self.n_envs

            # Reset the recurrent state for envs that just finished, so the next
            # episode does not inherit the previous one's memory.
            if done.any():
                mask = torch.as_tensor(done.astype(bool), device=self.device)
                fresh = self.policy.initial_state(self.n_envs, self.device)
                self._state = self.policy._reset_state(self._state, mask, fresh)

        image, proprio = obs_to_tensors(self._obs, self.device)
        _, last_values, _ = self.policy.step(image, proprio, self._state)

        if self.ret_norm is not None:
            self.ret_norm.update(self.buffer.rewards, self.buffer.dones)
            std = self.ret_norm.std
        else:
            std = 1.0
        self.buffer.compute_returns(
            last_values.cpu().numpy(), self._ep_start,
            self.cfg.gamma, self.cfg.gae_lambda, std)

    # -- update --------------------------------------------------------------

    def update(self) -> dict:
        cfg = self.cfg
        stats = {"pg": [], "vf": [], "ent": [], "kl": [], "aux": [], "sil": []}
        stop = False

        for _ in range(cfg.n_epochs):
            order = np.random.permutation(self.n_envs)
            for start in range(0, self.n_envs, cfg.envs_per_batch):
                idx = order[start:start + cfg.envs_per_batch]
                if len(idx) == 0:
                    continue
                b = self.buffer.env_batch(idx)

                images = b["images"].permute(0, 1, 4, 2, 3).contiguous()
                logp, values, entropy, _, z_seq = self.policy.evaluate(
                    images, b["proprio"], b["state"], b["actions"],
                    b["ep_starts"])

                # (B, T) in {0, 1}. `pol` selects the transitions the clipped
                # surrogate is valid for; `exp` the pilot's.
                exp = b["expert"]
                pol = 1.0 - exp
                n_pol = pol.sum()
                n_exp = exp.sum()

                adv_raw = b["advantages"]
                if cfg.normalize_advantage:
                    # Standardize over the POLICY transitions only. The pilot's
                    # advantages are systematically higher -- that is the point
                    # of them -- so folding them into the mean would shift every
                    # policy advantage down by however many demonstrations
                    # happened to land in this minibatch.
                    if n_pol > 1:
                        mean = (adv_raw * pol).sum() / n_pol
                        # n_pol - 1, not n_pol: torch.std is Bessel-corrected,
                        # and matching it keeps a run with no demonstrations
                        # numerically identical to plain PPO.
                        var = (((adv_raw - mean) ** 2 * pol).sum()
                               / (n_pol - 1.0))
                        adv = (adv_raw - mean) / (var.sqrt() + 1e-8)
                    else:
                        adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
                else:
                    adv = adv_raw

                ratio = torch.exp(logp - b["logprobs"])
                pg_t = -torch.min(
                    adv * ratio,
                    adv * torch.clamp(ratio, 1 - cfg.clip_range,
                                      1 + cfg.clip_range))
                pg = (pg_t * pol).sum() / n_pol.clamp(min=1.0)

                # The value loss keeps every transition, demonstrations
                # included. They are mildly off-policy targets, but they are
                # also the only place the critic ever sees the end of the
                # course: the run that motivated this logged 151 completions in
                # 5.66M steps, so V was fitted almost entirely on trajectories
                # that never finished. A critic that has never seen the exit
                # cannot produce a meaningful advantage there.
                vf = nn.functional.mse_loss(values, b["returns"])
                ent = entropy.mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent

                # Self-imitation on the pilot's transitions. Advantage-weighted
                # log-likelihood, not a ratio: see PPOConfig.expert_frac.
                #
                # The gate is the RAW advantage, so `max(A, 0)` keeps its
                # meaning ("better than the critic expected") -- after
                # standardization roughly half of any batch is positive by
                # construction, which would gate on nothing.
                #
                # The weights are then divided by their own sum rather than by
                # n_exp. Dividing by n_exp leaves the reward scale in the
                # coefficient: raw advantages here run to several units, so
                # sil_coef=0.1 was applying an effective BC weight an order of
                # magnitude larger than intended, and a BC term that large
                # drives log_std down far faster than ent_coef can hold it up.
                # Self-normalized, `sil` is a weighted mean of -log pi over the
                # demonstrations the critic says beat expectations: sil_coef is
                # then a pure mixing fraction against `pg`, unchanged if every
                # reward in the env is scaled by a constant.
                sil_val = 0.0
                if n_exp > 0:
                    w = torch.clamp(adv_raw, min=0.0) * exp
                    wsum = w.sum()
                    if wsum > 0:
                        sil = -(logp * w).sum() / wsum
                        loss = loss + cfg.sil_coef * sil
                        sil_val = float(sil.detach())

                aux_val = 0.0
                if self.aux_loss_fn is not None:
                    aux = self.aux_loss_fn(self.policy, b, z_seq)
                    loss = loss + aux
                    aux_val = float(aux.detach())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Policy transitions only. The pilot's ratio is meaningless
                    # (its behaviour density is a delta), and left in here it
                    # would trip the target_kl early stop below and silently
                    # truncate the epoch.
                    kl_t = (ratio - 1) - (logp - b["logprobs"])
                    approx_kl = float(
                        (kl_t * pol).sum() / n_pol.clamp(min=1.0))
                stats["pg"].append(float(pg.detach()))
                stats["vf"].append(float(vf.detach()))
                stats["ent"].append(float(ent.detach()))
                stats["kl"].append(approx_kl)
                stats["aux"].append(aux_val)
                stats["sil"].append(sil_val)

                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break

        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out["expert_frac"] = self.expert_fraction()
        return out

    # -- driver --------------------------------------------------------------

    def learn(self, total_timesteps: int,
              on_rollout_end: Optional[Callable] = None) -> "RecurrentPPO":
        rollout = 0
        while self.num_timesteps < total_timesteps:
            self.collect()
            stats = self.update()
            rollout += 1
            if on_rollout_end is not None:
                on_rollout_end(self, rollout, stats)
        return self

    # -- checkpointing -------------------------------------------------------

    def save(self, path) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory_type": self.policy.memory_type,
            "aux_segments": list(self.policy.aux.segments),
            "num_timesteps": self.num_timesteps,
        }, path)

    def load(self, path, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"], strict=strict)
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass          # param groups changed (e.g. encoder unfrozen)
        self.num_timesteps = ckpt.get("num_timesteps", 0)
