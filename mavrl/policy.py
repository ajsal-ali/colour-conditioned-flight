#!/usr/bin/env python3
"""Actor-critic network: SeVAE encoder -> memory -> policy/value heads.

Dataflow, matching the paper except inside the SeVAE box:

    image --[SeVAE enc]--> z_vae(64) --[memory]--> z_t(256)
                                                     |
          +------------------------------------------+----------------+
          | TRAIN ONLY (dropped at deploy)                            | DEPLOY
          v                                                           v
    concat(z_t, proprio) -FC-> kx64 -> [frozen decoder] -> I_hat   concat(z_t, proprio) -> pi

where k is the number of supervised aux segments (paper Eq. (2)'s lambda).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from mavrl.config import (
    AUX_SEGMENTS, LATENT_DIM, MEMORY_DIM, N_ACTIONS, N_PROPRIO,
)
from mavrl.memory import AuxReconstructionHead, build_memory
from mavrl.sevae import SeVAE


#: Canonical segment order, used when a checkpoint records a width but not names.
AUX_SEGMENT_ORDER = ("past", "current", "future")


def load_policy_state(policy, state: dict, segments: Optional[Sequence[str]] = None,
                      strict: bool = False):
    """Load a saved policy, resizing the training-only aux head to fit first.

    `strict=False` forgives missing and unexpected keys but NOT a size mismatch,
    and `aux.fc` is exactly the layer whose width varies -- it is sized by the
    lambda configuration stage 3 happened to run with. Since the head is dropped
    at deploy, resize it to the checkpoint rather than refuse the checkpoint.
    """
    w = state.get("aux.fc.weight")
    if w is not None:
        n = w.shape[0] // policy.aux.latent_dim
        segs = tuple(segments) if segments and len(segments) == n \
            else AUX_SEGMENT_ORDER[:n]
        if segs != policy.aux.segments:
            device = next(policy.parameters()).device
            policy.aux = AuxReconstructionHead(segments=segs).to(device)
    return policy.load_state_dict(state, strict=strict)


def _mlp(in_dim: int, hidden, out_dim: int) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MavrlActorCritic(nn.Module):
    """Shared encoder + memory, separate policy and value heads."""

    def __init__(self, memory_type: str = "lstm", n_actions: int = N_ACTIONS,
                 n_proprio: int = N_PROPRIO, hidden=(128, 128),
                 log_std_init: float = -1.0, freeze_encoder: bool = False,
                 mem_tokens: Optional[int] = None,
                 aux_segments: Sequence[str] = AUX_SEGMENTS):
        super().__init__()
        self.memory_type = memory_type
        self.sevae = SeVAE()
        mem_kwargs = {} if mem_tokens is None else {"n_mem": mem_tokens}
        self.memory = build_memory(memory_type, **mem_kwargs)
        # Training-only head; carried so a stage-3 checkpoint loads whole. Its
        # width depends on the lambda configuration it was trained with.
        self.aux = AuxReconstructionHead(segments=aux_segments)

        feat = self.memory.output_dim + n_proprio
        self.pi = _mlp(feat, hidden, n_actions)
        self.vf = _mlp(feat, hidden, 1)
        self.log_std = nn.Parameter(torch.full((n_actions,), log_std_init))

        self.freeze_encoder(freeze_encoder)

    # -- encoder freezing ----------------------------------------------------

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Stage 1 runs with a frozen random encoder, as the paper's step 1 does."""
        self._encoder_frozen = frozen
        for p in self.sevae.parameters():
            p.requires_grad = not frozen

    # -- state ---------------------------------------------------------------

    def initial_state(self, batch: int, device):
        return self.memory.initial_state(batch, device)

    @property
    def state_batch_dim(self) -> int:
        return self.memory.state_batch_dim

    def _reset_state(self, state, mask: torch.Tensor, fresh):
        """Zero the state where `mask` is True (episode boundaries).

        The batch axis comes from the memory module rather than from the tensor
        shape: LSTM state is (layers, B, H) and attention state is (B, ...), and
        when n_envs happens to equal n_mem those are indistinguishable by shape.
        """
        bd = self.state_batch_dim
        out = []
        for s, f in zip(state, fresh):
            view = [1] * s.dim()
            view[bd] = -1
            out.append(torch.where(mask.view(*view), f, s))
        return tuple(out)

    # -- forward -------------------------------------------------------------

    def encode(self, image: torch.Tensor, sample: bool = True):
        if self._encoder_frozen:
            with torch.no_grad():
                z, mu, logvar = self.sevae.encode(image, sample)
            return z.detach(), mu.detach(), logvar.detach()
        return self.sevae.encode(image, sample)

    def _heads(self, z_t: torch.Tensor, proprio: torch.Tensor):
        feat = torch.cat([z_t, proprio], dim=-1)
        mean = self.pi(feat)
        value = self.vf(feat).squeeze(-1)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        return dist, value

    def step(self, image: torch.Tensor, proprio: torch.Tensor, state,
             sample: bool = True):
        """One timestep. image (B,4,H,W) uint8, proprio (B,P)."""
        z, _, _ = self.encode(image, sample)
        z_t, state = self.memory.step(z, state)
        dist, value = self._heads(z_t, proprio)
        return dist, value, state

    def sequence(self, image: torch.Tensor, proprio: torch.Tensor,
                 state, episode_starts: Optional[torch.Tensor] = None,
                 sample: bool = True):
        """A whole rollout segment. image (B,T,4,H,W), proprio (B,T,P).

        Rolled per timestep rather than batched through the memory, because the
        state has to be reset at episode boundaries *inside* the segment -- a
        rollout routinely spans several episodes per env.
        """
        b, t = image.shape[:2]
        flat = image.reshape(b * t, *image.shape[2:])
        z, _, _ = self.encode(flat, sample)
        z = z.view(b, t, -1)

        fresh = self.initial_state(b, image.device)
        outs = []
        for i in range(t):
            if episode_starts is not None:
                state = self._reset_state(state, episode_starts[:, i].bool(), fresh)
            z_t, state = self.memory.step(z[:, i], state)
            outs.append(z_t)
        z_seq = torch.stack(outs, dim=1)

        feat = torch.cat([z_seq, proprio], dim=-1)
        mean = self.pi(feat)
        value = self.vf(feat).squeeze(-1)
        dist = Normal(mean, self.log_std.exp().expand_as(mean))
        return dist, value, state, z_seq

    # -- convenience for rollout collection ---------------------------------

    @torch.no_grad()
    def act(self, image, proprio, state, deterministic: bool = False):
        dist, value, state = self.step(image, proprio, state)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value, state

    def evaluate(self, image, proprio, state, actions, episode_starts=None):
        dist, value, state, z_seq = self.sequence(
            image, proprio, state, episode_starts)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, value, entropy, state, z_seq


def obs_to_tensors(obs: dict, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vec/single-env observation dict -> (image NCHW uint8, proprio)."""
    image = torch.as_tensor(obs["image"], device=device)
    if image.dim() == 3:                          # HWC -> NHWC (single env)
        image = image.unsqueeze(0)
    if image.dim() == 4 and image.shape[-1] in (3, 4):  # NHWC -> NCHW
        image = image.permute(0, 3, 1, 2)
    proprio = torch.as_tensor(np.asarray(obs["proprio"], dtype=np.float32),
                              device=device)
    if proprio.dim() == 1:                        # (P,) -> (1,P)
        proprio = proprio.unsqueeze(0)
    return image.contiguous(), proprio
