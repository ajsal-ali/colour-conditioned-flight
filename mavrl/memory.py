#!/usr/bin/env python3
"""Memory backbones: LSTM, and temporal attention with recurrent memory tokens.

Both map a stream of VAE latents z_vae (64) to a memory-augmented state
z_t (256), which is concatenated with proprio and handed to PPO.

Why the attention variant carries memory tokens
-----------------------------------------------
Plain windowed self-attention is hard-capped at K steps; an LSTM's hidden state
is not. Comparing them directly would be comparing a bounded model against an
unbounded one. Recurrent memory tokens (Bulatov et al. 2022, RMT; cf.
Transformer-XL segment recurrence) close that gap: the tokens' output at step t
becomes their input at t+1, so forward memory is unbounded while attention keeps
sharp access inside the window.

    step t:  [ m_{t-1} | z_{t-K+1} ... z_t ] --attn--> [ m_t | ... | z_t^out ]
                   |                                       |         |
                   +------------- to t+1 ------------------+         +--> policy

Why AUX_T > ATTN_WINDOW
-----------------------
The aux loss asks the model to reconstruct the image from T steps ago. If the
attention window still contained step t-T, that is satisfiable by *copying* --
no compression, no memory. With K=16 < T=20 step t-T is outside the window, so
the only route to it is through the memory tokens, and both backbones face a
real memory task.

Note the gradient horizon is still bounded by BPTT truncation for both. The
forward memory is unbounded; the learning signal is not.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from mavrl.config import (
    ATTN_HEADS, ATTN_LAYERS, ATTN_WINDOW, AUX_OFFSETS, AUX_SEGMENTS, AUX_T,
    LATENT_DIM, MEMORY_DIM, N_MEM_TOKENS, N_PROPRIO,
)


class LSTMMemory(nn.Module):
    """Single-layer LSTM, as in the paper."""

    kind = "lstm"
    #: nn.LSTM state is (num_layers, batch, hidden) -- batch is axis 1.
    state_batch_dim = 1

    def __init__(self, latent_dim: int = LATENT_DIM,
                 hidden: int = MEMORY_DIM, **_):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden, num_layers=1, batch_first=True)
        self.output_dim = hidden
        self.hidden = hidden

    def initial_state(self, batch: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch, self.hidden, device=device)
        return z, z.clone()

    def forward(self, z_seq: torch.Tensor, state=None):
        """(B,T,latent) -> (B,T,hidden), new_state."""
        out, new_state = self.lstm(z_seq, state)
        return out, new_state

    def step(self, z: torch.Tensor, state=None):
        """(B,latent) -> (B,hidden), new_state."""
        out, new_state = self.lstm(z.unsqueeze(1), state)
        return out.squeeze(1), new_state


class TemporalAttentionMemory(nn.Module):
    """Windowed self-attention over the last K latents, plus RMT memory tokens.

    State is `(buffer, mem)`:
        buffer (B, K, latent)  ring of recent latents, oldest first
        mem    (B, n_mem, d)   recurrent memory tokens, carried across steps
    """

    kind = "attention"
    #: Both state tensors are (batch, ...) -- batch is axis 0. Declared rather
    #: than inferred: with n_envs == n_mem the axes are indistinguishable by
    #: shape, and guessing silently slices the wrong one.
    state_batch_dim = 0

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = MEMORY_DIM,
                 window: int = ATTN_WINDOW, n_mem: int = N_MEM_TOKENS,
                 n_heads: int = ATTN_HEADS, n_layers: int = ATTN_LAYERS, **_):
        super().__init__()
        self.window = window
        self.n_mem = n_mem
        self.hidden = hidden
        self.output_dim = hidden
        self.latent_dim = latent_dim

        self.proj = nn.Linear(latent_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, window, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.mem_init = nn.Parameter(torch.zeros(1, max(n_mem, 1), hidden))
        nn.init.normal_(self.mem_init, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=4 * hidden,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def initial_state(self, batch: int, device):
        buffer = torch.zeros(batch, self.window, self.latent_dim, device=device)
        mem = self.mem_init.expand(batch, -1, -1).to(device).clone()
        return buffer, mem

    def _attend(self, buffer: torch.Tensor, mem: torch.Tensor):
        tokens = self.proj(buffer) + self.pos
        if self.n_mem > 0:
            tokens = torch.cat([mem, tokens], dim=1)
        out = self.encoder(tokens)
        if self.n_mem > 0:
            new_mem, out = out[:, :self.n_mem], out[:, self.n_mem:]
        else:
            new_mem = mem
        return self.norm(out[:, -1]), new_mem

    def step(self, z: torch.Tensor, state=None):
        """(B,latent) -> (B,hidden), new_state."""
        if state is None:
            state = self.initial_state(z.shape[0], z.device)
        buffer, mem = state
        buffer = torch.cat([buffer[:, 1:], z.unsqueeze(1)], dim=1)
        out, mem = self._attend(buffer, mem)
        return out, (buffer, mem)

    def forward(self, z_seq: torch.Tensor, state=None):
        """(B,T,latent) -> (B,T,hidden), new_state.

        Rolled explicitly rather than vectorized: the memory tokens are
        genuinely recurrent, so step t+1 depends on step t's output.
        """
        if state is None:
            state = self.initial_state(z_seq.shape[0], z_seq.device)
        outs = []
        for t in range(z_seq.shape[1]):
            out, state = self.step(z_seq[:, t], state)
            outs.append(out)
        return torch.stack(outs, dim=1), state


class AuxReconstructionHead(nn.Module):
    """Paper's training-only head: concat(z_t, proprio) -> several image latents.

    Paper Eq. (2): the fully connected layer emits `3 * N_e`, split into past,
    current and future segments (`I_{t-T}`, `I_t`, `I_{t+T}`), all decoded by the
    SAME frozen decoder; `lambda_i in {0,1}` picks which are supervised.

    Here the head is sized to the *selected* segments only -- an unsupervised
    segment would receive no gradient, so emitting it would just be dead weights.
    `segments` is the lambda configuration by name; order is preserved and is the
    order of the returned axis.
    """

    def __init__(self, memory_dim: int = MEMORY_DIM, n_proprio: int = N_PROPRIO,
                 latent_dim: int = LATENT_DIM,
                 segments: Sequence[str] = AUX_SEGMENTS):
        super().__init__()
        bad = [s for s in segments if s not in AUX_OFFSETS]
        if bad or not segments:
            raise ValueError(
                f"segments must be a non-empty subset of {list(AUX_OFFSETS)}, "
                f"got {list(segments)}")
        self.segments = tuple(segments)
        self.n_targets = len(self.segments)
        self.latent_dim = latent_dim
        self.fc = nn.Linear(memory_dim + n_proprio, self.n_targets * latent_dim)

    def offsets(self, offset: int = AUX_T) -> Tuple[int, ...]:
        """Per-segment time offset relative to t, in policy steps."""
        return tuple(int(round(AUX_OFFSETS[s] / AUX_T * offset))
                     for s in self.segments)

    def forward(self, z_t: torch.Tensor, proprio: torch.Tensor):
        """-> (..., n_targets, latent), segments in `self.segments` order."""
        h = self.fc(torch.cat([z_t, proprio], dim=-1))
        return h.view(*h.shape[:-1], self.n_targets, self.latent_dim)


MEMORY_TYPES = {
    "lstm": LSTMMemory,
    "attention": TemporalAttentionMemory,
}


def build_memory(memory_type: str, **kwargs) -> nn.Module:
    if memory_type not in MEMORY_TYPES:
        raise ValueError(
            f"unknown memory_type {memory_type!r}; expected one of "
            f"{sorted(MEMORY_TYPES)}")
    return MEMORY_TYPES[memory_type](**kwargs)


def aux_valid_range(seq_len: int, offsets: Sequence[int]) -> Tuple[int, int]:
    """[lo, hi) timesteps of a window whose every aux target is in the window.

    A `future` segment costs T steps off the *end* as well as the T the `past`
    segment already costs off the front, so the three-way loss needs a window
    longer than 2T to supervise anything at all.
    """
    lo = max(0, -min(offsets))
    hi = seq_len - max(0, max(offsets))
    return lo, hi
