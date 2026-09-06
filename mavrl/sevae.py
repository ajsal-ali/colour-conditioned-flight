#!/usr/bin/env python3
"""Semantically-enhanced VAE.

The paper's VAE is depth-only, reconstruction-weighted uniformly. Two changes
here, both deliberate:

1. **RGB-D input.** Colour is load-bearing on this course -- a red bar means fly
   above, a blue bar means fly below, and depth alone cannot tell them apart.
2. **Proximity-weighted reconstruction plus a segmentation head.** The weight
   makes the latent spend capacity on near geometry; the seg head is what makes
   it *semantic*. Weighting alone cannot teach "red bar" vs "blue bar", so both
   are needed rather than either.

The segmentation cross-entropy is deliberately **not** proximity-weighted. With
the weight falling to 0.05 by 12 m it would zero out gradients exactly where the
next station's colour cue first appears, which is the information the above/below
decision depends on.

No state input anywhere -- matching the paper, where the state vector enters at
the memory module's aux head and at the policy, never inside the VAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mavrl.config import (
    BETA_KL, DEPTH_MAX, D_FAR, D_NEAR, IMG_RES, LAMBDA_SEG, LATENT_DIM, W_MIN,
)
from mavrl.course_world import N_SEM_CLASSES


def proximity_weight(depth_m: torch.Tensor) -> torch.Tensor:
    """Torch twin of mavrl.imageproc.proximity_weight. Keep them in step."""
    t = ((depth_m - D_NEAR) / (D_FAR - D_NEAR)).clamp(0.0, 1.0)
    return W_MIN + (1.0 - W_MIN) * (1.0 - t) ** 2


@dataclass
class SeVAEOutput:
    rgb: torch.Tensor          # (B,3,H,W) in [0,1]
    depth: torch.Tensor        # (B,1,H,W) in [0,1], normalized by DEPTH_MAX
    seg_logits: torch.Tensor   # (B,C,H,W)
    mu: torch.Tensor           # (B,latent)
    logvar: torch.Tensor       # (B,latent)
    z: torch.Tensor            # (B,latent)


class Encoder(nn.Module):
    """Six strided conv layers, 128 -> 2, then two heads for mu and logvar."""

    def __init__(self, in_ch: int = 4, latent: int = LATENT_DIM,
                 res: int = IMG_RES):
        super().__init__()
        chans = (32, 64, 64, 128, 128, 256)
        layers = []
        c_prev = in_ch
        for c in chans:
            layers += [nn.Conv2d(c_prev, c, 4, stride=2, padding=1), nn.ReLU(True)]
            c_prev = c
        self.conv = nn.Sequential(*layers)
        self.spatial = res // (2 ** len(chans))          # 128 -> 2
        flat = chans[-1] * self.spatial ** 2
        self.fc_mu = nn.Linear(flat, latent)
        self.fc_logvar = nn.Linear(flat, latent)
        self.out_ch = chans[-1]

    def forward(self, x: torch.Tensor):
        h = self.conv(x).flatten(1)
        # Clamped so a diverging run fails visibly rather than producing NaNs.
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)


class Decoder(nn.Module):
    """Mirror of the encoder with three output heads off a shared trunk."""

    def __init__(self, latent: int = LATENT_DIM, res: int = IMG_RES,
                 n_classes: int = N_SEM_CLASSES):
        super().__init__()
        chans = (256, 128, 128, 64, 64, 32)
        self.spatial = res // (2 ** len(chans))
        self.c0 = chans[0]
        self.fc = nn.Linear(latent, chans[0] * self.spatial ** 2)

        layers = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
                       nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(chans[-1], chans[-1], 4, stride=2, padding=1),
                   nn.ReLU(True)]
        self.deconv = nn.Sequential(*layers)

        self.head_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.head_depth = nn.Conv2d(chans[-1], 1, 3, padding=1)
        self.head_seg = nn.Conv2d(chans[-1], n_classes, 3, padding=1)

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(-1, self.c0, self.spatial, self.spatial)
        h = self.deconv(h)
        return (torch.sigmoid(self.head_rgb(h)),
                torch.sigmoid(self.head_depth(h)),
                self.head_seg(h))


class SeVAE(nn.Module):
    def __init__(self, in_ch: int = 4, latent: int = LATENT_DIM,
                 res: int = IMG_RES, n_classes: int = N_SEM_CLASSES):
        super().__init__()
        self.encoder = Encoder(in_ch, latent, res)
        self.decoder = Decoder(latent, res, n_classes)
        self.latent_dim = latent

    @staticmethod
    def reparameterize(mu, logvar, sample: bool = True):
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, image_u8: torch.Tensor, sample: bool = True):
        """(B,4,H,W) uint8 or float in [0,255] -> latent."""
        x = image_u8.float() / 255.0
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar, sample), mu, logvar

    def forward(self, image_u8: torch.Tensor, sample: bool = True) -> SeVAEOutput:
        z, mu, logvar = self.encode(image_u8, sample)
        rgb, depth, seg_logits = self.decoder(z)
        return SeVAEOutput(rgb, depth, seg_logits, mu, logvar, z)


def sevae_loss(out: SeVAEOutput,
               target_image_u8: torch.Tensor,
               target_depth_m: torch.Tensor,
               target_seg: torch.Tensor,
               class_weights: Optional[torch.Tensor] = None,
               beta: float = BETA_KL,
               lambda_seg: float = LAMBDA_SEG,
               weight_seg_by_proximity: bool = False):
    """Total loss plus a breakdown for logging.

    `target_image_u8`  (B,4,H,W)  CLEAN target -- the input may be noise-corrupted
                                  (denoising VAE), the target never is.
    `target_depth_m`   (B,1,H,W)  metres, drives the proximity weight
    `target_seg`       (B,H,W)    int64 class ids
    """
    tgt = target_image_u8.float() / 255.0
    w = proximity_weight(target_depth_m)                       # (B,1,H,W)

    err_rgb = (out.rgb - tgt[:, :3]).pow(2).mean(1, keepdim=True)
    err_depth = (out.depth - tgt[:, 3:4]).pow(2)
    l_recon = (w * (err_rgb + err_depth)).mean()

    ce = F.cross_entropy(out.seg_logits, target_seg.long(),
                         weight=class_weights, reduction="none")  # (B,H,W)
    if weight_seg_by_proximity:
        ce = ce * w.squeeze(1)
    l_seg = ce.mean()

    # Standard-sign KL. Eq. (1) as printed in the paper is the negative of this;
    # minimizing it as written would blow the posterior up.
    l_kl = -0.5 * (1.0 + out.logvar - out.mu.pow(2) - out.logvar.exp()).sum(1).mean()

    total = l_recon + lambda_seg * l_seg + beta * l_kl
    return total, {"recon": l_recon.item(), "seg": l_seg.item(),
                   "kl": l_kl.item(), "total": total.item()}


def seg_class_weights(counts: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean 1.

    Bars occupy 1-4% of the frame (an 0.08 m bar subtends ~5 px at 2 m and ~1 px
    at 8 m over a 128 px image), so unweighted CE would happily predict "wall"
    everywhere and call it a good day.
    """
    freq = counts.float() / counts.float().sum().clamp_min(floor)
    w = 1.0 / freq.clamp_min(floor)
    return w / w.mean()
