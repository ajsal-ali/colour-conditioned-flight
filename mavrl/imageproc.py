#!/usr/bin/env python3
"""512 -> 128 downsampling, depth packing, and the proximity weight.

Numpy only -- no mujoco, no torch -- so the whole path from a rendered frame to
an observation is unit-testable on a machine with no simulator.

Each channel gets the reduction that preserves what it is *for*:

* RGB   -- area average. Standard box downsample; noise-reducing.
* depth -- **min** pool. A mean would blur a thin bar into the background behind
  it and quietly delete the obstacle; taking the nearest surface in each cell is
  both safety-conservative and what keeps a 20 px bar at 512 alive as ~5 px at
  128.
* seg   -- nearest neighbour. Averaging class ids would invent classes that do
  not exist, so any interpolating resize is simply wrong here.
"""

from __future__ import annotations

import numpy as np

from mavrl.config import (
    DEPTH_MAX, D_FAR, D_NEAR, IMG_RES, RENDER_RES, W_MIN,
)


def _factor(src: int, dst: int) -> int:
    if src % dst != 0:
        raise ValueError(f"{src} is not an integer multiple of {dst}")
    return src // dst


def downsample_rgb(rgb: np.ndarray, out: int = IMG_RES) -> np.ndarray:
    """(H, W, 3) uint8 -> (out, out, 3) uint8 by area average.

    Reduced in two contiguous passes instead of one `mean(axis=(1, 3))` over a
    5-D strided view, and accumulated in integers rather than promoting the
    whole 512x512x3 frame to float32 first. Both are exact for uint8 input --
    each block sum is at most f*f*255, which the accumulator holds without loss,
    and the divisor is a power of two -- so the output is bit-identical to the
    float mean it replaces. Measured 6.3 ms -> 1.7 ms per frame at 512.
    """
    h, w = rgb.shape[:2]
    f = _factor(h, out)
    if w // out != f:
        raise ValueError("non-square downsample factor")
    c = rgb.shape[2]
    if rgb.dtype == np.uint8:
        acc = np.uint16 if f * f * 255 <= np.iinfo(np.uint16).max else np.uint32
    else:
        # Not the documented input. Fall back rather than silently truncating a
        # float frame into an integer accumulator.
        acc = np.float32
    block = rgb.reshape(out, f, out * f * c).sum(axis=1, dtype=acc)
    block = block.reshape(out, out, f, c).sum(axis=2, dtype=acc)
    return (block / np.float32(f * f)).round().clip(0, 255).astype(np.uint8)


def downsample_depth(depth: np.ndarray, out: int = IMG_RES) -> np.ndarray:
    """(H, W) float -> (out, out) float by MIN pool (nearest surface wins)."""
    h, w = depth.shape[:2]
    f = _factor(h, out)
    if w // out != f:
        raise ValueError("non-square downsample factor")
    # Two contiguous passes rather than one strided reduction over a 5-D view;
    # min is associative, so this is exactly the same value.
    block = depth.reshape(out, f, out * f).min(axis=1)
    return block.reshape(out, out, f).min(axis=2)


def downsample_seg(seg: np.ndarray, out: int = IMG_RES) -> np.ndarray:
    """(H, W) int -> (out, out) int by nearest neighbour (cell centre)."""
    h, w = seg.shape[:2]
    f = _factor(h, out)
    if w // out != f:
        raise ValueError("non-square downsample factor")
    off = f // 2
    return seg[off::f, off::f][:out, :out]


def depth_to_uint8(depth_m: np.ndarray, depth_max: float = DEPTH_MAX) -> np.ndarray:
    """Metres -> uint8, clipped at `depth_max`."""
    scaled = np.clip(depth_m, 0.0, depth_max) / depth_max * 255.0
    return scaled.astype(np.uint8)


def uint8_to_depth(depth_u8: np.ndarray, depth_max: float = DEPTH_MAX) -> np.ndarray:
    """Inverse of `depth_to_uint8`, for viewers and the proximity weight."""
    return depth_u8.astype(np.float32) / 255.0 * depth_max


def proximity_weight(depth_m: np.ndarray) -> np.ndarray:
    """Per-pixel reconstruction weight: near geometry matters more.

        t = clip((d - D_NEAR) / (D_FAR - D_NEAR), 0, 1)
        w = W_MIN + (1 - W_MIN) * (1 - t)**2

    Monotonically decreasing from D_NEAR outward, flooring at W_MIN (never zero,
    so the far field is de-emphasised rather than discarded).
    """
    t = np.clip((depth_m - D_NEAR) / (D_FAR - D_NEAR), 0.0, 1.0)
    return W_MIN + (1.0 - W_MIN) * (1.0 - t) ** 2


def build_observation(rgb: np.ndarray, depth_m: np.ndarray,
                      out: int = IMG_RES) -> tuple:
    """Render buffers -> (image uint8 (out,out,4), depth metres (out,out)).

    The metric depth is returned alongside because the proximity weight needs
    metres, and recovering them from the uint8 channel would re-quantize.
    """
    rgb_s = downsample_rgb(rgb, out)
    # MuJoCo returns the far-clip distance for sky pixels (order 1e3 m), so the
    # metric buffer is clipped to DEPTH_MAX here rather than left raw -- it is
    # declared in the observation space and stored as float16 in the dataset,
    # and an unclipped 1130 m would be both out of spec and wasteful.
    depth_s = np.minimum(downsample_depth(depth_m, out), DEPTH_MAX)
    image = np.concatenate([rgb_s, depth_to_uint8(depth_s)[..., None]], axis=2)
    return image, depth_s


def seg_to_classes(seg_buf: np.ndarray, geom_class_lut: np.ndarray,
                   out: int = IMG_RES) -> np.ndarray:
    """MuJoCo segmentation buffer -> downsampled semantic class map.

    `mujoco.Renderer` in segmentation mode returns (H, W, 2) int32 -- channel 0
    is the object id, channel 1 the object type. The zero fallback in
    BaseAviary._getDroneImages is (H, W) instead, so both shapes have to be
    accepted here.
    """
    if seg_buf.ndim == 3:
        objid = seg_buf[..., 0]
    else:
        objid = seg_buf
    objid = downsample_seg(objid, out)
    # -1 marks "no object" (sky); anything out of range falls back to class 0.
    valid = (objid >= 0) & (objid < len(geom_class_lut))
    classes = np.zeros(objid.shape, dtype=np.uint8)
    classes[valid] = geom_class_lut[objid[valid]]
    return classes
