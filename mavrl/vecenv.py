#!/usr/bin/env python3
"""Vectorized env construction.

Separate module so `ald` and `train_course` do not import each other, and so the
stable-baselines3 import is **deferred into the function**. Data collection and
the offline training stages (SeVAE, memory, BC) need neither SB3 nor a vec env;
making it a module-level import would force it on them anyway.

Rendering is shared by default. One `mujoco.Renderer` per env means one GL
context and one copy of the arena per env, and since every env runs the *same*
layout at any moment (see mavrl.curriculum) those copies are byte-identical --
VRAM, not compute, becomes the ceiling on `--n-envs`. `SharedRenderVecEnv` puts
N envs in M processes with one context each, so VRAM scales with M while the
physics still runs M-ways parallel.
"""

from __future__ import annotations

from typing import Optional

from mavrl.course_world import CourseLayout
from mavrl.sensor_noise import NoiseConfig


def make_env(layout: Optional[CourseLayout], seed: int,
             noise: Optional[NoiseConfig]):
    def _init():
        from mavrl.course_aviary import CourseAviary
        return CourseAviary(layout=layout, seed=seed, noise=noise)
    return _init


def default_render_workers(n_envs: int) -> int:
    """M = ceil(N/4) -- four envs to a GL context.

    A quarter of the contexts is most of the VRAM saving, while leaving enough
    processes that physics is not serialized behind a single one. Override it
    with the wait-time sweep in multi_drone_mujoco/bench/calibrate.py if the
    renderer turns out to be the queue.
    """
    return max(1, -(-int(n_envs) // 4))


def layout_slots(n_envs: int, subproc: bool = True,
                 shared_render: bool = True,
                 render_workers: Optional[int] = None) -> int:
    """How many *distinct* layouts this vec env can hold at once.

    A `SharedStaticRenderer` serves one uploaded model to every env in its
    worker, and asserts they are identical (rendering/shared.py). So under
    shared rendering the granularity is the worker, not the env. Without it each
    env owns its context and can hold its own layout.

    This is what lets one gradient update see many courses instead of one --
    see mavrl.curriculum. It has to be computable *before* the vec env exists,
    because the sampler is built first and has to know how many to draw.
    """
    if not subproc or n_envs <= 1:
        return n_envs                      # DummyVecEnv: a renderer per env
    if not shared_render:
        return n_envs                      # SubprocVecEnv: a process per env
    return min(render_workers or default_render_workers(n_envs), n_envs)


def build_venv(layout: Optional[CourseLayout], n_envs: int, seed: int,
               noise: Optional[NoiseConfig] = None, subproc: bool = True,
               shared_render: bool = True,
               render_workers: Optional[int] = None):
    """SharedRenderVecEnv for n_envs > 1, DummyVecEnv otherwise.

    `shared_render=False` falls back to SubprocVecEnv -- one context per env,
    the previous behaviour, worth keeping as the control when a rendering bug is
    suspected.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env(layout, seed + i, noise) for i in range(n_envs)]
    if not subproc or n_envs <= 1:
        return DummyVecEnv(fns)
    if not shared_render:
        return SubprocVecEnv(fns)

    from multi_drone_mujoco.vec import SharedRenderVecEnv

    workers = render_workers or default_render_workers(n_envs)
    # The training env is never built with collect_mode, so segmentation is
    # rendered and then discarded -- a full extra pass per frame for nothing.
    return SharedRenderVecEnv(fns, n_workers=min(workers, n_envs),
                              want_seg=False)
