#!/usr/bin/env python3
"""Curriculum over course complexity, with synchronized layout switching.

Two triggers, as specified:
  * a stage advances once rolling success clears a threshold -- progression is
    earned, not timed;
  * within a stage only the **bar heights** resample, every K rollouts, so the
    policy sees variety without the difficulty running ahead of it.

Station count and colour order change only when the stage does. Resampling them
every K rollouts as well would move three things at once, and a course that is
suddenly blue-first is a different task, not a variation on the same one.

**One layout per render worker.** The sampler produces `n_layouts` of them and
each is broadcast to one worker's whole group at a rollout boundary. The
constraint a shared renderer actually imposes is that envs sharing a *context*
run identical models (rendering/shared.py) -- not that the whole vec env does.
Honouring it at the group granularity instead of globally is what puts many
courses inside one gradient update.

That distinction is not cosmetic. With a single global layout every update saw
exactly one colour order, and memorising it was the optimal response: the first
run to reach 5.6M steps held 4/5 gates for 400 rollouts, then collapsed
permanently to `wrong_side` 97% the moment a redraw changed the order.

A max-dwell fallback exists because a stalled stage is otherwise silent: the run
sits at stage 2 forever and the only symptom is a success curve that stopped
moving.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mavrl.config import (
    MAX_DWELL_ROLLOUTS, RESAMPLE_EVERY_ROLLOUTS, SUCCESS_THRESHOLD,
    SUCCESS_WINDOW,
)
from mavrl.course_world import (
    STAGE_STATIONS, CourseLayout, resample_heights, sample_layout_for_stage,
)


@dataclass
class CurriculumState:
    stage: int = 0
    rollouts_in_stage: int = 0
    layouts: List[CourseLayout] = field(default_factory=list)
    advanced_at: List[int] = field(default_factory=list)


class CourseSampler:
    def __init__(self, seed: Optional[int] = None, split: str = "train",
                 start_stage: int = 0,
                 success_window: int = SUCCESS_WINDOW,
                 success_threshold: float = SUCCESS_THRESHOLD,
                 resample_every: int = RESAMPLE_EVERY_ROLLOUTS,
                 max_dwell: int = MAX_DWELL_ROLLOUTS,
                 n_layouts: int = 1):
        self.rng = np.random.default_rng(seed)
        self.split = split
        self.success_threshold = success_threshold
        self.resample_every = resample_every
        self.max_dwell = max_dwell
        self.n_layouts = max(1, int(n_layouts))
        self.successes = deque(maxlen=success_window)
        self.state = CurriculumState(stage=start_stage)
        self.state.layouts = self._sample()

    # -- bookkeeping ---------------------------------------------------------

    def _sample(self) -> List[CourseLayout]:
        return [sample_layout_for_stage(self.rng, self.state.stage, self.split)
                for _ in range(self.n_layouts)]

    @property
    def layouts(self) -> List[CourseLayout]:
        return self.state.layouts

    @property
    def layout(self) -> CourseLayout:
        """The first layout. Only for building the vec env, which needs *a* world
        before `broadcast_layout` hands each group its own."""
        return self.state.layouts[0]

    @property
    def stage(self) -> int:
        return self.state.stage

    @property
    def max_stage(self) -> int:
        return len(STAGE_STATIONS) - 1

    @property
    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return float(np.mean(self.successes))

    def record_episodes(self, infos) -> None:
        for info in infos:
            if info is None:
                continue
            self.successes.append(bool(info.get("is_success", False)))

    # -- the two triggers ----------------------------------------------------

    def on_rollout_end(self) -> Optional[List[CourseLayout]]:
        """Returns one new layout per group if they should be broadcast, else None."""
        self.state.rollouts_in_stage += 1

        ready = (len(self.successes) >= self.successes.maxlen
                 and self.success_rate >= self.success_threshold)
        stalled = self.state.rollouts_in_stage >= self.max_dwell

        if (ready or stalled) and self.state.stage < self.max_stage:
            self.state.stage += 1
            self.state.rollouts_in_stage = 0
            self.state.advanced_at.append(self.state.stage)
            self.successes.clear()
            self.state.layouts = self._sample()
            return self.state.layouts

        if self.state.rollouts_in_stage % self.resample_every == 0:
            if not self.state.layouts[0].stations:
                # Stage 0 has no bars, so there is no height to redraw. Sending
                # the identical layout anyway would rebuild every worker's
                # MjModel -- and its GL context -- for nothing.
                return None
            if self.state.stage >= self.max_stage:
                # Terminal stage: resample the colour order too.
                #
                # Heights-only is right while the curriculum is still moving --
                # a stage change is what varies colour order, and varying both
                # at once makes an improvement unattributable. But at the top
                # stage no further advance ever comes, so heights-only freezes
                # the colour sequence for the entire run and the policy can
                # memorise "up, down, up" instead of learning that red means
                # above. Anyone training with --stage 3 sits here from step 0.
                self.state.layouts = self._sample()
            else:
                self.state.layouts = [
                    resample_heights(self.rng, lay, self.split)
                    for lay in self.state.layouts]
            return self.state.layouts

        return None

    # -- reporting -----------------------------------------------------------

    def describe(self) -> str:
        extra = (f" +{len(self.state.layouts) - 1}"
                 if len(self.state.layouts) > 1 else "")
        return (f"stage {self.state.stage}/{self.max_stage} "
                f"({STAGE_STATIONS[self.state.stage]} bars) "
                f"success={self.success_rate:.2f} "
                f"n={len(self.successes)} "
                f"layout={self.state.layouts[0].describe()}{extra}")


def layout_index_groups(venv) -> List[List[int]]:
    """Env indices grouped by what must share a layout.

    `SharedRenderVecEnv` publishes its groups; anything else (Dummy, Subproc)
    gives every env its own renderer, so each is its own group.
    """
    groups = getattr(venv, "groups", None)
    if groups is not None:
        return [list(g) for g in groups]
    return [[i] for i in range(venv.num_envs)]


def broadcast_layout(venv, layouts) -> None:
    """Hand each group its own layout.

    Called at a rollout boundary, never mid-rollout: swapping geometry inside a
    rollout would invalidate the value estimates already collected against the
    old course.

    Fewer layouts than groups is cycled rather than rejected, so `n_layouts=1`
    reproduces the old global broadcast exactly.
    """
    if isinstance(layouts, CourseLayout):
        layouts = [layouts]
    layouts = list(layouts)
    for g, idx in enumerate(layout_index_groups(venv)):
        # Address the group's *first* env only. On SharedRenderVecEnv the worker
        # applies an env_method to every env it owns and then re-adopts the
        # recompiled model once (`_resync`), so one call per group is both
        # sufficient and required -- addressing them individually would rebuild
        # the same context once per env.
        venv.env_method("set_layout", layouts[g % len(layouts)],
                        indices=idx[:1])
