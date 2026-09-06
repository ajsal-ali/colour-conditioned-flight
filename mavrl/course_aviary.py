#!/usr/bin/env python3
"""CourseAviary -- vision-only multi-gate obstacle course.

Built on the repo's BaseAviary (real CF2 model, PID velocity engine, onboard
camera). The course arena is injected at runtime by mavrl.course_world.

Design:
  * Observation = Dict{ image  : RGB-D (128,128,4) uint8, downsampled from a
                                 512x512 render,
                        proprio: 16-d, all onboard-observable }
    No goal pose and no window geometry in the observation -- the corridor is
    straight, so a goal vector would carry almost no information the camera does
    not already provide.
  * Action = body acceleration (3) + yaw rate (1), integrated to a velocity
    setpoint and driven through PIDControl. This is the paper's action space.
  * Policy runs at 10 Hz over a 60 Hz PID over 240 Hz physics. One acceleration
    command is held across 6 inner control ticks and the frame is rendered once
    per policy step, which is what makes a 512 raster affordable.
  * Reward and termination MAY use privileged gate geometry -- that is sim-side
    and never enters the observation.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mavrl import config as C                                    # noqa: E402
from mavrl.course_gates import TRAVEL_SIGN, Outcome              # noqa: E402
from mavrl.course_world import (                                 # noqa: E402
    CourseLayout, ENTRY_Y, EXIT_Y, N_SEM_CLASSES, YAW_DOWN_COURSE,
    build_geom_class_lut, build_gates, make_course_world_injector, spawn_pose,
)
from mavrl.imageproc import build_observation, seg_to_classes    # noqa: E402
from mavrl.sensor_noise import NoiseConfig, corrupt              # noqa: E402

from multi_drone_mujoco.envs import base_aviary as _BA           # noqa: E402
from multi_drone_mujoco.envs.base_aviary import BaseAviary       # noqa: E402
from multi_drone_mujoco.control.pid_control import PIDControl    # noqa: E402
from multi_drone_mujoco.utils.enums import (                     # noqa: E402
    DroneModel, Physics, ActionType, ObservationType)


class CourseAviary(BaseAviary):
    """Multi-gate course: entry window -> 0..3 bars -> exit window."""

    def __init__(self, layout: Optional[CourseLayout] = None,
                 seed: Optional[int] = None,
                 noise: Optional[NoiseConfig] = None,
                 collect_mode: bool = False):
        self.layout = layout if layout is not None else CourseLayout(())
        self.collect_mode = collect_mode
        self.noise = noise if noise is not None else NoiseConfig()
        self._rng = np.random.default_rng(seed)

        self.gates = build_gates(self.layout)
        self._reset_task_state()

        pos, yaw = spawn_pose(self._rng)
        original, patched = make_course_world_injector(self.layout, C.RENDER_RES)
        _BA._generate_aviary_xml = patched
        try:
            super().__init__(
                drone_model=DroneModel.CF2X, num_drones=1,
                initial_xyzs=np.array([pos], dtype=float),
                initial_rpys=np.array([[0.0, 0.0, yaw]]),
                physics=Physics.MJC,
                sim_freq=C.SIM_FREQ, ctrl_freq=C.CTRL_FREQ,
                vision_attributes=True,
                obs_type=ObservationType.KIN, act_type=ActionType.RPM,
            )
        finally:
            _BA._generate_aviary_xml = original

        # Render at RENDER_RES; the observation is downsampled from it.
        self.IMG_RES = np.array([C.RENDER_RES, C.RENDER_RES])
        # Only the collector consumes seg (it is the SeVAE head's target); on the
        # PPO path the pass was rendered and then dropped, ~10% of the step.
        self._render_seg = collect_mode

        self.pid = PIDControl(self)
        self._cache_model_ids()
        self.yaw_sp = float(yaw)
        self._cached_obs = None

        #: When True, every `step` and `reset` info carries `expert_action`: what
        #: the scripted pilot would do from the state the returned observation
        #: describes. RecurrentPPO uses it to fly a decaying fraction of the envs
        #: on demonstrations (see mavrl.ppo, `expert_frac`).
        #:
        #: It rides along in `info` rather than being fetched with `env_method`
        #: because the trainer needs it *before* it acts, and a per-step
        #: round-trip to every worker would cost more than the pilot does. The
        #: pilot is pure numpy over state the env already has.
        self.emit_expert = False
        self._pilot = None

    # -- model / layout ------------------------------------------------------

    def _cache_model_ids(self) -> None:
        self.drone_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")
        self.drone_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "drone0_collision")
        self.geom_class_lut = build_geom_class_lut(self.model)

    def set_layout(self, layout: CourseLayout) -> None:
        """Swap in a new course, rebuilding the MuJoCo model.

        All parallel envs are given the same layout at the same time (see
        mavrl.curriculum), so this is a synchronized barrier rather than
        per-env randomization -- which is what keeps a shared-renderer path
        viable later.
        """
        self.layout = layout
        self.gates = build_gates(layout)

        original, patched = make_course_world_injector(layout, C.RENDER_RES)
        _BA._generate_aviary_xml = patched
        try:
            xml = _BA._generate_aviary_xml(
                num_drones=self.NUM_DRONES,
                drone_model=self.DRONE_MODEL,
                init_xyzs=self.INIT_XYZS,
                init_rpys=self.INIT_RPYS,
                obstacles=False,
                vision=True,
                timestep=self.SIM_TIMESTEP,
            )
        finally:
            _BA._generate_aviary_xml = original

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None        # rebuilt lazily against the new model
        self._cache_model_ids()
        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()

    # -- spaces --------------------------------------------------------------

    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, shape=(C.N_ACTIONS,), dtype=np.float32)

    def _observationSpace(self):
        d = {
            "image": spaces.Box(0, 255,
                                shape=(C.IMG_RES, C.IMG_RES, C.IMG_CHANNELS),
                                dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf,
                                  shape=(C.N_PROPRIO,), dtype=np.float32),
        }
        if self.collect_mode:
            # Only the collector needs these; keeping them out of the RL
            # observation keeps them out of the PPO rollout buffer.
            d["image_gt"] = spaces.Box(0, 255,
                                       shape=(C.IMG_RES, C.IMG_RES,
                                              C.IMG_CHANNELS), dtype=np.uint8)
            d["seg"] = spaces.Box(0, N_SEM_CLASSES - 1,
                                  shape=(C.IMG_RES, C.IMG_RES), dtype=np.uint8)
            d["depth_m"] = spaces.Box(0.0, C.DEPTH_MAX,
                                      shape=(C.IMG_RES, C.IMG_RES),
                                      dtype=np.float32)
        return spaces.Dict(d)

    # -- action: acceleration -> velocity setpoint -> PID --------------------

    def _preprocessAction(self, action):
        a = np.clip(np.asarray(action, dtype=float).flatten(), -1.0, 1.0)
        self.cur_action = a.copy()

        accel_body = a[:3] * np.array(C.A_MAX)
        yaw_rate = a[3] * C.YAW_RATE_MAX

        # Integrate once per *policy* step, not once per inner control tick,
        # or the same command would be integrated CTRL_PER_POLICY times.
        if self._integrate_this_tick:
            self.v_cmd_body = np.clip(
                self.v_cmd_body + accel_body * C.POLICY_DT,
                -np.array(C.V_MAX), np.array(C.V_MAX))
            self._integrate_this_tick = False

        yaw = self.rpy[0, 2]
        cz, sz = math.cos(yaw), math.sin(yaw)
        v_world = np.array([
            cz * self.v_cmd_body[0] - sz * self.v_cmd_body[1],
            sz * self.v_cmd_body[0] + cz * self.v_cmd_body[1],
            self.v_cmd_body[2],
        ])
        self.yaw_sp += yaw_rate * self.CTRL_TIMESTEP

        rpm, _, _ = self.pid.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=self.pos[0], cur_quat=self.quat[0],
            cur_vel=self.vel[0], cur_ang_vel=self.ang_v[0],
            target_pos=self.pos[0],
            target_rpy=np.array([0.0, 0.0, self.yaw_sp]),
            target_vel=v_world)
        rpm = np.clip(rpm, 0, self.MAX_RPM).flatten()
        self.last_rpm = rpm
        return rpm.reshape(1, 4)

    # -- observation ---------------------------------------------------------

    def _computeObs(self):
        if not self._render_this_tick and self._cached_obs is not None:
            return self._cached_obs

        rgb, dep, seg = self._getDroneImages(0)
        rgb = rgb[..., :3]
        rgb_n, dep_n = corrupt(rgb, dep, self.noise, self._rng)
        image, depth_m = build_observation(rgb_n, dep_n, C.IMG_RES)

        R = self.data.xmat[self.drone_bid].reshape(3, 3)     # body -> world
        proprio = np.concatenate([
            R.T @ np.array([0.0, 0.0, -1.0]),                # gravity direction
            R.T @ self.ang_v[0],                             # gyro
            R.T @ self.vel[0],                               # body velocity
            self.v_cmd_body,                                 # integrator state
            self.prev_action,
        ]).astype(np.float32)

        obs = {"image": image, "proprio": proprio}
        if self.collect_mode:
            # Reconstruction targets come from the CLEAN render: the encoder
            # input is corrupted but the target never is, which is what makes
            # this a denoising VAE rather than one that models its own noise.
            image_clean, depth_clean = build_observation(rgb, dep, C.IMG_RES)
            obs["image_gt"] = image_clean
            obs["seg"] = seg_to_classes(seg, self.geom_class_lut, C.IMG_RES)
            obs["depth_m"] = depth_clean.astype(np.float32)

        self._cached_obs = obs
        return obs

    # -- reward / termination ------------------------------------------------

    def _check_collision(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.drone_gid or c.geom2 == self.drone_gid:
                return True
        return False

    def heading_error(self) -> float:
        """Signed yaw error against down-course, wrapped to [-pi, pi]."""
        yaw = self.rpy[0, 2]
        return math.atan2(math.sin(YAW_DOWN_COURSE - yaw),
                          math.cos(YAW_DOWN_COURSE - yaw))

    def _refresh_progress_target(self) -> None:
        wp = self.gates.waypoint()
        self._wp = wp
        self.d_prev = (float(np.linalg.norm(self.pos[0] - wp))
                       if wp is not None else 0.0)

    def _computeReward(self):
        """Per-inner-tick reward. Time and jerk are added once per policy step."""
        pos = self.pos[0]
        rpy = self.rpy[0]
        reward = 0.0

        if self._wp is not None:
            d = float(np.linalg.norm(pos - self._wp))
            # Closing pays only up to V_PROG_CAP; past that the credit is flat,
            # so mid-course speed stops being worth anything on its own and the
            # remaining incentive to hurry is the (uncapped) completion bonus.
            # min() and not clip(): moving AWAY still costs the full distance,
            # otherwise a sawtooth of dash-and-retreat would farm the cap.
            closed = self.d_prev - d
            reward += C.K_PROG * min(closed,
                                     C.V_PROG_CAP * self.CTRL_TIMESTEP)
            self.d_prev = d

            if abs(pos[1] - self.gates.current.y) < C.CENTER_BAND:  # signless
                off = math.hypot(pos[0] - self._wp[0], pos[2] - self._wp[2])
                reward -= C.K_CENTER * off

        crash, reason = False, None
        outcome = self.gates.update(self.prev_pos, pos)

        if outcome is Outcome.PASS:
            self.gates_cleared += 1
            reward += C.R_TOTAL / max(1, self.gates.n_gates)
            if self.gates.done:
                self._success = True
                self._terminal_reason = "completed"
                t = self.step_counter / self.SIM_FREQ
                reward += C.R_FINISH
                reward += C.K_TBONUS * max(
                    0.0, C.t_nominal(self.layout.n_stations) - t)
            self._refresh_progress_target()

        elif outcome in (Outcome.WRONG_GATE, Outcome.WRONG_SIDE, Outcome.BLOCKED):
            self.missed_gates += 1
            reason = outcome.name.lower()
            if self.collect_mode:
                # Keep flying: the SeVAE needs frames from failure states, and
                # ending here would bias the dataset toward clean traversals.
                self._missed.append((self.gates.index, reason))
                self.gates.index += 1
                self._refresh_progress_target()
            else:
                crash = True

        if not self._success and not crash:
            past_entry = TRAVEL_SIGN * (pos[1] - ENTRY_Y) > 0.0
            if self._check_collision():
                crash, reason = True, "collision"
            elif pos[2] < 0.05:
                crash, reason = True, "ground"
            elif abs(rpy[0]) > math.pi / 2 or abs(rpy[1]) > math.pi / 2:
                crash, reason = True, "flip"
            elif past_entry and abs(pos[0]) > C.CORRIDOR_X_LIMIT:
                # The arena has no side walls along the corridor; it is enforced
                # here rather than with geometry so the obstacle set stays
                # exactly the competition's.
                crash, reason = True, "out_of_corridor"
            elif abs(pos[0]) > 7.5:
                # Also sideways, but from before the entry wall, where
                # CORRIDOR_X_LIMIT does not apply yet. Same offence.
                crash, reason = True, "out_of_corridor"
            elif TRAVEL_SIGN * (pos[1] - ENTRY_Y) < -6.0:
                crash, reason = True, "backwards"
            elif pos[2] > 7.0 or TRAVEL_SIGN * (pos[1] - EXIT_Y) > 4.0:
                # Overshoot: ran on more than 4 past the exit, or climbed out
                # of the room. Split from the two above because it is a
                # different failure -- the drone got all the way down the
                # course to do this, so it is priced as a crash, not as
                # straying.
                crash, reason = True, "out_of_bounds"

        if crash:
            if self.collect_mode and reason in (
                    "wrong_gate", "wrong_side", "blocked"):
                crash = False
            else:
                self._crashed = True
                self._terminal_reason = reason
                # Going out the side or backwards costs more than hitting
                # something. Both end the episode, so without the split the
                # policy picks whichever exit is easier to reach, and sideways
                # is always easier than committing to a gate. Overshooting the
                # exit is not in that set -- see C.STRAY_REASONS.
                reward -= (C.R_STRAY if reason in C.STRAY_REASONS
                           else C.R_CRASH)

        self.prev_pos = pos.copy()
        return float(reward)

    def _computeTerminated(self):
        return bool(self._success or self._crashed)

    def _computeTruncated(self):
        limit = C.t_max(self.layout.n_stations)
        return bool(self.step_counter / self.SIM_FREQ >= limit)

    def _computeInfo(self):
        return {
            "is_success": bool(self._success),
            "crashed": bool(self._crashed),
            "terminal_reason": self._terminal_reason,
            "gates_cleared": int(self.gates_cleared),
            "n_gates": int(self.gates.n_gates),
            "missed_gates": int(self.missed_gates),
            "n_stations": int(self.layout.n_stations),
            "agv": float(self._agv()),
            # Degrees, signed. Worth logging: if this trends away from 0 across
            # training, K_YAW is too small to hold heading.
            "heading_err": float(math.degrees(self.heading_error())),
        }

    def expert_action(self) -> np.ndarray:
        """What the scripted pilot would command from the current state.

        Imported lazily: mavrl.pilot is cheap, but the env is constructed in
        every worker process and only the PPO path ever wants a teacher.
        """
        if self._pilot is None:
            from mavrl.pilot import CascadedPilot
            self._pilot = CascadedPilot()
        return self._pilot(self)

    def _agv(self) -> float:
        """Average goal velocity: distance made good along the course, over
        elapsed time. Signed by TRAVEL_SIGN so it survives a direction flip."""
        t = self.step_counter / self.SIM_FREQ
        if t <= 0.0:
            return 0.0
        made_good = TRAVEL_SIGN * (self.pos[0][1] - self._start_y)
        return float(max(0.0, made_good) / t)

    # -- step: hold one action across CTRL_PER_POLICY inner ticks ------------

    def step(self, action):
        self._integrate_this_tick = True
        self._render_this_tick = False

        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(C.CTRL_PER_POLICY):
            _, r, terminated, truncated, info = super().step(action)
            total_reward += r
            if terminated or truncated:
                break

        # One render per policy step, wherever the inner loop stopped.
        self._render_this_tick = True
        obs = self._computeObs()
        self._render_this_tick = False

        total_reward -= C.K_TIME
        total_reward -= C.K_SMOOTH * float(
            np.linalg.norm(self.cur_action - self.prev_action))
        # Hold heading down-course. Applied here, once per policy step, rather
        # than in _computeReward, which runs CTRL_PER_POLICY times -- otherwise
        # K_YAW would silently be six times what it says.
        total_reward -= C.K_YAW * abs(self.heading_error())
        # Speed band: free up to V_PROG_CAP (which is where progress credit
        # stops accruing), unpaid but unpunished to V_SPEED_LIMIT, penalised
        # above it. Horizontal only -- the vertical axis has to stay free for
        # the climbs the station spacing forces. Suppressed once the course is
        # finished, so the uncapped completion bonus is never clawed back from
        # a run that was fast *and* made it.
        if not self._success:
            v_horiz = float(np.linalg.norm(self.vel[0][:2]))
            total_reward -= C.K_OVERSPEED * max(
                0.0, v_horiz - C.V_SPEED_LIMIT)
        self.prev_action = self.cur_action.copy()

        info = self._computeInfo()
        if self.emit_expert:
            info["expert_action"] = self.expert_action()
        return obs, float(total_reward), bool(terminated), bool(truncated), info

    # -- reset ---------------------------------------------------------------

    def _reset_task_state(self) -> None:
        self._success = False
        self._crashed = False
        self._terminal_reason = None
        self.gates_cleared = 0
        self.missed_gates = 0
        self._missed = []
        self.cur_action = np.zeros(C.N_ACTIONS)
        self.prev_action = np.zeros(C.N_ACTIONS)
        self.v_cmd_body = np.zeros(3)
        self._integrate_this_tick = True
        self._render_this_tick = True
        self._start_y = 0.0
        self._wp = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "layout" in options:
            self.set_layout(options["layout"])

        pos, yaw = spawn_pose(self._rng)
        self.INIT_XYZS = np.array([pos], dtype=float)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]])

        self._reset_task_state()
        self.gates.reset()
        self._cached_obs = None

        obs, info = super().reset(seed=seed, options=options)

        self.yaw_sp = float(yaw)
        self.pid.reset()
        self.prev_pos = self.pos[0].copy()
        self._start_y = float(self.pos[0][1])
        self._refresh_progress_target()
        info = self._computeInfo()
        if self.emit_expert:
            # The first action of a fresh episode has no preceding step to carry
            # it, so reset has to supply one or the trainer would be a step late
            # on every episode.
            info["expert_action"] = self.expert_action()
        return obs, info


def make_course_env(layout: Optional[CourseLayout] = None, seed=None,
                    noise: Optional[NoiseConfig] = None,
                    collect_mode: bool = False):
    """Factory for SubprocVecEnv / DummyVecEnv."""
    return CourseAviary(layout=layout, seed=seed, noise=noise,
                        collect_mode=collect_mode)


if __name__ == "__main__":
    from mavrl.course_world import sample_layout

    rng = np.random.default_rng(0)
    lay = sample_layout(rng, 2)
    env = CourseAviary(layout=lay, seed=0)
    obs, info = env.reset(seed=0)
    print("layout   :", lay.describe())
    print("obs image:", obs["image"].shape, obs["image"].dtype,
          "proprio:", obs["proprio"].shape)
    print("action   :", env.action_space)

    t0 = env.step_counter
    obs, r, term, trunc, info = env.step(np.array([0, 1, 0, 0], dtype=np.float32))
    print("sim steps per policy step:", env.step_counter - t0,
          f"(expected {C.SIM_STEPS_PER_POLICY})")

    total = r
    for _ in range(200):
        obs, r, term, trunc, info = env.step(
            np.array([0, 1, -0.3, 0], dtype=np.float32))
        total += r
        if term or trunc:
            break
    print("episode end:", info, "cum_reward=%.2f" % total)
    env.close()
