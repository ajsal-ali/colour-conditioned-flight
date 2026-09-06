#!/usr/bin/env python3
"""The scripted privileged pilot.

Lives in its own module because two very different callers need it and neither
may import the other: `mavrl.collect` drives it to generate BC shards, and
`mavrl.course_aviary` calls it per step to emit a demonstration action alongside
the observation (see `CourseAviary.emit_expert`). Keeping it in collect.py made
that second use a circular import, since collect.py imports the env.

It is *privileged*: it reads `env.pos`, `env.vel`, `env.rpy` and `env.gates`
directly rather than the camera. That is the point -- it is a teacher, not a
policy.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from mavrl import config as C
from mavrl.course_gates import TRAVEL_SIGN
from mavrl.course_world import RED_Z_HI, RED_Z_LO, TUBE_VIAS, YAW_DOWN_COURSE


#: Distance before a gate plane at which the legal altitude must already be
#: held. One third of a station spacing.
APPROACH_DIST = 0.75


def gate_targets(gate) -> list:
    """Approach point then gate centre, both on the legal side.

    The approach point is the whole trick: it forces the altitude to be correct
    *before* the plane is crossed. Aiming straight at a bar's centre from above
    would cross the plane while still descending, which is exactly the
    wrong-side failure the dense reward's waypoint placement also guards against.
    """
    c = gate.center
    return [np.array([c[0], gate.y - TRAVEL_SIGN * APPROACH_DIST, c[2]]),
            c.copy()]


class CascadedPilot:
    """P on position -> velocity target; P on velocity -> acceleration.

    The velocity target is **deceleration-limited**: per axis it never exceeds
    `sqrt(2 * a_max * |error|)`, the fastest approach from which the drone can
    still stop on the target.

    Without that limit a plain P-P cascade overshoots badly here, because the
    command passes through three lags -- pilot acceleration, the env's
    integration into a velocity setpoint, and the PID tracking that setpoint.
    Measured overshoot before the fix: lateral excursion to x = 0.45 against an
    opening edge at 0.44, and arrival at the exit window at z = 1.06 against a
    sill at 1.06. Both scraped the frame.
    """

    #: Per-axis braking conservatism (world x, y, z). Z is tightest because a
    #: descent has to be arrested by thrust exceeding weight, so the achievable
    #: deceleration is well below A_MAX.
    BRAKE_SAFETY = (0.5, 0.5, 0.35)

    #: Forward speed is throttled by the vertical error still outstanding.
    #: Stations are 2.20 apart and a blue@0.880 followed by a red@4.356 is a
    #: 3.9-unit climb inside that spacing -- flat out, the plane arrives before
    #: the altitude does. This is the scripted stand-in for the varying-speed
    #: behaviour the policy is supposed to learn.
    CLIMB_SLOWDOWN = 0.55

    #: The pilot's own speed cap, per world axis, well under C.V_MAX = 3.0.
    #:
    #: This is a BC-data decision, not a flight-envelope one. At V_MAX the
    #: teacher spends its whole run against the acceleration limit (see kp_vel
    #: below), so the demonstrations are bang-bang and there is nothing in the
    #: middle for a Gaussian policy to regress onto. Cruising at V_SPEED_LIMIT
    #: keeps the commanded acceleration inside +-1 and the labels smooth.
    #:
    #: Tied to C.V_SPEED_LIMIT rather than written as a number, because the two
    #: have to agree: the env docks K_OVERSPEED per step per unit of horizontal
    #: speed above that line, so a teacher that cruises above it demonstrates
    #: behaviour the reward is actively punishing. Whichever way this is used --
    #: BC targets, or the in-buffer expert of mavrl.ppo -- that is a teacher
    #: pulling against the objective.
    #:
    #: Z is exempt on purpose, exactly as the penalty is: the penalty measures
    #: horizontal speed only, because a blue@0.880 -> red@4.356 pair is a
    #: 3.9-unit climb inside one 2.20 spacing and capping the climb would make
    #: that pair unflyable rather than merely slow.
    V_CRUISE = (C.V_SPEED_LIMIT, C.V_SPEED_LIMIT, 2.0)

    #: Fraction of C.V_SPEED_LIMIT the pilot actually aims for.
    #:
    #: The per-axis cap above is not enough on its own: it is applied per axis,
    #: so a diagonal command of (1.3, 1.3) is a horizontal speed of 1.84, and
    #: the penalty is measured on the norm. `_velocity_target` therefore also
    #: clamps the horizontal *norm*, and does it at 0.9 of the limit so the
    #: tracked velocity has room to overshoot the setpoint -- the command passes
    #: through the env's integration and then the PID, and arriving exactly at
    #: the line means crossing it on every transient.
    SPEED_MARGIN = 0.9

    def __init__(self, kp_pos: float = 1.6, kp_vel: float = 2.5,
                 kp_yaw: float = 2.0, brake_safety=None,
                 climb_slowdown: Optional[float] = None,
                 v_cruise=None, speed_margin: Optional[float] = None):
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.v_cruise = (self.V_CRUISE if v_cruise is None else v_cruise)
        self.horiz_limit = C.V_SPEED_LIMIT * (
            self.SPEED_MARGIN if speed_margin is None else speed_margin)
        self.kp_yaw = kp_yaw
        self.brake_safety = np.asarray(
            brake_safety if brake_safety is not None else self.BRAKE_SAFETY,
            dtype=float)
        self.climb_slowdown = (self.CLIMB_SLOWDOWN if climb_slowdown is None
                               else climb_slowdown)

    def _velocity_target(self, err: np.ndarray) -> np.ndarray:
        a_max = np.array(C.A_MAX)
        v_max = np.minimum(np.array(C.V_MAX), np.array(self.v_cruise))
        v_brake = self.brake_safety * np.sqrt(2.0 * a_max * np.abs(err))
        speed = np.minimum(np.abs(self.kp_pos * err),
                           np.minimum(v_max, v_brake))
        v = np.sign(err) * speed

        # Throttle the along-course axis by how much altitude is still owed, so
        # the drone arrives at the plane already at the legal height instead of
        # arriving first and correcting after.
        v_z_needed = min(abs(v[2]), v_max[2])
        if v_z_needed > 1e-6:
            scale = 1.0 / (1.0 + self.climb_slowdown * v_z_needed)
            v[1] *= scale

        # Clamp the horizontal *norm*, not the axes. C.K_OVERSPEED is charged on
        # hypot(vx, vy), so two axes each legally at the cap are together over
        # it. Scaling both by the same factor preserves the heading of the
        # command -- normalising per axis would steer the drone.
        horiz = float(np.hypot(v[0], v[1]))
        if horiz > self.horiz_limit > 0.0:
            v[:2] *= self.horiz_limit / horiz
        return v

    def target(self, env) -> np.ndarray:
        """Approach point until the drone is past it, then the gate centre --
        unless a fixed tube obstacle stands between here and that target, in
        which case thread the tube first.

        The pilot has no obstacle avoidance; it chains gate waypoints. That is
        fine for gates but not for `tube_B`, which sits between the last bar and
        the exit window with a post 0.22 from the window's centreline.
        """
        gate = env.gates.current
        if gate is None:
            # Course cleared: keep going straight out past the exit wall.
            return np.array([0.0, env.layout.exit_y + TRAVEL_SIGN * 2.0,
                             0.5 * (RED_Z_LO + RED_Z_HI)])
        approach, centre = gate_targets(gate)
        reached = TRAVEL_SIGN * (env.pos[0][1] - approach[1]) >= 0.0
        tgt = centre if reached else approach

        y = env.pos[0][1]
        for via_y, via_x in TUBE_VIAS:
            ahead = TRAVEL_SIGN * (via_y - y) > 0.0
            before_target = TRAVEL_SIGN * (tgt[1] - via_y) > 0.0
            if ahead and before_target:
                # Hold the target altitude through the tube, so the only thing
                # left to do after it is close the lateral gap to the window.
                return np.array([via_x, via_y, tgt[2]])
        return tgt

    def __call__(self, env) -> np.ndarray:
        pos = env.pos[0]
        vel = env.vel[0]
        yaw = env.rpy[0, 2]

        tgt = self.target(env)
        v_des = self._velocity_target(tgt - pos)

        cz, sz = math.cos(-yaw), math.sin(-yaw)
        v_des_body = np.array([cz * v_des[0] - sz * v_des[1],
                               sz * v_des[0] + cz * v_des[1],
                               v_des[2]])

        # P on velocity error, NOT deadbeat. `(v_des - v_cmd) / POLICY_DT` is
        # gain 10/s: it demands the whole error be erased in one policy step,
        # which saturates the action at +-1 for any error above A_MAX*dt =
        # 0.40 units/s. Measured spin-up from rest to 3.0 was seven consecutive
        # steps at exactly +1.00, then 0.00 -- bang-bang. Those are unlearnable
        # BC targets: regressing a Gaussian on a two-valued signal returns the
        # mean, which is the collapse the check at the end of bc.py warns about.
        #
        # kp_vel = 2.5 saturates only above 1.6 units/s, so paired with
        # V_CRUISE the teacher stays in its linear region and the labels carry
        # real structure. The overshoot that motivated deadbeat (lateral
        # excursion to x = -1.02 against an opening edge at -0.44) came from a
        # soft gain at *full* cruise; the speed cap is what makes this safe, so
        # the two changes only work together. Watch the completion rate.
        a_body = self.kp_vel * (v_des_body - env.v_cmd_body)

        yaw_err = math.atan2(math.sin(YAW_DOWN_COURSE - yaw),
                             math.cos(YAW_DOWN_COURSE - yaw))
        action = np.empty(4, dtype=np.float32)
        action[:3] = np.clip(a_body / np.array(C.A_MAX), -1.0, 1.0)
        action[3] = np.clip(self.kp_yaw * yaw_err / C.YAW_RATE_MAX, -1.0, 1.0)
        return action
