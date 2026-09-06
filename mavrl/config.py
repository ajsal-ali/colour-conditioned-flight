#!/usr/bin/env python3
"""Shared constants for the MAVRL pipeline.

One module so the env, the dataset writer, the SeVAE and the notebook cannot
drift apart on image size, depth scaling or the proximity-weight curve. Nothing
here imports mujoco or torch.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------

#: MuJoCo renders at this resolution...
RENDER_RES = 512
#: ...and is downsampled to this before it reaches the encoder.
#: The gap is not vanity: an 0.08 m bar subtends 9.78/d pixels at 128 px
#: (~5 px at 2 m, ~1 px at 8 m), so rendering natively at 128 aliases thin
#: bars away. Rendering at 512 and min-pooling preserves them.
IMG_RES = 128

#: Depth clip, in SDF units (the arena is the competition course scaled 2.2x).
#: The course is 15.4 units from entry wall to exit wall and stations sit 2.20
#: apart, so 16 keeps the next two stations inside the sensor while still
#: cutting the far room off. The proximity weight below runs to the same value.
DEPTH_MAX = 16.0

IMG_CHANNELS = 4        # RGB + depth

# --------------------------------------------------------------------------
# Proximity weight for the SeVAE reconstruction loss
#
#   t = clip((d - D_NEAR) / (D_FAR - D_NEAR), 0, 1)
#   w = W_MIN + (1 - W_MIN) * (1 - t)**2
#
# Distances are SDF units; divide by 2.2 for competition metres. 1.00 at
# contact, ~0.77 at 2.2 (one station spacing), ~0.31 at 6.6 (three), 0.05 floor
# from 16 out. Quadratic rather than exponential so the decay stays gradual
# through the mid-field where the *next* station first becomes visible;
# exp(-d/2) would already be at 0.05 by 6.
# --------------------------------------------------------------------------

W_MIN = 0.05
D_NEAR = 0.5
D_FAR = 16.0

# --------------------------------------------------------------------------
# Control rates
#
# 240 Hz physics / 60 Hz PID / 10 Hz policy. The policy holds one acceleration
# command across CTRL_PER_POLICY inner ticks; rendering happens once per policy
# step, which is what makes the 512 raster affordable.
# --------------------------------------------------------------------------

SIM_FREQ = 240
CTRL_FREQ = 60
POLICY_FREQ = 10
CTRL_PER_POLICY = CTRL_FREQ // POLICY_FREQ          # 6
SIM_STEPS_PER_POLICY = SIM_FREQ // POLICY_FREQ      # 24
POLICY_DT = 1.0 / POLICY_FREQ

# --------------------------------------------------------------------------
# Action space: body acceleration + yaw rate, integrated to a velocity setpoint
#
# Z authority is deliberately generous, and it has to be: consecutive stations
# are only 2.20 apart, so a blue bar at 0.880 followed by a red bar at 4.356 is
# a 3.9-unit climb inside one spacing. That is not flyable at full forward
# speed, which is the point -- MAVRL's whole thesis is that the policy must
# slow down for the tight sections. The scripted pilot throttles its forward
# speed by the pending vertical error for the same reason.
# --------------------------------------------------------------------------

A_MAX = (4.0, 4.0, 6.0)      # m/s^2, body frame
V_MAX = (3.0, 3.0, 3.0)      # m/s, body frame
YAW_RATE_MAX = 1.0           # rad/s

N_PROPRIO = 16               # grav(3) gyro(3) vel(3) v_cmd(3) prev_action(4)
N_ACTIONS = 4

# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------

K_PROG = 1.0
K_TIME = 0.03
K_SMOOTH = 0.01

#: Speed above which closing on the waypoint stops paying, in the SAME units as
#: the logged `agv` (SDF units/s; the arena is the course x2.2, so 1.3 here is
#: ~0.59 competition m/s). Below the cap the progress term is the usual
#: distance-made-good; above it the per-tick credit is clipped flat.
#:
#: The point is to move the speed incentive to where it belongs. Racing between
#: gates buys nothing and costs control margin on the next bar, so mid-course
#: speed is worth exactly one thing -- arriving. The completion bonus
#: (K_TBONUS, below) stays uncapped, so a *finished* course still pays for
#: every second saved, with no ceiling.
#:
#: Retreating is deliberately NOT clipped: backing away from the waypoint costs
#: the full distance, so the cap cannot be farmed by oscillating.
V_PROG_CAP = 1.0

#: Hard end of the speed band. Between V_PROG_CAP and this, speed is free but
#: unpaid; above it, it costs K_OVERSPEED per unit of excess per policy step.
#:
#: Measured on the HORIZONTAL speed only. Vertical is left alone deliberately:
#: a blue@0.880 -> red@4.356 pair is a 3.9-unit climb inside one 2.20 spacing,
#: so a penalty on total speed would punish the climb the course itself
#: demands. What we are limiting is how fast it travels down the corridor.
#:
#: Applied only while the course is unfinished. Crossing the last gate ends the
#: episode and pays K_TBONUS with no ceiling, so a completed run is still
#: rewarded for every second saved no matter how fast it flew.
V_SPEED_LIMIT = 1.3

#: Penalty per policy step per unit of horizontal speed above V_SPEED_LIMIT.
#:
#: 0.5 puts 2.0 units/s at 0.15/step (0.5/s of flight, ~1.7x K_TIME) and the
#: 3.0 ceiling at 0.65/step (6.5/s, ~22x K_TIME). So drifting just over the
#: line is a nudge and flat out is unaffordable, which is the shape we want:
#: the band is 1.3-1.7, not a wall at 1.7.
K_OVERSPEED = 0.5
K_CENTER = 0.1
CENTER_BAND = 2.2            # apply the centering term within one station spacing

#: Penalty per policy step on |heading error|, radians, against YAW_DOWN_COURSE.
#:
#: Yaw is a real action (a[3], a rate) feeding an integrator with no restoring
#: term, and absolute heading is *not* in the proprio vector -- gravity in the
#: body frame is invariant to rotation about z. So without this the policy has
#: no gradient telling it to hold heading and any constant bias on a[3] just
#: accumulates. Every scripted pilot pins yaw (collect.py:181, teleop.py:111),
#: so BC data never shows the drift and only PPO discovers it.
#:
#: 0.1 puts a 30 deg error at 0.052/step, ~1.7x K_TIME, and a 90 deg error at
#: 0.157/step -- roughly half a second of the time penalty, every step it is
#: held. Firm enough that sustained drift is never worth it, still far short of
#: a hard lock: a brief yaw excursion to line up a bar costs well under one
#: gate. The paper's own config carries the equivalent term (`yaw_coeff:
#: -0.003` in mavrloriginal/configs/control/config_new_out.yaml).
K_YAW = 0.1

R_TOTAL = 150.0              # split evenly across the course's gates
R_FINISH = 100.0

#: Must stay ABOVE the per-gate payout, which is R_TOTAL / n_gates -- 30.0 at
#: stage 3 (5 gates), 37.5 at stage 2, 50.0 at stage 1. At the old 25.0 the
#: arithmetic said to gamble: attempting a marginal gate and crashing scored
#: +5 against stopping safely, so "try it and die" beat "hold position", and
#: the terminal-reason mix filled up with `collision`. At 40.0 a failed attempt
#: costs more than the gate it was reaching for at stage 3, without being so
#: steep that the policy learns to sit still -- hovering to timeout is ~-11.7,
#: so one gate then a crash (-10) is still roughly break-even against doing
#: nothing, and two gates then a crash is clearly better. Watch the
#: terminal-reason mix: `collision` dominating means raise it, `timeout`
#: dominating means it is already too high.
R_CRASH = 40.0

#: Leaving the course *sideways or backwards*. Twice R_CRASH on purpose.
#:
#: These used to cost the same as hitting a bar, which made them
#: interchangeable, and peeling off sideways is the cheaper way to end an
#: episode you have decided is lost: it is available from anywhere, at any
#: speed, while a crash at least requires committing to the gate first. In a
#: real venue they are not interchangeable -- clipping a bar is a failed
#: attempt, leaving the course sideways is a drone in the crowd. 80 makes
#: flying into the obstacle strictly preferable to flying out of the arena.
R_STRAY = 80.0

#: Which terminal reasons R_STRAY applies to, rather than R_CRASH.
#:
#: Deliberately NOT the whole out-of-bounds family. Overshooting forward past
#: the exit, or climbing out of the room, is priced as an ordinary crash: the
#: drone got all the way down the course to do it, and penalising that at the
#: sideways rate would punish exactly the runs that came closest to finishing.
#: What earns the doubled penalty is going somewhere the course never went --
#: out the side, or back the way it came.
STRAY_REASONS = ("out_of_corridor", "backwards")

K_TBONUS = 5.0

#: Terminate past the entry wall if |x| exceeds this. The red bar spans
#: x = +-2.20 and the station posts sit at +-1.65, so anything beyond 2.6 has
#: left the course rather than dodged within it.
CORRIDOR_X_LIMIT = 2.6

#: Spawn standoff plus entry-to-exit distance, in SDF units. Imported from
#: course_world would be circular, so it is restated here and pinned by a test.
COURSE_LENGTH = 2.20 + (3.30 - (-12.10))          # 17.6


def t_nominal(n_stations: int) -> float:
    """Reference time for the completion speed bonus.

    17.6 units at a 1.6 average, plus a second and a half per bar for the
    climb/descent each one forces.
    """
    return COURSE_LENGTH / 1.6 + 1.5 * n_stations


def t_max(n_stations: int) -> float:
    """Episode time limit, seconds. Generous: the course is 2.2x longer than a
    competition lap, and a policy that has to slow for a tight pair of stations
    must not be truncated for doing the right thing."""
    return COURSE_LENGTH / 0.65 + 4.0 * n_stations


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

LATENT_DIM = 64              # N_e, VAE latent
MEMORY_DIM = 256             # N_l, memory output
ATTN_WINDOW = 16             # K -- deliberately < AUX_T, see below
N_MEM_TOKENS = 4             # RMT recurrent memory tokens
ATTN_HEADS = 4
ATTN_LAYERS = 2

#: Auxiliary reconstruction offset. NOT a memory window -- it is the prediction
#: target ("from z_t, rebuild the image from T steps ago"), which pressures the
#: recurrent state into retaining that much history.
#:
#: AUX_T > ATTN_WINDOW on purpose: if the attention window could still see step
#: t-T, the aux loss would be satisfiable by copying rather than remembering,
#: and only the LSTM would face a real memory task.
AUX_T = 20

#: Paper Eq. (2): the head emits 3*N_e and splits it into past / current /
#: future segments, with lambda_i in {0,1} choosing which are supervised. This
#: is the lambda configuration, by name.
#:
#: Default is (past, current) -- lambda = (1, 1, 0). Fig. 2(b) of the paper shows
#: the future reconstruction is by far the blurriest of the three ("highest MAE
#: loss"), which is what you would expect: the drone cannot see what is 20 steps
#: ahead, so that segment is partly unlearnable and mostly adds gradient noise.
#: Set `--aux-segments past,current,future` to reproduce the full three-way loss.
AUX_SEGMENTS = ("past", "current")

#: Offset in policy steps of each segment relative to t.
AUX_OFFSETS = {"past": -AUX_T, "current": 0, "future": AUX_T}

BETA_KL = 1e-4
LAMBDA_SEG = 1.0

# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------

SUCCESS_WINDOW = 200         # episodes in the rolling success estimate
SUCCESS_THRESHOLD = 0.8      # advance a stage above this
#: Redraw the layout this often, in rollouts.
#:
#: Was 200. At the top stage a redraw is the *only* thing that varies colour
#: order, so 200 meant ~1.8M steps of a single course -- one sample of the task
#: distribution per gradient update, for hundreds of updates. Memorising it is
#: then the optimal response, and the first run to try it did exactly that:
#: 4/5 gates held for 400 rollouts, then a permanent collapse to `wrong_side`
#: 97% the moment rollout 400 drew a new colour order.
#:
#: 5 is short enough that the colour order cannot be baked in and long enough
#: that a swap (which recompiles every worker's MjModel and rebuilds its GL
#: context) stays a few percent of rollout time. It works with the per-group
#: layouts in mavrl.curriculum, which is what supplies variety *within* an
#: update rather than merely across them.
RESAMPLE_EVERY_ROLLOUTS = 5
MAX_DWELL_ROLLOUTS = 400     # force-advance if a stage stalls
