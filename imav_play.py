#!/usr/bin/env python3
"""Fly the repo's Crazyflie in the IMAV2026 world — full interactive 3rd-person
MuJoCo viewer, plus a live onboard RGB + depth window.

Nothing invented: the package's own BaseAviary CF2X drone, its own PIDControl
velocity engine, and its own onboard-camera renderer (env._getDroneImages). The
only addition is the world — the IMAV2026 SDF arena converted to MJCF
(imav_teleop.SdfToMjcf) and injected into the package's drone XML.

The 3rd-person view is the standard interactive mujoco.viewer (mouse orbit / pan /
zoom), exactly like the pid demos. The onboard RGB + depth are shown in a separate
window that runs in its own process — because the GLFW viewer and matplotlib's Tk
window deadlock if run in the same process.

Run inside the rl_mujoco conda env:
    conda run -n rl_mujoco python imav_play.py

Velocity teleop — HOLD a key to fly, release and the drone brakes to rest:
    W / S        forward / backward
    A / D        left / right
    Up / Down    up / down
    Left / Right yaw left / right
    H            hover instantly (zero all commands)
    R            reset to spawn
    ESC          quit
Only velocity commands are sent (target_vel; no position targets). Held keys are
detected via OS key auto-repeat, so one key at a time flies best; releasing all
keys lets the command decay and the drone comes to rest on its own.
Mouse orbit / pan / zoom work in the 3rd-person window as usual.
"""

import argparse
import math
import queue
import time
import multiprocessing as mp
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from imav_teleop import SdfToMjcf, DEFAULT_SDF

from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType
from multi_drone_mujoco.control.pid_control import PIDControl

# --- visual spinning propellers -------------------------------------------
# The package applies thrust as a body force; its CF2 prop meshes are rigidly
# attached and never move, so a correct hover looks frozen. These four disks
# are purely cosmetic: injected into the drone body and spun kinematically
# (qpos written directly, qvel kept 0 so the solver never touches them).
ARM_L = 0.0397                       # CF2X motor-to-center (DRONE_PARAMS)
PROP_R = 0.0232                      # prop radius
_D = ARM_L / math.sqrt(2)
PROP_OFFSETS = [(_D, _D), (-_D, _D), (-_D, -_D), (_D, -_D)]   # X config
SPIN_DIRS = [1.0, -1.0, 1.0, -1.0]   # alternating CW/CCW
VIS_SPIN = 55.0                      # rad/s at hover RPM (clearly visible)


def _inject_props(drone_body: ET.Element):
    for i, (px, py) in enumerate(PROP_OFFSETS):
        drone_body.append(ET.fromstring(
            f'<body name="drone0_prop{i}" pos="{px:.5f} {py:.5f} 0.008">'
            f'<joint name="drone0_prop{i}_spin" type="hinge" axis="0 0 1" limited="false"/>'
            f'<inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>'
            f'<geom type="cylinder" size="{PROP_R:.4f} 0.0015" '
            f'rgba="0.08 0.08 0.08 0.5" contype="0" conaffinity="0"/>'
            f'<geom type="box" size="{PROP_R:.4f} 0.0015 0.002" '
            f'rgba="0.55 0.55 0.6 0.9" contype="0" conaffinity="0"/>'
            f'</body>'))


def make_world_injector(sdf_path: Path, include_dolls: bool):
    """Replacement for base_aviary._generate_aviary_xml: append the IMAV world."""
    conv = SdfToMjcf(sdf_path, include_dolls=include_dolls)
    conv.build()
    world_assets = "\n".join(conv.assets)
    world_bodies = "\n".join(conv.bodies)

    from multi_drone_mujoco.envs import base_aviary as BA
    original = BA._generate_aviary_xml

    def patched(*args, **kwargs):
        root = ET.fromstring(original(*args, **kwargs))
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("inertiafromgeom", "auto")   # keep drone inertials, add platform's
        asset = root.find("asset")
        for el in ET.fromstring(f"<r>{world_assets}</r>"):
            asset.append(el)
        worldbody = root.find("worldbody")
        for floor in worldbody.findall("geom[@name='floor']"):
            worldbody.remove(floor)                    # arena has its own floor
        for el in ET.fromstring(f"<r>{world_bodies}</r>"):
            worldbody.append(el)
        _inject_props(worldbody.find("body[@name='drone0']"))
        return ET.tostring(root, encoding="unicode")

    return BA, original, patched


def _onboard_window(q: "mp.Queue"):
    """Separate process: show onboard RGB + depth from a frame queue."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.ion()
    fig, (ax_rgb, ax_dep) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.canvas.manager.set_window_title("drone0 onboard camera")
    im_rgb = ax_rgb.imshow(np.zeros((256, 256, 3), np.uint8))
    im_dep = ax_dep.imshow(np.zeros((256, 256)), cmap="turbo", vmin=0, vmax=8)
    ax_rgb.set_title("onboard RGB"); ax_dep.set_title("onboard depth (m)")
    for a in (ax_rgb, ax_dep):
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im_dep, ax=ax_dep, fraction=0.046)
    fig.tight_layout()
    plt.show(block=False)

    while True:
        try:
            item = q.get(timeout=0.05)
        except queue.Empty:
            fig.canvas.flush_events()
            if not plt.fignum_exists(fig.number):
                break
            continue
        if item is None:
            break
        rgb, dep = item
        im_rgb.set_data(rgb)
        im_dep.set_data(dep)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    plt.close(fig)


class Teleop:
    """Hold-to-fly velocity teleop.

    The viewer's key_callback fires on key press AND on OS auto-repeat while a
    key is held. Each event refreshes that axis's velocity command and its
    timestamp; once events stop (key released) the axis times out and its
    command decays exponentially to zero — the drone brakes to rest by itself.

    Axes: 0=fwd/back(W/S) 1=left/right(A/D) 2=up/down(Up/Down) 3=yaw(Left/Right)
    """
    # Commanded speeds while held. The repo model simulates air drag (density/
    # viscosity in its XML) and the velocity loop is D-only, so actual speed
    # settles at ~50-80% of the command — these are calibrated for ~1.5 m/s
    # horizontal / ~1.2 m/s vertical actual.
    SPEED_XY, SPEED_Z, SPEED_YAW = 3.0, 1.5, 1.0
    HOLD_TIMEOUT = 0.7    # s without an event -> treat as released
    DECAY_TAU = 0.3       # s, exponential brake after release

    W, A, S, D, H, R = 87, 65, 83, 68, 72, 82
    UP, DOWN, LEFT, RIGHT = 265, 264, 263, 262
    _KEYMAP = {}          # keycode -> (axis, signed speed); filled below

    def __init__(self):
        self.cmd = np.zeros(4)                     # [vx, vy, vz, yaw_rate]
        self.t_axis = np.full(4, -1e9)             # last refresh per axis
        self.reset_requested = False
        if not Teleop._KEYMAP:
            Teleop._KEYMAP = {
                self.W: (0, +self.SPEED_XY), self.S: (0, -self.SPEED_XY),
                self.A: (1, +self.SPEED_XY), self.D: (1, -self.SPEED_XY),
                self.UP: (2, +self.SPEED_Z), self.DOWN: (2, -self.SPEED_Z),
                self.LEFT: (3, +self.SPEED_YAW), self.RIGHT: (3, -self.SPEED_YAW),
            }

    def on_key(self, k):
        if k == self.H:
            self.cmd[:] = 0; self.t_axis[:] = -1e9
        elif k == self.R:
            self.reset_requested = True
            self.cmd[:] = 0; self.t_axis[:] = -1e9
        elif k in self._KEYMAP:
            axis, speed = self._KEYMAP[k]
            self.cmd[axis] = speed
            self.t_axis[axis] = time.perf_counter()

    def update(self, dt):
        """Called every control step: decay released axes toward zero."""
        now = time.perf_counter()
        decay = math.exp(-dt / self.DECAY_TAU)
        for i in range(4):
            if now - self.t_axis[i] > self.HOLD_TIMEOUT:
                self.cmd[i] *= decay
                if abs(self.cmd[i]) < 0.02:
                    self.cmd[i] = 0.0
        return self.cmd[:3], self.cmd[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    ap.add_argument("--no-dolls", action="store_true")
    ap.add_argument("--spawn", type=float, nargs=3, default=[0.0, -11.0, 1.5])
    args = ap.parse_args()

    print(f"Injecting IMAV world from: {args.sdf}")
    BA, original, patched = make_world_injector(args.sdf, not args.no_dolls)
    BA._generate_aviary_xml = patched
    try:
        env = BA.BaseAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=np.array([args.spawn], dtype=float),
            initial_rpys=np.array([[0.0, 0.0, math.pi / 2]]),   # face +y
            physics=Physics.MJC, sim_freq=240, ctrl_freq=48,
            vision_attributes=True,                              # enables onboard camera
            obs_type=ObservationType.KIN, act_type=ActionType.RPM,
        )
    finally:
        BA._generate_aviary_xml = original

    print(f"World ready: {env.model.ngeom} geoms, {env.model.nbody} bodies")

    # The package defaults the onboard camera to 64x48; bump it to 256x256.
    # _getDroneImages builds its Renderer lazily from IMG_RES, so overriding
    # here (before the first render) is enough — no repo edits needed.
    env.IMG_RES = np.array([256, 256])

    pid = PIDControl(env)
    teleop = Teleop()
    dt = env.CTRL_TIMESTEP
    drone_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "drone0")

    # visual prop joints (kinematic spin, see _inject_props)
    prop_qpos = [env.model.jnt_qposadr[mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, f"drone0_prop{i}_spin")] for i in range(4)]
    prop_dofs = [env.model.jnt_dofadr[mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, f"drone0_prop{i}_spin")] for i in range(4)]
    prop_angle = [0.0] * 4
    env.reset()

    # onboard RGB + depth window in its own process (avoids GLFW/Tk deadlock)
    frame_q: "mp.Queue" = mp.Queue(maxsize=2)
    img_proc = mp.Process(target=_onboard_window, args=(frame_q,), daemon=True)
    img_proc.start()

    print(__doc__[__doc__.index("Velocity"):])

    with mujoco.viewer.launch_passive(env.model, env.data,
                                      key_callback=teleop.on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = drone_bid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 90, -20

        step = 0
        yaw_sp = env.rpy[0, 2]           # heading setpoint, driven by yaw-rate cmd
        while viewer.is_running():
            tic = time.perf_counter()

            if teleop.reset_requested:
                env.reset(); pid.reset(); teleop.reset_requested = False
                yaw_sp = env.rpy[0, 2]

            v_body, yaw_rate = teleop.update(dt)
            pos, quat, yaw = env.pos[0], env.quat[0], env.rpy[0, 2]
            cz, sz = math.cos(yaw), math.sin(yaw)
            v_world = np.array([cz * v_body[0] - sz * v_body[1],
                                sz * v_body[0] + cz * v_body[1],
                                v_body[2]])
            yaw_sp += yaw_rate * dt

            # PURE velocity command: zero position error (target_pos = current
            # position), velocity set-point via target_vel only.
            rpm, _, _ = pid.computeControl(
                control_timestep=dt, cur_pos=pos, cur_quat=quat,
                cur_vel=env.vel[0], cur_ang_vel=env.ang_v[0],
                target_pos=pos,
                target_rpy=np.array([0.0, 0.0, yaw_sp]),
                target_vel=v_world)

            # spin the visual props proportionally to each motor's RPM
            r = rpm.flatten()
            for i in range(4):
                prop_angle[i] += SPIN_DIRS[i] * VIS_SPIN * (r[i] / env.HOVER_RPM) * dt
                env.data.qpos[prop_qpos[i]] = prop_angle[i]
                env.data.qvel[prop_dofs[i]] = 0.0

            env.step(r)                    # repo physics + engine
            viewer.sync()

            # push onboard RGB + depth to the image process ~16 Hz (repo renderer)
            if step % 3 == 0:
                rgb, dep, _ = env._getDroneImages(0)
                try:
                    frame_q.put_nowait((rgb.copy(), np.clip(dep, 0.0, 8.0)))
                except queue.Full:
                    pass
            step += 1

            slack = dt - (time.perf_counter() - tic)
            if slack > 0:
                time.sleep(slack)

    # shut the image process down
    try:
        frame_q.put_nowait(None)
    except queue.Full:
        pass
    img_proc.join(timeout=1.0)
    if img_proc.is_alive():
        img_proc.terminate()
    env.close()


if __name__ == "__main__":
    main()
