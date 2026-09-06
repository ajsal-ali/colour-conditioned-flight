#!/usr/bin/env python3
"""IMAV2026 Gazebo world in MuJoCo, with a keyboard-teleoperated drone.

Converts the PX4/Gazebo SDF world in `assets/imav2026/imav2026_scaled.sdf` to MJCF
at runtime (boxes, cylinders, textured meshes, the springy landing platform)
and drops a simple quadrotor with a velocity-tracking flight controller in it.

Usage (run inside the rl_mujoco conda env):
    conda run -n rl_mujoco python imav_teleop.py
    # or: conda activate rl_mujoco && python imav_teleop.py

Controls:
    W / S        forward / backward
    A / D        strafe left / right
    Up / Down    climb / descend
    Left / Right yaw left / right
    Shift        speed boost
    Space        brake (hold position velocity = 0)
    R            reset drone to start
    C            toggle follow-cam / free-cam
    Mouse drag   orbit camera, Scroll: zoom
    ESC          quit

Options:
    --sdf PATH    world SDF (default: assets/imav2026/imav2026_scaled.sdf next to this file)
    --dump PATH   also write the generated MJCF XML for inspection
    --no-dolls    skip the (heavy) doll meshes
"""

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import mujoco
import glfw

HERE = Path(__file__).resolve().parent
DEFAULT_SDF = HERE / "assets" / "imav2026" / "imav2026_scaled.sdf"


# ---------------------------------------------------------------------------
# Small quaternion toolbox (wxyz order, matching MuJoCo)
# ---------------------------------------------------------------------------

def quat_from_rpy(r: float, p: float, y: float) -> np.ndarray:
    """SDF fixed-axis roll-pitch-yaw -> quaternion (R = Rz(y) Ry(p) Rx(r))."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    qx = np.array([cr, sr, 0, 0])
    qy = np.array([cp, 0, sp, 0])
    qz = np.array([cy, 0, 0, sy])
    return quat_mul(quat_mul(qz, qy), qx)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def compose(p1, q1, p2, q2):
    """Frame composition: (p1,q1) o (p2,q2)."""
    return p1 + quat_rotate(q1, p2), quat_mul(q1, q2)


# ---------------------------------------------------------------------------
# SDF -> MJCF conversion
# ---------------------------------------------------------------------------

def parse_pose(el) -> tuple:
    """Read a <pose> child of `el` (SDF 'x y z r p y'); identity if absent."""
    pose_el = el.find("pose")
    if pose_el is None or pose_el.text is None:
        return np.zeros(3), np.array([1.0, 0, 0, 0])
    vals = [float(v) for v in pose_el.text.split()]
    pos = np.array(vals[:3])
    quat = quat_from_rpy(*vals[3:6]) if len(vals) >= 6 else np.array([1.0, 0, 0, 0])
    return pos, quat


def fmt(arr) -> str:
    return " ".join(f"{v:.6g}" for v in np.asarray(arr).flatten())


class SdfToMjcf:
    """Converts the imav2026 SDF world into an MJCF XML string."""

    def __init__(self, sdf_path: Path, include_dolls: bool = True):
        self.sdf_path = Path(sdf_path)
        self.root_dir = self.sdf_path.parent  # model:// URIs resolve here
        self.include_dolls = include_dolls
        self.assets = []           # asset XML lines
        self.bodies = []           # worldbody XML lines
        self._tex_mats = {}        # texture file -> material name
        self._meshes = {}          # (obj path, scale) -> mesh name
        self._geom_id = 0

    # -- asset helpers ------------------------------------------------------

    def _resolve_uri(self, uri: str) -> Path:
        return self.root_dir / uri.replace("model://", "")

    def _as_png(self, tex_file: Path, square: bool = False) -> Path:
        """MuJoCo only loads PNG textures (and cube textures must be square);
        convert/resize once into a cache dir."""
        if tex_file.suffix.lower() == ".png" and not square:
            return tex_file
        cache = self.root_dir / ".mj_png_cache"
        cache.mkdir(exist_ok=True)
        out = cache / (tex_file.stem + ("_sq" if square else "") + ".png")
        if not out.exists():
            from PIL import Image
            img = Image.open(tex_file).convert("RGB")
            if square and img.width != img.height:
                side = min(1024, max(img.size))
                img = img.resize((side, side))
            out.parent.mkdir(exist_ok=True)
            img.save(out)
        return out

    def _material_for_texture(self, tex_file: Path, kind: str) -> str:
        """kind: 'cube' for boxes, '2d' for UV-mapped meshes."""
        tex_file = self._as_png(tex_file, square=(kind == "cube"))
        key = (str(tex_file), kind)
        if key in self._tex_mats:
            return self._tex_mats[key]
        name = f"mat_{tex_file.stem}_{kind}"
        self.assets.append(
            f'<texture name="tex_{name}" type="{kind}" file="{tex_file}"/>')
        self.assets.append(
            f'<material name="{name}" texture="tex_{name}" texuniform="false"/>')
        self._tex_mats[key] = name
        return name

    def _mesh_asset(self, obj_file: Path, scale: np.ndarray) -> str:
        key = (str(obj_file), tuple(scale))
        if key in self._meshes:
            return self._meshes[key]
        name = f"mesh_{obj_file.stem}_{len(self._meshes)}"
        self.assets.append(
            f'<mesh name="{name}" file="{obj_file}" scale="{fmt(scale)}"/>')
        self._meshes[key] = name
        return name

    # -- geometry -----------------------------------------------------------

    def _visual_to_geom(self, visual, link_pos, link_quat, collide=True,
                        fallback_box_size=None) -> str:
        """One SDF <visual> -> one MJCF <geom> line ('' if unsupported)."""
        vpos, vquat = parse_pose(visual)
        pos, quat = compose(link_pos, link_quat, vpos, vquat)
        geo = visual.find("geometry")
        if geo is None:
            return ""
        self._geom_id += 1
        gid = f"g{self._geom_id}"
        contact = "" if collide else ' contype="0" conaffinity="0"'
        common = f'name="{gid}" pos="{fmt(pos)}" quat="{fmt(quat)}"{contact}'

        # material: texture beats flat color
        rgba, tex_uri = self._read_material(visual)
        tex_file = self._resolve_uri(tex_uri) if tex_uri else None

        box = geo.find("box")
        if box is not None:
            half = np.array([float(v) for v in box.find("size").text.split()]) / 2
            if tex_file is not None:
                mat = self._material_for_texture(tex_file, "cube")
                return f'<geom type="box" size="{fmt(half)}" material="{mat}" {common}/>'
            return f'<geom type="box" size="{fmt(half)}" rgba="{fmt(rgba)}" {common}/>'

        cyl = geo.find("cylinder")
        if cyl is not None:
            r = float(cyl.find("radius").text)
            h = float(cyl.find("length").text) / 2
            return f'<geom type="cylinder" size="{r:.6g} {h:.6g}" rgba="{fmt(rgba)}" {common}/>'

        mesh = geo.find("mesh")
        if mesh is not None:
            # Flat decal meshes (floor strips) can't be convex-hulled by MuJoCo;
            # render them as the link's thin collision box with the texture on it.
            if fallback_box_size is not None:
                half = fallback_box_size / 2
                if tex_file is not None:
                    mat = self._material_for_texture(tex_file, "cube")
                    return f'<geom type="box" size="{fmt(half)}" material="{mat}" {common}/>'
                return f'<geom type="box" size="{fmt(half)}" rgba="{fmt(rgba)}" {common}/>'
            obj = self._resolve_uri(mesh.find("uri").text.strip())
            scale_el = mesh.find("scale")
            scale = (np.array([float(v) for v in scale_el.text.split()])
                     if scale_el is not None else np.ones(3))
            mesh_name = self._mesh_asset(obj, scale)
            if tex_file is not None:
                mat = self._material_for_texture(tex_file, "2d")
                return f'<geom type="mesh" mesh="{mesh_name}" material="{mat}" {common}/>'
            return f'<geom type="mesh" mesh="{mesh_name}" rgba="{fmt(rgba)}" {common}/>'

        return ""

    @staticmethod
    def _read_material(visual):
        rgba = np.array([0.7, 0.7, 0.7, 1.0])
        tex_file = None
        mat = visual.find("material")
        if mat is not None:
            for tag in ("diffuse", "ambient"):
                el = mat.find(tag)
                if el is not None and el.text:
                    vals = [float(v) for v in el.text.split()]
                    rgba = np.array((vals + [1.0])[:4])
                    break
            albedo = mat.find(".//albedo_map")
            if albedo is not None and albedo.text:
                tex_file = albedo.text.strip()
        if tex_file is not None:
            return rgba, tex_file
        return rgba, None

    # -- models -------------------------------------------------------------

    def _convert_static_model(self, model):
        name = model.get("name")
        mpos, mquat = parse_pose(model)
        geoms = []
        for link in model.findall("link"):
            lpos, lquat = parse_pose(link)
            # thin collision box of this link, if any (fallback for flat meshes)
            col_box = link.find("collision/geometry/box/size")
            box_size = (np.array([float(v) for v in col_box.text.split()])
                        if col_box is not None else None)
            for visual in link.findall("visual"):
                g = self._visual_to_geom(visual, lpos, lquat,
                                         fallback_box_size=box_size)
                if g:
                    geoms.append("  " + g)
        if not geoms:
            return
        self.bodies.append(
            f'<body name="{name}" pos="{fmt(mpos)}" quat="{fmt(mquat)}">')
        self.bodies.extend(geoms)
        self.bodies.append("</body>")

    def _convert_landing_platform(self):
        """Special case: the only jointed model (springy sliding/tilting pad)."""
        aruco1 = self._material_for_texture(
            self.root_dir / "materials/textures/aruco_marker_1.jpg", "cube")
        yaw_neg90 = quat_from_rpy(0, 0, -math.pi / 2)
        self.bodies.append(f"""
<body name="landing_platform" pos="5.5 -14.3 0.22">
  <joint name="lp_stroke" type="slide" axis="0 1 0" range="-0.22 2.42"
         stiffness="12.5" springref="0.5" damping="1.0"/>
  <geom name="lp_carriage" type="box" size="0.06 0.06 0.06" rgba="0.25 0.25 0.25 1"
        mass="4" contype="0" conaffinity="0"/>
  <body name="lp_base">
    <joint name="lp_pitch" type="hinge" axis="1 0 0" range="-1.151 0"
           stiffness="0.5" springref="-0.2618" damping="0.15"/>
    <geom name="lp_deck" type="box" size="0.55 0.55 0.055" rgba="0.12 0.60 0.12 1" mass="1"/>
    <geom name="lp_aruco" type="box" size="0.44 0.44 0.0011" pos="0 0 0.0572"
          quat="{fmt(yaw_neg90)}" material="{aruco1}" mass="0.001"/>
  </body>
</body>""")

    def _convert_dolls(self, world):
        obj = self.root_dir / "models/MODERN_DOLL_FAMILY/meshes/model.obj"
        tex = self.root_dir / "models/MODERN_DOLL_FAMILY/materials/textures/texture.png"
        mat = self._material_for_texture(tex, "2d")
        mesh = self._mesh_asset(obj, np.ones(3))
        for inc in world.findall("include"):
            uri = inc.find("uri")
            if uri is None or "MODERN_DOLL_FAMILY" not in (uri.text or ""):
                continue
            name = inc.find("name").text
            pos, quat = parse_pose(inc)
            self.bodies.append(
                f'<body name="{name}" pos="{fmt(pos)}" quat="{fmt(quat)}">\n'
                f'  <geom type="mesh" mesh="{mesh}" material="{mat}" '
                f'contype="0" conaffinity="0"/>\n'
                f'</body>')

    # -- top level ----------------------------------------------------------

    def build(self) -> str:
        world = ET.parse(self.sdf_path).getroot().find("world")

        for model in world.findall("model"):
            if model.get("name") == "landing_platform":
                self._convert_landing_platform()
            else:
                self._convert_static_model(model)
        if self.include_dolls:
            self._convert_dolls(world)

        assets = "\n    ".join(self.assets)
        bodies = "\n    ".join("\n".join(self.bodies).split("\n"))
        return f"""
<mujoco model="imav2026_teleop">
  <option timestep="0.004" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <map znear="0.01" zfar="300"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.45 0.65 0.95" rgb2="0.9 0.95 1.0" width="512" height="512"/>
    <texture name="tex_ground" type="2d" builtin="checker"
             rgb1="0.30 0.32 0.34" rgb2="0.36 0.38 0.40" width="512" height="512"/>
    <material name="mat_ground" texture="tex_ground" texrepeat="30 30" reflectance="0.05"/>
    {assets}
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 22" dir="0.1 0.1 -1"
           diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <geom name="ground" type="plane" pos="0 0 -0.045" size="60 60 0.1" material="mat_ground"/>
    {bodies}

    <!-- teleop quadrotor -->
    <body name="drone" pos="0 -13.2 1.5">
      <freejoint name="drone_free"/>
      <geom name="d_core" type="box" size="0.09 0.09 0.03" rgba="0.13 0.13 0.16 1" mass="0.8"/>
      <geom name="d_nose" type="box" size="0.05 0.025 0.014" pos="0.10 0 0.035"
            rgba="1 0.3 0.05 1" mass="0.02"/>
      <geom name="d_arm_fr" type="capsule" fromto="0 0 0.01  0.22 -0.22 0.02" size="0.016" rgba="0.55 0.55 0.6 1" mass="0.04"/>
      <geom name="d_arm_fl" type="capsule" fromto="0 0 0.01  0.22  0.22 0.02" size="0.016" rgba="0.55 0.55 0.6 1" mass="0.04"/>
      <geom name="d_arm_br" type="capsule" fromto="0 0 0.01 -0.22 -0.22 0.02" size="0.016" rgba="0.35 0.35 0.4 1" mass="0.04"/>
      <geom name="d_arm_bl" type="capsule" fromto="0 0 0.01 -0.22  0.22 0.02" size="0.016" rgba="0.35 0.35 0.4 1" mass="0.04"/>
      <geom name="d_rot_fr" type="cylinder" pos="0.22 -0.22 0.045" size="0.13 0.006" rgba="1 0.35 0.1 0.45" mass="0.03"/>
      <geom name="d_rot_fl" type="cylinder" pos="0.22  0.22 0.045" size="0.13 0.006" rgba="1 0.35 0.1 0.45" mass="0.03"/>
      <geom name="d_rot_br" type="cylinder" pos="-0.22 -0.22 0.045" size="0.13 0.006" rgba="0.2 0.2 0.25 0.45" mass="0.03"/>
      <geom name="d_rot_bl" type="cylinder" pos="-0.22  0.22 0.045" size="0.13 0.006" rgba="0.2 0.2 0.25 0.45" mass="0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Flight controller: world-frame velocity tracking + attitude stabilization
# ---------------------------------------------------------------------------

class VelocityController:
    KV = 3.0          # velocity error gain -> acceleration
    KR = 8.0          # attitude error gain -> torque (per unit inertia-ish)
    KW = 1.2          # angular damping
    KYAW = 2.0        # yaw-rate tracking gain
    MAX_TILT_ACC = 6.0

    def __init__(self, model, data, body_name="drone"):
        self.model = model
        self.data = data
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.mass = model.body_subtreemass[self.bid]
        self.g = -model.opt.gravity[2]

    def step(self, v_cmd_body: np.ndarray, yaw_rate_cmd: float):
        d = self.data
        quat = d.xquat[self.bid].copy()
        v = d.cvel[self.bid][3:6].copy()      # linear vel, world frame
        w = d.cvel[self.bid][0:3].copy()      # angular vel, world frame

        # rotate command from heading (yaw-only) frame into the world
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        cz, sz = math.cos(yaw), math.sin(yaw)
        v_des = np.array([
            cz * v_cmd_body[0] - sz * v_cmd_body[1],
            sz * v_cmd_body[0] + cz * v_cmd_body[1],
            v_cmd_body[2],
        ])

        # thrust: gravity feedforward + velocity error
        acc = self.KV * (v_des - v)
        acc[:2] = np.clip(acc[:2], -self.MAX_TILT_ACC, self.MAX_TILT_ACC)
        acc[2] = np.clip(acc[2], -0.8 * self.g, 2.0 * self.g)
        force = self.mass * (acc + np.array([0.0, 0.0, self.g]))

        # attitude: align body z with the thrust direction (banking look),
        # plus yaw-rate tracking and damping
        z_body = R[:, 2]
        z_des = force / max(np.linalg.norm(force), 1e-6)
        inertia_scale = self.mass * 0.04    # rough moment-of-inertia scale
        torque = inertia_scale * (
            self.KR * np.cross(z_body, z_des) - self.KW * w)
        torque[2] += inertia_scale * self.KYAW * (yaw_rate_cmd - w[2])

        d.xfrc_applied[self.bid, :3] = force
        d.xfrc_applied[self.bid, 3:] = torque


# ---------------------------------------------------------------------------
# GLFW viewer with polled keys (real hold-to-fly teleop)
# ---------------------------------------------------------------------------

class TeleopViewer:
    SPEED_XY = 3.0
    SPEED_Z = 2.0
    SPEED_YAW = 1.6
    BOOST = 2.5

    def __init__(self, model, data, controller):
        self.model, self.data, self.ctrl = model, data, controller
        self.qpos0 = data.qpos.copy()

        if not glfw.init():
            raise RuntimeError("GLFW failed to init (do you have a display?)")
        self.window = glfw.create_window(1440, 900, "IMAV2026 — drone teleop", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = controller.bid
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 7.0, 90, -25
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=5000)
        self.pert = mujoco.MjvPerturb()
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._last_mouse = (0.0, 0.0)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)

    # -- events -------------------------------------------------------------

    def _on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_R:
            self.data.qpos[:] = self.qpos0
            self.data.qvel[:] = 0
            self.data.xfrc_applied[:] = 0
            mujoco.mj_forward(self.model, self.data)
        elif key == glfw.KEY_C:
            if self.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.cam.trackbodyid = self.ctrl.bid

    def _on_scroll(self, window, dx, dy):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0, -0.05 * dy, self.scene, self.cam)

    def _on_mouse_move(self, window, x, y):
        dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
        self._last_mouse = (x, y)
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS:
            return
        w, h = glfw.get_window_size(self.window)
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                              dx / h, dy / h, self.scene, self.cam)

    # -- teleop command from held keys --------------------------------------

    def _read_command(self):
        k = lambda key: glfw.get_key(self.window, key) == glfw.PRESS
        boost = self.BOOST if (k(glfw.KEY_LEFT_SHIFT) or k(glfw.KEY_RIGHT_SHIFT)) else 1.0
        v = np.zeros(3)
        yaw_rate = 0.0
        if k(glfw.KEY_W):
            v[0] += self.SPEED_XY
        if k(glfw.KEY_S):
            v[0] -= self.SPEED_XY
        if k(glfw.KEY_A):
            v[1] += self.SPEED_XY      # +y is left in body/heading frame
        if k(glfw.KEY_D):
            v[1] -= self.SPEED_XY
        if k(glfw.KEY_UP):
            v[2] += self.SPEED_Z
        if k(glfw.KEY_DOWN):
            v[2] -= self.SPEED_Z
        if k(glfw.KEY_LEFT):
            yaw_rate += self.SPEED_YAW
        if k(glfw.KEY_RIGHT):
            yaw_rate -= self.SPEED_YAW
        if k(glfw.KEY_SPACE):
            v[:] = 0
            yaw_rate = 0.0
        return v * boost, yaw_rate

    # -- main loop ----------------------------------------------------------

    HELP = ("W/S fwd/back   A/D strafe   Up/Down alt   L/R arrows yaw\n"
            "Shift boost   Space brake   R reset   C cam   ESC quit")

    def run(self):
        model, data = self.model, self.data
        dt = model.opt.timestep
        last = time.perf_counter()
        acc = 0.0
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            v_cmd, yaw_cmd = self._read_command()

            now = time.perf_counter()
            acc = min(acc + (now - last), 0.1)   # avoid spiral of death
            last = now
            while acc >= dt:
                data.xfrc_applied[self.ctrl.bid, :] = 0
                self.ctrl.step(v_cmd, yaw_cmd)
                mujoco.mj_step(model, data)
                acc -= dt

            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(model, data, self.opt, self.pert, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.ctx)

            pos = data.xpos[self.ctrl.bid]
            vel = data.cvel[self.ctrl.bid][3:6]
            status = (f"pos ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f}) m\n"
                      f"speed {np.linalg.norm(vel):5.2f} m/s")
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, self.HELP, status, self.ctx)
            glfw.swap_buffers(self.window)

        glfw.terminate()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    ap.add_argument("--dump", type=Path, default=None,
                    help="write generated MJCF XML to this path")
    ap.add_argument("--no-dolls", action="store_true",
                    help="skip the doll-family meshes")
    args = ap.parse_args()

    print(f"Converting SDF world: {args.sdf}")
    xml = SdfToMjcf(args.sdf, include_dolls=not args.no_dolls).build()
    if args.dump:
        args.dump.write_text(xml)
        print(f"MJCF dumped to: {args.dump}")

    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:
        if not args.no_dolls:
            print(f"[WARN] compile failed ({e}); retrying without doll meshes")
            xml = SdfToMjcf(args.sdf, include_dolls=False).build()
            model = mujoco.MjModel.from_xml_string(xml)
        else:
            raise
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"World ready: {model.ngeom} geoms, {model.nbody} bodies")

    ctrl = VelocityController(model, data)
    TeleopViewer(model, data, ctrl).run()


if __name__ == "__main__":
    main()
