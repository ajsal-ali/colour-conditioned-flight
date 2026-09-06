#!/usr/bin/env python3
"""The real IMAV2026 arena, with only the bar stations randomized.

The arena is `assets/imav2026/imav2026_scaled.sdf`, converted to MJCF by
`imav_teleop.SdfToMjcf` and injected into BaseAviary's XML exactly as
`imav_play.make_world_injector` does. Nothing is invented and nothing else is
moved: the entry wall, exit wall, tubes, boxes, turbine, ring board, platforms,
strips and floor are all the competition geometry.

Three things vary between episodes, and only these three:

    * **how many** bar stations are present  (0..3)
    * **which colour** sits at each station   (red / blue, any order)
    * **the bar's height**                    (3 options per colour)

Everything else is fixed. The default stations that ship in the SDF
(`red_bar`, `blue_bar1`, `blue_bar2` and their posts and feet) are skipped
during conversion and re-emitted per layout, at the SDF's own station planes and
with the SDF's own tube radii, post heights and colours.

Scale
-----
The SDF is the competition arena scaled by **2.2x** -- the file name says so and
the numbers confirm it: `red_bar` sits at z = 4.356 = 2.2 x 1980 mm, `blue_bar`
at z = 0.880 = 2.2 x 400 mm, stations 2.20 m apart = 2.2 x 1.0 m. Every constant
in this package is in **SDF units**. The real-world millimetres appear only in
the comments beside the height tables.

That distinction matters: an earlier version of this file mixed the two, taking
the entry-window numbers from the SDF (z ~ 3.4) and the bar heights from the
competition spec (z = 0.4..1.98), which put the drone on a physically wrong 3 m
dive out of every window.

Direction of travel
-------------------
The course runs along **+y**, matching the arena's own `takeoff_platform` at
y = -14.30: the drone enters through the wall at y = -12.10 and leaves through
the one at y = +3.30, meeting red_bar -> blue_bar1 -> blue_bar2 -> tube_B ->
tube_A on the way. The two walls are geometrically identical, so the course is
symmetric and the direction is pure convention -- but it is the competition's
convention, and the bars come *before* the tubes because of it.

`course_gates.TRAVEL_SIGN` is the single source of truth; nothing here or in the
env or the pilot hardcodes a y comparison.
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mavrl.course_gates import (                                  # noqa: E402
    TRAVEL_SIGN, BarGate, BarSide, GateSequence, Opening, WallGate,
)

#: The competition arena. `imav_teleop.DEFAULT_SDF` points at the same file.
SDF_PATH = _REPO / "assets" / "imav2026" / "imav2026_scaled.sdf"

#: SDF units per competition metre. Only used to document the height tables.
SCALE = 2.2

# --------------------------------------------------------------------------
# Fixed arena geometry, read straight out of the SDF
# --------------------------------------------------------------------------

#: The wall the drone enters through. This is the SDF's `exit_wall_*` -- the
#: model names are backwards relative to the flight direction, because the
#: drone starts at the takeoff platform beyond it at y = -14.30.
ENTRY_Y = -12.10
#: ...and leaves through the SDF's `obs_wall_*`. Geometrically identical.
EXIT_Y = 3.30

#: Openings in both walls. Derived from the frame members, e.g. the red opening
#: runs from obs_wall_red_L's inner face (0.440) to obs_wall_red_R's (1.320).
RED_X_LO, RED_X_HI = 0.440, 1.320
RED_Z_LO, RED_Z_HI = 2.970, 3.850
BLUE_X_LO, BLUE_X_HI = -1.320, 0.000
BLUE_Z_LO, BLUE_Z_HI = 2.860, 3.960

#: Bar-station planes, **in the order the drone meets them**: the SDF's red_bar,
#: blue_bar1, blue_bar2. Slot 0 is filled first, so a one-bar layout puts its bar
#: at -9.90, which is the SDF's own first station.
STATION_Y: Tuple[float, ...] = (-9.90, -7.70, -5.50)
STATION_SPACING = 2.20

#: The fixed obstacles between the last bar station and the exit wall. They are
#: never moved; listed here so the reward, the pilot and the tests know they
#: exist.
#:
#: `tube_B` (y = -2.20) is a frame: vertical posts at x = +-1.10, a horizontal at
#: z = 1.014, and a diagonal running from (x=+1.10, z=2.03) to (x=-1.10, z=4.23).
#: `tube_A` (y = 0.00) is a single post at x = 0 spanning the full height.
TUBE_B_Y = -2.20
TUBE_A_Y = 0.00
TUBE_POST_X = 1.10

#: Lateral offset at which to thread `tube_B`, and it is not the window centre.
#: The exit window's red opening is centred at x = 0.88, which sits only 0.22
#: from `tube_B_post_R` -- about 0.12 once the post radius and the airframe are
#: taken off. A straight run from the last bar (x = 0) to the window therefore
#: passes within the pilot's own lateral overshoot of the post, and does collide.
#: 0.35 keeps 0.75 from that post, stays above the diagonal (which is at
#: z = 2.78 for this x) at any sane cruise altitude, and still leaves 0.35 of
#: clearance past `tube_A` on the way out.
TUBE_B_SAFE_X = 0.35

#: (y, x) pairs the pilot threads between gates. Not gates -- nothing scores
#: them -- but a teacher with no obstacle avoidance has to be told about them.
TUBE_VIAS: Tuple[Tuple[float, float], ...] = ((TUBE_B_Y, TUBE_B_SAFE_X),)

# -- bar station geometry (SDF values, unchanged) --------------------------
RED_BAR_RADIUS, RED_BAR_LENGTH = 0.044, 4.40
BLUE_BAR_RADIUS, BLUE_BAR_LENGTH = 0.0396, 3.30
POST_RADIUS = 0.055
POST_X = 1.65
RED_POST_HEIGHT = 4.51
BLUE_POST_HEIGHT = 2.64
FOOT_SIZE = (0.44, 0.264, 0.11)

RED_RGBA = "0.9 0.1 0.1 1"
BLUE_RGBA = "0.1 0.2 0.8 1"
WOOD_RGBA = "0.50 0.35 0.22 1"

#: Bar heights, in SDF units. The comment is the competition value.
RED_BAR_HEIGHTS = (2.640, 3.520, 4.356)      # 1200 / 1600 / 1980 mm
BLUE_BAR_HEIGHTS = (0.880, 1.760, 2.640)     # 400 / 800 / 1200 mm

#: Held out of training so evaluation can ask whether the *rule* was learned
#: rather than four specific heights.
TRAIN_RED_HEIGHTS = (2.640, 4.356)
EVAL_RED_HEIGHTS = (3.520,)
TRAIN_BLUE_HEIGHTS = (0.880, 2.640)
EVAL_BLUE_HEIGHTS = (1.760,)

#: Spawn standoff before the entry wall, i.e. at y = ENTRY_Y - TRAVEL_SIGN*this.
#: One station spacing, which puts the 0.88-wide window at ~40% of the frame
#: height -- and lands on y = -14.30, the arena's own takeoff platform plane.
SPAWN_STANDOFF = 2.20
SPAWN_STANDOFF_JIT = 0.35

#: Curriculum stages: number of bar stations.
STAGE_STATIONS = (0, 1, 2, 3)

#: The SDF models that make up the three shipped bar stations. Skipped during
#: conversion and re-emitted per layout.
SDF_BAR_MODELS = frozenset({
    "red_bar", "red_post_L", "red_post_R", "red_foot_L", "red_foot_R",
    "blue_bar1", "blue_post_L1", "blue_post_R1", "blue_foot_L1", "blue_foot_R1",
    "blue_bar2", "blue_post_L2", "blue_post_R2", "blue_foot_L2", "blue_foot_R2",
})


# --------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------

class SemClass(IntEnum):
    FREE = 0
    WALL = 1          # wall frames, room, posts and feet -- structure
    RED_WINDOW = 2
    BLUE_WINDOW = 3
    RED_BAR = 4
    BLUE_BAR = 5
    FLOOR = 6
    OBSTACLE = 7      # tubes, boxes, turbine, ring board, platforms


N_SEM_CLASSES = len(SemClass)


def classify_body_name(name: Optional[str]) -> SemClass:
    """Body name -> semantic class.

    Classification is by **body**, not geom: `SdfToMjcf` names its geoms `g1`,
    `g2`, ... and puts the meaningful name on the enclosing body. Doing it this
    way covers the converted arena and the bar stations generated here with one
    rule.
    """
    if not name:
        return SemClass.FREE
    n = name.lower()

    if "_bar" in n and n.startswith("station"):
        return SemClass.RED_BAR if "_red_" in n else SemClass.BLUE_BAR
    if n.startswith("station"):                      # posts and feet
        return SemClass.WALL

    if "wall_red" in n or "room_red" in n:
        return SemClass.RED_WINDOW
    if "wall_blue" in n or "room_blue" in n:
        return SemClass.BLUE_WINDOW
    if n.startswith(("obs_wall", "exit_wall", "room_")):
        return SemClass.WALL

    if n.startswith(("arena_floor", "strip_", "room_floor", "ground", "floor")):
        return SemClass.FLOOR
    if n.startswith(("tube_", "box_", "turbine", "ring_board", "platform",
                     "takeoff_platform", "landing_platform", "lp_")):
        return SemClass.OBSTACLE
    return SemClass.WALL


def build_geom_class_lut(model) -> np.ndarray:
    """geom id -> SemClass, via each geom's body name."""
    import mujoco
    lut = np.zeros(model.ngeom, dtype=np.uint8)
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        lut[gid] = int(classify_body_name(name))
    return lut


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StationSpec:
    """One bar station: which slot, which colour, how high."""

    index: int          # 0..2, index into STATION_Y
    colour: str         # "red" | "blue"
    height: float       # bar centreline, SDF units

    @property
    def y(self) -> float:
        return STATION_Y[self.index]

    @property
    def side(self) -> BarSide:
        """Red = pass above, blue = pass below."""
        return BarSide.ABOVE if self.colour == "red" else BarSide.BELOW

    @property
    def post_height(self) -> float:
        return RED_POST_HEIGHT if self.colour == "red" else BLUE_POST_HEIGHT

    @property
    def bar_radius(self) -> float:
        return RED_BAR_RADIUS if self.colour == "red" else BLUE_BAR_RADIUS

    @property
    def bar_length(self) -> float:
        return RED_BAR_LENGTH if self.colour == "red" else BLUE_BAR_LENGTH

    def describe(self) -> str:
        return f"{self.colour}@{self.height:.2f}"


@dataclass(frozen=True)
class CourseLayout:
    """Entry wall -> fixed obstacles -> 0..3 bar stations -> exit wall."""

    stations: Tuple[StationSpec, ...] = ()

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def exit_y(self) -> float:
        return EXIT_Y

    def describe(self) -> str:
        mid = " -> ".join(s.describe() for s in self.stations) or "(no bars)"
        return f"entry -> {mid} -> exit"


def _entry_wall_gate() -> WallGate:
    return WallGate(
        y=ENTRY_Y,
        target=Opening(RED_X_LO, RED_X_HI, RED_Z_LO, RED_Z_HI),
        decoy=Opening(BLUE_X_LO, BLUE_X_HI, BLUE_Z_LO, BLUE_Z_HI))


def _exit_wall_gate() -> WallGate:
    # The exit wall is the entry wall duplicated at y = -12.10, so the target is
    # the red opening there too.
    return WallGate(
        y=EXIT_Y,
        target=Opening(RED_X_LO, RED_X_HI, RED_Z_LO, RED_Z_HI),
        decoy=Opening(BLUE_X_LO, BLUE_X_HI, BLUE_Z_LO, BLUE_Z_HI))


def build_gates(layout: CourseLayout) -> GateSequence:
    gates: List[object] = [_entry_wall_gate()]
    for s in layout.stations:
        gates.append(BarGate(y=s.y, height=s.height, side=s.side))
    gates.append(_exit_wall_gate())
    return GateSequence(gates)


def heights_for(colour: str, split: str) -> Tuple[float, ...]:
    if colour == "red":
        return {"train": TRAIN_RED_HEIGHTS, "eval": EVAL_RED_HEIGHTS,
                "all": RED_BAR_HEIGHTS}[split]
    return {"train": TRAIN_BLUE_HEIGHTS, "eval": EVAL_BLUE_HEIGHTS,
            "all": BLUE_BAR_HEIGHTS}[split]


def sample_layout(rng: np.random.Generator, n_stations: int = 3,
                  split: str = "train") -> CourseLayout:
    """Vary count, colour order and height. Nothing else moves."""
    n = int(np.clip(n_stations, 0, len(STATION_Y)))
    stations = []
    for i in range(n):
        colour = "red" if rng.random() < 0.5 else "blue"
        height = float(rng.choice(heights_for(colour, split)))
        stations.append(StationSpec(index=i, colour=colour, height=height))
    return CourseLayout(tuple(stations))


def sample_layout_for_stage(rng: np.random.Generator, stage: int,
                            split: str = "train") -> CourseLayout:
    """Curriculum stage -> a layout. Stage k means k bar stations."""
    stage = int(np.clip(stage, 0, len(STAGE_STATIONS) - 1))
    return sample_layout(rng, STAGE_STATIONS[stage], split)


def resample_heights(rng: np.random.Generator, layout: CourseLayout,
                     split: str = "train") -> CourseLayout:
    """Same stations, same colours, new heights.

    The narrow within-stage variation: only the bars move. A course whose colour
    order also changed varies two things at once, so a policy that improves
    cannot be attributed to either -- and the vertical profile is the part the
    memory has to hold, so it is the part worth resampling on its own.
    """
    return CourseLayout(tuple(
        replace(s, height=float(rng.choice(heights_for(s.colour, split))))
        for s in layout.stations))


def all_layouts(n_stations: int, split: str = "train") -> List[CourseLayout]:
    """Every colour/height combination -- for exhaustive evaluation."""
    import itertools
    out = []
    per_slot = [[(c, h) for c in ("red", "blue") for h in heights_for(c, split)]
                for _ in range(n_stations)]
    for combo in itertools.product(*per_slot):
        out.append(CourseLayout(tuple(
            StationSpec(i, c, h) for i, (c, h) in enumerate(combo))))
    return out


def spawn_pose(rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Start in front of the entry wall, facing down the course (-y).

    Centred on the RED opening, not on the corridor, so the very first thing the
    camera sees is the target rather than the decoy.
    """
    standoff = SPAWN_STANDOFF + float(rng.uniform(-SPAWN_STANDOFF_JIT,
                                                  SPAWN_STANDOFF_JIT))
    x = 0.5 * (RED_X_LO + RED_X_HI) + float(rng.uniform(-0.15, 0.15))
    z = 0.5 * (RED_Z_LO + RED_Z_HI) + float(rng.uniform(-0.15, 0.15))
    return (np.array([x, ENTRY_Y - TRAVEL_SIGN * standoff, z]),
            YAW_DOWN_COURSE)


#: Heading that looks along the direction of travel.
YAW_DOWN_COURSE = math.pi / 2 if TRAVEL_SIGN > 0 else -math.pi / 2


# --------------------------------------------------------------------------
# XML
# --------------------------------------------------------------------------

def _cyl(name: str, pos, size, rgba: str, quat: Optional[str] = None) -> str:
    q = f' quat="{quat}"' if quat else ""
    return (f'<body name="{name}" pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}"{q}>'
            f'<geom name="{name}_g" type="cylinder" '
            f'size="{size[0]:.6g} {size[1]:.6g}" rgba="{rgba}"/></body>')


def _box(name: str, pos, half, rgba: str) -> str:
    return (f'<body name="{name}" pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}">'
            f'<geom name="{name}_g" type="box" '
            f'size="{half[0]:.6g} {half[1]:.6g} {half[2]:.6g}" '
            f'rgba="{rgba}"/></body>')


#: 90-degree rotation about y: turns a cylinder's axis from +z to +x, which is
#: how the SDF lays its horizontal bars (`<pose> ... 0 1.5708 0`).
_BAR_QUAT = "0.7071068 0 0.7071068 0"


def station_xml(spec: StationSpec) -> str:
    """Posts, feet and bar for one station, in the SDF's own style.

    Post height is the SDF's for that colour and does not follow the bar: the
    shipped blue station already has a 2.64 post carrying a 0.88 bar.
    """
    tag = f"station{spec.index}_{spec.colour}"
    rgba = RED_RGBA if spec.colour == "red" else BLUE_RGBA
    ph = spec.post_height
    fx, fy, fz = FOOT_SIZE
    parts = []
    for side, sx in (("L", -POST_X), ("R", POST_X)):
        parts.append(_cyl(f"{tag}_post_{side}", (sx, spec.y, ph / 2),
                          (POST_RADIUS, ph / 2), rgba))
        parts.append(_box(f"{tag}_foot_{side}", (sx, spec.y, fz / 2),
                          (fx / 2, fy / 2, fz / 2), WOOD_RGBA))
    parts.append(_cyl(f"{tag}_bar", (0.0, spec.y, spec.height),
                      (spec.bar_radius, spec.bar_length / 2), rgba,
                      quat=_BAR_QUAT))
    return "\n".join(parts)


def _set_visual_attrs(xml: str, tag: str, attrs: str) -> str:
    """Force `attrs` onto the first `<tag>` under `<visual>`, creating it if
    absent. A second element of the same name is invalid MJCF, so an existing
    one is edited in place rather than appended to."""
    names = re.findall(r"(\w+)=", attrs)
    pattern = rf"<{tag}\b[^>]*/?>"
    if re.search(pattern, xml):
        def _fix(m):
            t = m.group(0)
            for n in names:
                t = re.sub(rf'\s{n}="[^"]*"', "", t)
            return (t[:-2].rstrip() + f" {attrs}/>") if t.endswith("/>") \
                else (t[:-1].rstrip() + f" {attrs}>")
        return re.sub(pattern, _fix, xml, count=1)
    return xml.replace("<visual>", f"<visual><{tag} {attrs}/>", 1)


def patch_offscreen_framebuffer(xml: str, res: int, offsamples: int = 0) -> str:
    """Size the offscreen framebuffer to `res` and set its multisampling.

    MuJoCo defaults to 640x480 and rejects a larger render outright, hence
    `offwidth`/`offheight`.

    `offsamples=0` disables multisampling, and is not merely a performance
    choice. MuJoCo asks for a 4x multisampled offscreen buffer by default, and
    NVIDIA's surfaceless EGL contexts reject that combination with
    GL_FRAMEBUFFER_UNSUPPORTED (0x8CDD) -- headless GPU rendering fails outright
    while the same model renders fine under GLX or osmesa. Antialiasing would be
    thrown away here in any case: depth is **min-pooled** 512 -> 128 and
    segmentation is nearest-sampled, so smoothed edges would either be discarded
    or, for seg, blend two class ids into a third that means nothing.
    """
    xml = _set_visual_attrs(xml, "global",
                            f'offwidth="{res}" offheight="{res}"')
    return _set_visual_attrs(xml, "quality", f'offsamples="{offsamples}"')


class _CourseSdf:
    """`SdfToMjcf` with the shipped bar stations left out.

    Lazily imported and cached: the conversion walks 100+ models and rasterizes
    textures, and the fixed part of the arena is identical for every layout, so
    it is done once per process rather than once per `set_layout`.
    """

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is not None:
            return cls._cache
        from imav_teleop import SdfToMjcf

        class _Skipping(SdfToMjcf):
            def _convert_static_model(self, model):
                if model.get("name") in SDF_BAR_MODELS:
                    return
                super()._convert_static_model(model)

        conv = _Skipping(SDF_PATH, include_dolls=False)
        conv.build()
        cls._cache = ("\n".join(conv.assets), "\n".join(conv.bodies))
        return cls._cache


def make_course_world_injector(layout: CourseLayout, render_res: int = 512):
    """`base_aviary._generate_aviary_xml` replacement: arena + this layout.

    Same mechanism as `imav_play.make_world_injector` -- the package's own drone
    XML with the arena appended, no repo files edited.
    """
    import xml.etree.ElementTree as ET
    from multi_drone_mujoco.envs import base_aviary as BA

    if not SDF_PATH.exists():
        raise FileNotFoundError(
            f"the IMAV arena is missing: {SDF_PATH}\n"
            f"It ships with the repo under assets/imav2026/ -- a partial\n"
            f"clone or a moved checkout is the usual cause.")

    world_assets, world_bodies = _CourseSdf.get()
    bars = "\n".join(station_xml(s) for s in layout.stations)
    original = BA._generate_aviary_xml

    def patched(*args, **kwargs):
        root = ET.fromstring(original(*args, **kwargs))
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("inertiafromgeom", "auto")
        asset = root.find("asset")
        for el in ET.fromstring(f"<r>{world_assets}</r>"):
            asset.append(el)
        worldbody = root.find("worldbody")
        for floor in worldbody.findall("geom[@name='floor']"):
            worldbody.remove(floor)          # the arena brings its own
        for el in ET.fromstring(f"<r>{world_bodies}</r>"):
            worldbody.append(el)
        if bars:
            for el in ET.fromstring(f"<r>{bars}</r>"):
                worldbody.append(el)
        return patch_offscreen_framebuffer(
            ET.tostring(root, encoding="unicode"), render_res)

    return original, patched


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in STAGE_STATIONS:
        lay = sample_layout(rng, n, "all")
        gates = build_gates(lay)
        print(f"{n} bars: {lay.describe()}")
        for g in gates.gates:
            kind = "wall" if isinstance(g, WallGate) else f"bar {g.side.name}"
            print(f"    y={g.y:+7.2f}  {kind:<12} target_z="
                  f"{getattr(g, 'target_z', g.center[2]):.3f}")
