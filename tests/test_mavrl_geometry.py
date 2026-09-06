#!/usr/bin/env python3
"""Geometry, gate and image-pipeline tests.

Runs anywhere -- mavrl.course_gates, mavrl.course_world and mavrl.imageproc
import no mujoco and no torch, which is the whole reason they are separate
modules. Run with `pytest tests/` or `python tests/test_mavrl_geometry.py`.

All distances are **SDF units**: the arena is the IMAV2026 course scaled 2.2x.
Divide by 2.2 for competition metres.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mavrl.course_gates import (
    TRAVEL_SIGN, BarGate, BarSide, MIN_FLIGHT_Z, Opening, Outcome, WallGate,
)
from mavrl.course_world import (
    BLUE_BAR_HEIGHTS, BLUE_X_HI, BLUE_X_LO, BLUE_Z_HI, BLUE_Z_LO, ENTRY_Y,
    EVAL_BLUE_HEIGHTS, EVAL_RED_HEIGHTS, EXIT_Y, POST_X, RED_BAR_HEIGHTS,
    RED_BAR_LENGTH, RED_X_HI, RED_X_LO, RED_Z_HI, RED_Z_LO, SCALE,
    SPAWN_STANDOFF, SPAWN_STANDOFF_JIT, STATION_SPACING, STATION_Y,
    TRAIN_BLUE_HEIGHTS, TRAIN_RED_HEIGHTS, YAW_DOWN_COURSE,
    CourseLayout, SemClass, StationSpec, all_layouts, build_gates,
    classify_body_name, sample_layout, spawn_pose, station_xml,
)
from mavrl.imageproc import (
    downsample_depth, downsample_rgb, downsample_seg, proximity_weight,
    seg_to_classes,
)
from mavrl.sensor_noise import NoiseConfig, corrupt

#: A step in the direction of travel, taken from the one source of truth so a
#: direction flip cannot leave these tests asserting the old convention.
FWD = TRAVEL_SIGN


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def test_wall_gate_target_decoy_and_blocked():
    g = WallGate(y=ENTRY_Y,
                 target=Opening(RED_X_LO, RED_X_HI, RED_Z_LO, RED_Z_HI),
                 decoy=Opening(BLUE_X_LO, BLUE_X_HI, BLUE_Z_LO, BLUE_Z_HI))
    before, after = ENTRY_Y - 0.1 * FWD, ENTRY_Y + 0.1 * FWD
    assert g.check([0.88, before, 3.41], [0.88, after, 3.41]) is Outcome.PASS
    assert g.check([-0.66, before, 3.41], [-0.66, after, 3.41]) is Outcome.WRONG_GATE
    assert g.check([0.0, before, 1.0], [0.0, after, 1.0]) is Outcome.BLOCKED
    assert g.check([0.88, before, 3.41],
                   [0.88, before + 0.02 * FWD, 3.41]) is Outcome.NONE


def test_backward_crossing_does_not_trigger():
    """Retreating through a gate must not re-score it."""
    g = WallGate(y=ENTRY_Y, target=Opening(RED_X_LO, RED_X_HI, RED_Z_LO, RED_Z_HI))
    assert g.check([0.88, ENTRY_Y + 0.1 * FWD, 3.41],
                   [0.88, ENTRY_Y - 0.1 * FWD, 3.41]) is Outcome.NONE


def test_bar_gate_sides():
    """Red = pass above, blue = pass below."""
    y = STATION_Y[0]
    lo, hi = y - 0.1 * FWD, y + 0.1 * FWD          # before, after the plane
    red = BarGate(y=y, height=3.520, side=BarSide.ABOVE)
    assert red.check([0, lo, 4.2], [0, hi, 4.2]) is Outcome.PASS
    assert red.check([0, lo, 2.0], [0, hi, 2.0]) is Outcome.WRONG_SIDE

    blue = BarGate(y=y, height=1.760, side=BarSide.BELOW)
    assert blue.check([0, lo, 1.0], [0, hi, 1.0]) is Outcome.PASS
    assert blue.check([0, lo, 2.6], [0, hi, 2.6]) is Outcome.WRONG_SIDE


def test_bar_gate_uses_interpolated_crossing_height():
    """The z at the plane decides, not the z at either endpoint."""
    y = STATION_Y[0]
    bar = BarGate(y=y, height=2.640, side=BarSide.BELOW)
    # starts below, ends above, but crosses the plane above the bar
    assert bar.check([0, y - 1.0 * FWD, 2.0],
                     [0, y + 1.0 * FWD, 4.0]) is Outcome.WRONG_SIDE
    # same endpoints in z but crossing while still low
    assert bar.check([0, y - 0.1 * FWD, 2.0],
                     [0, y + 1.9 * FWD, 4.0]) is Outcome.PASS


def test_waypoints_lie_in_the_gate_plane():
    """Regression: a waypoint past the plane makes pursuit cross at the wrong
    altitude. Found by flying a straight line at a blue bar and crossing above
    it."""
    layout = CourseLayout((StationSpec(0, "blue", 2.640),))
    gs = build_gates(layout)
    for gate in gs.gates:
        assert gate.waypoint()[1] == gate.y
    # ...and the sequence must not quietly re-introduce a standoff of its own.
    assert np.array_equal(gs.waypoint(), gs.gates[0].waypoint())


def test_bar_target_never_reaches_the_floor():
    """The lowest blue bar is at 0.880; `height - clearance` must still leave
    the drone flying rather than aiming at the ground."""
    bar = BarGate(y=STATION_Y[0], height=min(BLUE_BAR_HEIGHTS),
                  side=BarSide.BELOW)
    assert bar.target_z >= MIN_FLIGHT_Z
    assert bar.target_z < min(BLUE_BAR_HEIGHTS)


def test_gate_sequence_orders_and_completes():
    layout = CourseLayout((StationSpec(0, "red", 2.640),
                           StationSpec(1, "blue", 1.760)))
    gs = build_gates(layout)
    assert gs.n_gates == 4

    pos = np.array([0.88, ENTRY_Y - FWD * SPAWN_STANDOFF, 3.41])
    outcomes = []
    for _ in range(20000):
        wp = gs.waypoint()
        if wp is None:
            break
        prev = pos.copy()
        step = wp - pos
        n = float(np.linalg.norm(step))
        pos = pos + step / max(n, 1e-9) * min(0.05, n)
        o = gs.update(prev, pos)
        if o is not Outcome.NONE:
            outcomes.append(o)
        if n < 1e-3:                       # reached the waypoint; nudge past it
            nudge = np.array([0.0, FWD * 0.05, 0.0])
            o = gs.update(pos, pos + nudge)
            pos = pos + nudge
            if o is not Outcome.NONE:
                outcomes.append(o)
    assert gs.done, f"course not completed; outcomes={outcomes}"
    assert all(o is Outcome.PASS for o in outcomes), outcomes


# --------------------------------------------------------------------------
# world -- the real IMAV arena
# --------------------------------------------------------------------------

def test_scale_relates_sdf_units_to_competition_millimetres():
    """The arena is the competition course x2.2. Mixing the two frames is the
    bug this pins: the entry window came from the SDF and the bar heights from
    the spec sheet, which put a 3 m dive out of every window."""
    assert math.isclose(max(RED_BAR_HEIGHTS), 1.980 * SCALE, rel_tol=1e-3)
    assert math.isclose(min(BLUE_BAR_HEIGHTS), 0.400 * SCALE, rel_tol=1e-3)
    assert math.isclose(STATION_SPACING, 1.000 * SCALE, rel_tol=1e-3)
    # ...and the window is in the same frame as the bars: the tallest red bar
    # sits just above the window it is seen through, not three metres below it.
    assert RED_Z_LO < max(RED_BAR_HEIGHTS) < RED_Z_HI + 1.0


def test_station_planes_match_the_sdf_in_travel_order():
    """The SDF's red_bar / blue_bar1 / blue_bar2 planes, ordered as the drone
    meets them, so slot 0 is the first station past the entry wall."""
    assert set(STATION_Y) == {-9.90, -7.70, -5.50}
    steps = [b - a for a, b in zip(STATION_Y, STATION_Y[1:])]
    assert all(math.isclose(s, FWD * STATION_SPACING) for s in steps), steps
    assert math.isclose(abs(STATION_Y[0] - ENTRY_Y), STATION_SPACING),         "slot 0 should sit one spacing past the entry wall"


def test_gate_planes_advance_monotonically_along_the_course():
    """Every gate plane must be further along the direction of travel than the
    last, whichever direction that is."""
    rng = np.random.default_rng(0)
    for n in range(4):
        gates = build_gates(sample_layout(rng, n, "all")).gates
        ys = [g.y for g in gates]
        assert all(FWD * (b - a) > 0 for a, b in zip(ys, ys[1:])), ys
        assert ys[0] == ENTRY_Y and ys[-1] == EXIT_Y


def test_only_bars_vary():
    """Count, colour and height vary. Nothing else -- the station planes are
    the SDF's and the walls never move."""
    rng = np.random.default_rng(7)
    seen_y, seen_colour, seen_height, seen_n = set(), set(), set(), set()
    for _ in range(300):
        lay = sample_layout(rng, int(rng.integers(0, 4)), "all")
        seen_n.add(lay.n_stations)
        for st in lay.stations:
            seen_y.add(st.y)
            seen_colour.add(st.colour)
            seen_height.add(st.height)
    assert seen_y <= set(STATION_Y)
    assert seen_colour == {"red", "blue"}
    assert seen_height <= set(RED_BAR_HEIGHTS) | set(BLUE_BAR_HEIGHTS)
    assert seen_n == {0, 1, 2, 3}


def test_station_xml_is_named_and_classified():
    layout = CourseLayout((StationSpec(0, "red", 4.356),
                           StationSpec(1, "blue", 0.880)))
    xml = "\n".join(station_xml(s) for s in layout.stations)
    assert xml.count("<geom") == xml.count("<body")
    assert xml.count("<geom name=") == xml.count("<geom")

    names = [ln.split('"')[1] for ln in xml.splitlines() if "<body name=" in ln]
    classes = {classify_body_name(n) for n in names}
    assert SemClass.RED_BAR in classes and SemClass.BLUE_BAR in classes
    assert SemClass.WALL in classes                     # posts and feet


def test_arena_bodies_classify_into_every_class():
    """Body names taken from the SDF itself. Classification is by body because
    SdfToMjcf names its geoms g1, g2, ... and puts the meaning on the body."""
    cases = {
        "obs_wall_red_L": SemClass.RED_WINDOW,
        "exit_wall_red_top": SemClass.RED_WINDOW,
        "obs_wall_blue_bot": SemClass.BLUE_WINDOW,
        "obs_wall_bd_M": SemClass.WALL,
        "obs_wall_leg_L": SemClass.WALL,
        "room_N": SemClass.WALL,
        "arena_floor": SemClass.FLOOR,
        "strip_C": SemClass.FLOOR,
        "tube_A_post": SemClass.OBSTACLE,
        "tube_B_horiz_lo": SemClass.OBSTACLE,
        "box_red_floor": SemClass.OBSTACLE,
        "turbine_assembly": SemClass.OBSTACLE,
        "ring_board_assembly": SemClass.OBSTACLE,
        "station0_red_bar": SemClass.RED_BAR,
        "station2_blue_bar": SemClass.BLUE_BAR,
        "station1_red_post_L": SemClass.WALL,
        None: SemClass.FREE,
    }
    for name, want in cases.items():
        assert classify_body_name(name) is want, (name, classify_body_name(name))


def test_bars_span_past_their_posts():
    """A bar you can fly around laterally is not an obstacle. The SDF's red bar
    is 4.4 long against posts at x = +-1.65."""
    assert RED_BAR_LENGTH / 2 > POST_X


def test_held_out_heights_are_disjoint():
    assert not set(TRAIN_RED_HEIGHTS) & set(EVAL_RED_HEIGHTS)
    assert not set(TRAIN_BLUE_HEIGHTS) & set(EVAL_BLUE_HEIGHTS)
    assert set(TRAIN_RED_HEIGHTS) | set(EVAL_RED_HEIGHTS) == set(RED_BAR_HEIGHTS)
    assert set(TRAIN_BLUE_HEIGHTS) | set(EVAL_BLUE_HEIGHTS) == set(BLUE_BAR_HEIGHTS)


def test_sample_layout_respects_split():
    rng = np.random.default_rng(1)
    for _ in range(50):
        for st in sample_layout(rng, 3, "train").stations:
            allowed = (TRAIN_RED_HEIGHTS if st.colour == "red"
                       else TRAIN_BLUE_HEIGHTS)
            assert st.height in allowed


def test_all_layouts_covers_both_orders():
    combos = all_layouts(2, "train")
    assert len(combos) == 16          # 4 options per slot, 2 slots
    orders = {tuple(s.colour for s in c.stations) for c in combos}
    assert ("red", "blue") in orders and ("blue", "red") in orders


def test_spawn_is_in_front_of_the_entry_wall_facing_down_course():
    rng = np.random.default_rng(3)
    for _ in range(200):
        pos, yaw = spawn_pose(rng)
        standoff = -FWD * (pos[1] - ENTRY_Y)   # spawn is behind the wall
        assert SPAWN_STANDOFF - SPAWN_STANDOFF_JIT - 1e-9 <= standoff
        assert standoff <= SPAWN_STANDOFF + SPAWN_STANDOFF_JIT + 1e-9
        assert yaw == YAW_DOWN_COURSE
        assert math.isclose(math.sin(yaw), FWD, abs_tol=1e-9)   # nose downcourse
        # centred on the RED opening, so the target is the first thing seen
        assert RED_X_LO < pos[0] < RED_X_HI
        assert RED_Z_LO < pos[2] < RED_Z_HI


# --------------------------------------------------------------------------
# image pipeline
# --------------------------------------------------------------------------

def test_min_pool_preserves_a_one_pixel_bar():
    depth = np.full((512, 512), 10.0, np.float32)
    depth[256, :] = 2.0
    out = downsample_depth(depth, 128)
    assert (out.min(axis=1) < 3.0).sum() == 1
    mean_pooled = depth.reshape(128, 4, 128, 4).mean(axis=(1, 3))
    assert mean_pooled.min() > 7.0        # a mean would erase the bar


def test_seg_downsample_invents_no_classes():
    seg = np.zeros((512, 512), np.int32)
    seg[100:140] = 4
    seg[300:340] = 5
    out = downsample_seg(seg, 128)
    assert set(np.unique(out).tolist()) <= {0, 4, 5}


def test_rgb_downsample_shape_and_range():
    rgb = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    out = downsample_rgb(rgb, 128)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8


def test_proximity_weight_profile():
    """Distances are SDF units; divide by 2.2 for competition metres."""
    w = proximity_weight(np.array([0.0, 2.2, 6.6, 16.0, 30.0]))
    assert math.isclose(w[0], 1.0, abs_tol=1e-6)
    assert 0.70 < w[1] < 0.85          # one station spacing
    assert 0.25 < w[2] < 0.40          # three spacings
    assert math.isclose(w[3], 0.05, abs_tol=1e-6)
    assert math.isclose(w[4], 0.05, abs_tol=1e-6)      # floored, never zero
    d = np.linspace(0.5, 16.0, 200)
    assert np.all(np.diff(proximity_weight(d)) <= 1e-9)   # monotone decreasing


def test_seg_to_classes_handles_both_buffer_shapes():
    lut = np.array([0, 1, 4, 5], dtype=np.uint8)
    flat = np.full((512, 512), 2, dtype=np.int32)
    stacked = np.stack([flat, np.zeros_like(flat)], axis=-1)
    assert np.array_equal(seg_to_classes(flat, lut, 128),
                          seg_to_classes(stacked, lut, 128))
    assert seg_to_classes(flat, lut, 128)[0, 0] == 4


def test_seg_to_classes_ignores_out_of_range_ids():
    lut = np.array([0, 1], dtype=np.uint8)
    buf = np.full((512, 512), -1, dtype=np.int32)     # sky
    assert seg_to_classes(buf, lut, 128).max() == 0


def test_noise_disabled_is_a_noop():
    rng = np.random.default_rng(0)
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    depth = np.random.rand(64, 64).astype(np.float32) * 10
    r, d = corrupt(rgb, depth, NoiseConfig.disabled(), rng)
    assert np.array_equal(r, rgb) and np.array_equal(d, depth)


def test_noise_is_bounded_and_scales():
    rng = np.random.default_rng(0)
    depth = np.full((256, 256), 4.0, np.float32)
    strong = corrupt(np.zeros((256, 256, 3), np.uint8), depth,
                     NoiseConfig(), rng)[1]
    weak = corrupt(np.zeros((256, 256, 3), np.uint8), depth,
                   NoiseConfig().scaled(0.1), rng)[1]
    clean = np.abs(strong - 4.0)
    assert np.median(clean[clean < 1.0]) < 0.1
    assert (np.abs(weak - 4.0).mean() < np.abs(strong - 4.0).mean())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
