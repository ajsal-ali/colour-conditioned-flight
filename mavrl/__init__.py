"""MAVRL: memory-augmented, varying-speed RL for the IMAV2026 obstacle course.

Layered so the geometry and gate logic import cleanly without mujoco/torch --
that is what makes them testable on a machine with no simulator installed.

    course_gates  pure geometry: gate planes, traversal outcomes, gate ordering
    course_world  the real IMAV arena (assets/imav2026/imav2026_scaled.sdf) with the
                  bar stations randomized; layout sampling, semantic class map

Everything above those (env, SeVAE, memory, training) pulls in mujoco or torch.
`course_world`'s XML path additionally needs the SDF and `imav_teleop`, but its
constants and layout sampling do not.
"""

from mavrl.course_gates import (  # noqa: F401
    BarGate, BarSide, GateSequence, Opening, Outcome, WallGate,
)
from mavrl.course_world import (  # noqa: F401
    CourseLayout, SemClass, StationSpec,
    all_layouts, build_gates, sample_layout, spawn_pose,
)

__all__ = [
    "BarGate", "BarSide", "GateSequence", "Opening", "Outcome", "WallGate",
    "CourseLayout", "SemClass", "StationSpec",
    "all_layouts", "build_gates", "sample_layout", "spawn_pose",
]
