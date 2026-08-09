# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared geometry and predicates for the reconstructed-workstation pick-and-place task.

**Single source of truth.** The scene builds its bodies from these numbers, the box USD is
authored from them (``scripts/author_workstation_box_usd.py`` imports this module), and every
reward / termination / observation checks against the same ones. The v0/v1 basket does *not*
do this -- its geometry lives both in the env cfg and in a separately generated USD, and the
two can silently disagree.

Everything here is in the **env frame**: robot base at the origin, desk surface at z = 0,
+x away from the robot, +z up, quaternions **(x, y, z, w)**.

Relation to the reconstruction
------------------------------
The Re3Sim marker frame puts its origin on the marker's *top* surface, which sits on the iPad
that the arm's base plate replaces. The desk surface is therefore ``z_marker = -0.0065`` and
maps to ``z_env = 0`` by ``z_env = z_marker + 0.0065``. Nothing in this module needs the
marker frame; it is recorded so the splat/mesh placement transform has one stated convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------------------
# Tool frame
# ---------------------------------------------------------------------------------------

#: Offset from ``gripper_end`` to the true TCP, in the link frame [m]. Measured
#: (docs/CHALLENGE_SUITE.md C10): with the fingers shut the two finger-body origins coincide
#: and that point is the grasp point. **Never** -0.075 -- the lift task's stale value is
#: 33.1 mm too far forward and makes every scripted grasp close on air.
TCP_OFFSET = (-0.0419, 0.0, 0.0)

# ---------------------------------------------------------------------------------------
# The captured objects. Caliper measurements, data/captures/2026-08-05/measurements.txt.
# ---------------------------------------------------------------------------------------

#: rubix cube -- the **grasp target**. 56 mm on a side, 73 g.
CUBE_SIZE = 0.056
CUBE_MASS = 0.073

#: roll of tape -- clutter. 91 mm outside diameter, 24 mm tall, 42 g. Modelled as a solid
#: cylinder: it is an obstacle, never grasped, so the central hole changes nothing that is
#: measured. Note 91 mm exceeds the gripper's 89.1 mm *commanded* opening (C3), which is one
#: reason it is clutter rather than a target.
TAPE_DIAMETER = 0.091
TAPE_HEIGHT = 0.024
TAPE_MASS = 0.042

#: tape measure -- clutter. 36 mm tall, 71.5 mm longest, 184 g.
#: ⚠ The third dimension was NOT measured (the capture sheet records height + longest only).
#: 64 mm is an estimate. It affects the obstacle footprint only, never a grasp.
TAPEMEASURE_LONG = 0.0715
TAPEMEASURE_ACROSS = 0.064
TAPEMEASURE_HEIGHT = 0.036
TAPEMEASURE_MASS = 0.184

#: the one object a policy has to pick up
TARGET_NAME = "cube"
#: obstacles: real bodies occupying real space on the desk. Not grasp targets.
CLUTTER_NAMES = ("rolloftape", "tapemeasure")
#: canonical order for anything that emits per-object quantities
OBJECT_NAMES = (TARGET_NAME,) + CLUTTER_NAMES

# ---------------------------------------------------------------------------------------
# The open box (receptacle). Measured 2026-08-05: outer longest 218 mm, inner width 212 mm,
# wall 3 mm, height 93 mm, interior depth 87 mm (so the floor is 6 mm thick).
#
# ⚠ BOX_OUTER_Y is an ASSUMPTION. The capture sheet gives height and longest only, so the
# short horizontal dimension was never measured. 150 mm is a plausible shipping-box value and
# is the one number here that should be corrected from the reconstruction or a caliper.
# ---------------------------------------------------------------------------------------
BOX_OUTER_X = 0.218
BOX_OUTER_Y = 0.150
BOX_HEIGHT = 0.093
BOX_WALL = 0.003
BOX_FLOOR_THICKNESS = 0.006  # 93 mm outer height - 87 mm interior depth
BOX_INNER_X = BOX_OUTER_X - 2 * BOX_WALL  # 0.212, matches the measured inner width
BOX_INNER_Y = BOX_OUTER_Y - 2 * BOX_WALL

# ---------------------------------------------------------------------------------------
# "Placed" -- the success predicate.
#
# LESSONS_INHERITED B1: success predicates are usually weaker than they read. The slot task's
# `is_inserted` bounded z only from *below*, so a block resting on the walls, or still
# dangling in the gripper, passed; one probe scored 93.8 % with a mean lateral error that was
# geometrically impossible inside the channel.
#
# So this one is bounded on BOTH sides and requires the object to have come to rest:
#   * inside the interior footprint, with a margin off the wall faces
#   * ABOVE the interior floor  (it did not clip through the bottom)
#   * BELOW a ceiling well under the rim (it is not balanced on the rim, and not being held
#     above the box by a gripper that has not let go)
#   * SETTLED (speed below threshold), so an object still in flight over the box is not
#     counted the instant it crosses the plane
# ---------------------------------------------------------------------------------------

#: margin inside the wall faces, per side [m]
PLACED_MARGIN = 0.012
#: a cube resting on the interior floor has its root at 0.006 + 0.028 = 0.034 m.
#: The rim is at 0.093. This ceiling sits between the two, so "resting inside" passes and
#: "perched on the rim" or "still held above the box" both fail.
PLACED_Z_MAX = 0.062
#: ...and it must be above the interior floor, not clipped through it
PLACED_Z_MIN = 0.010
#: settled: linear speed below this [m/s]
PLACED_MAX_SPEED = 0.05


def _t(x):
    """Isaac Lab 3.0 wraps some buffers in a container exposing ``.torch``."""
    return x.torch if hasattr(x, "torch") else x


def object_pos_local(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Object root position in the env frame (desk surface at z = 0). Shape (N, 3)."""
    return _t(env.scene[name].data.root_pos_w) - env.scene.env_origins


def object_quat(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Object root orientation, (x, y, z, w). Shape (N, 4)."""
    return _t(env.scene[name].data.root_quat_w)


def object_speed(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Linear speed of the object root [m/s]. Shape (N,)."""
    return _t(env.scene[name].data.root_lin_vel_w).norm(dim=1)


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    """Yaw about world +z from an (x, y, z, w) quaternion. Shape (N,)."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def box_centers_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env box centre xy in the env frame. Shape (N, 2).

    Read from the buffer ``mdp.reset_box`` writes, **not** from sim data: during a reset
    event ``data.root_pos_w`` is lazily refreshed and still holds the PRE-reset pose. That
    aliasing is what the v1 basket's docstring warns about and it is a real bug source.
    """
    buf = getattr(env, "_box_center_local", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device)
        buf[:, 0] = BOX_DEFAULT_CENTER[0]
        buf[:, 1] = BOX_DEFAULT_CENTER[1]
        env._box_center_local = buf
    return buf


def box_yaws(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env box yaw about +z [rad]. Shape (N,)."""
    buf = getattr(env, "_box_yaw", None)
    if buf is None:
        buf = torch.full((env.num_envs,), BOX_DEFAULT_YAW, dtype=torch.float, device=env.device)
        env._box_yaw = buf
    return buf


#: where the box sits before the first reset event runs
BOX_DEFAULT_CENTER = (0.30, 0.0)
BOX_DEFAULT_YAW = 1.5707963


def in_box_frame(env: ManagerBasedRLEnv, pos: torch.Tensor) -> torch.Tensor:
    """Rotate an env-frame xy (N, 2 or 3) into the box's own axis-aligned frame. Shape (N, 2).

    The box is randomly yawed, so an axis-aligned test in the env frame would accept points
    outside it and reject points inside. This is the difference between a predicate that
    means what it says and one that merely correlates with it.
    """
    d = pos[:, :2] - box_centers_local(env)
    c, s = torch.cos(-box_yaws(env)), torch.sin(-box_yaws(env))
    return torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], dim=1)


def placed_mask(env: ManagerBasedRLEnv, name: str = TARGET_NAME) -> torch.Tensor:
    """The success predicate: the object is at rest inside the box. Shape (N,) bool."""
    pos = object_pos_local(env, name)
    b = in_box_frame(env, pos)
    return (
        (b[:, 0].abs() < BOX_INNER_X / 2 - PLACED_MARGIN)
        & (b[:, 1].abs() < BOX_INNER_Y / 2 - PLACED_MARGIN)
        & (pos[:, 2] > PLACED_Z_MIN)
        & (pos[:, 2] < PLACED_Z_MAX)
        & (object_speed(env, name) < PLACED_MAX_SPEED)
    )


def over_box(env: ManagerBasedRLEnv, name: str = TARGET_NAME) -> torch.Tensor:
    """Object is within the box footprint in xy, at any height. Shape (N,) bool.

    Used for shaping and for the failure taxonomy -- it separates "never got there" from
    "got there and did not end up inside".
    """
    b = in_box_frame(env, object_pos_local(env, name))
    return (b[:, 0].abs() < BOX_INNER_X / 2) & (b[:, 1].abs() < BOX_INNER_Y / 2)
