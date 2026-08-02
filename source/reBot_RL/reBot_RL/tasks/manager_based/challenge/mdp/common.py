# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry helpers shared by the challenge tasks.

Deliberately separate from ``pick_place/mdp/common.py``: that module hardcodes
``OBJECT_NAMES`` at module scope and its ``objects_canonical`` observation assumes exactly
two objects, so any change there would alter the environment behind the 87.9 % pick-place
result and everything downstream of it. This package keeps its own terms.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerData

# ---------------------------------------------------------------------------
# Slot fixture geometry. Single source of truth -- the scene builds its collision
# cuboids from these numbers and the success predicate reads the same ones, so the
# two can never silently disagree (a failure mode the v0/v1 basket does have: its
# geometry lives both in the env cfg and in a separately generated USD).
# ---------------------------------------------------------------------------

#: Offset from the ``gripper_end`` link to the true tool centre point, in the link's own
#: frame [m]. **Measured, not assumed**: with the fingers shut the two finger bodies'
#: origins coincide, and that coincident point is the grasp point. It sits 41.9 mm behind
#: ``gripper_end``.
#:
#: The value inherited from the lift task was -0.075, which is 33.1 mm too far forward. A
#: constant offset error is invisible in a reach reward -- the policy just learns a shifted
#: target -- but it is fatal to any scripted grasp, because the fingers are commanded to
#: close 33 mm past the object and shut on air every time. That was the cause of several
#: flatly wrong "this object cannot be grasped" measurements.
TCP_OFFSET = (-0.0419, 0.0, 0.0)

#: fixture centre in the robot/env frame [m]
SLOT_CENTER = (0.245, 0.0)
#: height of the slot floor above the table [m]
SLOT_FLOOR_Z = 0.020
#: block half-extents (x = insertion depth, y = graspable width, z = height) [m]
BLOCK_HALF = (0.0225, 0.0150, 0.0350)
#: how far the walls stand above the slot floor. Deliberately shorter than the block so
#: the gripper can hold the block's upper half while its lower half is inside the slot --
#: the fingers would otherwise collide with the walls (this arm cannot approach top-down,
#: so it cannot lower the block in from above; see docs/CHALLENGE_SUITE.md C1).
WALL_HEIGHT = 0.030
WALL_THICKNESS = 0.008
#: slot depth along the insertion axis [m]
SLOT_DEPTH = 0.070

#: insertion depth counted as success [m]
SUCCESS_DEPTH = 0.040
#: yaw misalignment tolerated at success [rad]
SUCCESS_YAW = 0.12


def slot_clearance_to_halfwidth(clearance: float) -> float:
    """Inner half-width of the slot for a given per-side radial clearance."""
    return BLOCK_HALF[1] + clearance


def object_pos_local(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Object root position in the env frame (table top is z = 0)."""
    return env.scene[name].data.root_pos_w.torch - env.scene.env_origins


def object_quat(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Object root orientation, (x, y, z, w) as everywhere in Isaac Lab 3.0."""
    return env.scene[name].data.root_quat_w.torch


def ee_pos_local(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """TCP position in the env frame."""
    data: FrameTransformerData = env.scene[ee_frame_cfg.name].data
    return data.target_pos_w.torch[..., 0, :] - env.scene.env_origins


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
    """Yaw about world +Z from an (x, y, z, w) quaternion."""
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def insertion_depth(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """How far the block has travelled into the slot along the insertion axis [m].

    The slot opens toward the robot, so the insertion axis is world +x and depth is
    measured from the slot mouth. Negative means the block is still outside.
    """
    mouth_x = SLOT_CENTER[0] - SLOT_DEPTH / 2
    return object_pos_local(env, name)[:, 0] - mouth_x


def lateral_error(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Absolute cross-slot (y) offset of the block from the slot centreline [m]."""
    return (object_pos_local(env, name)[:, 1] - SLOT_CENTER[1]).abs()


def yaw_error(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Absolute yaw misalignment of the block w.r.t. the slot axis [rad]."""
    return yaw_of(object_quat(env, name)).abs()


def is_inserted(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Success predicate: deep enough, on the centreline, and square to the slot.

    The lateral bound is the slot's own inner half-width, so a block that is physically
    between the walls always satisfies it -- the walls do the work, which is the point of
    the task. Depth and yaw are the parts the policy has to get right.
    """
    half = getattr(env.cfg, "slot_half_width", slot_clearance_to_halfwidth(0.0015))
    return (
        (insertion_depth(env, name) >= SUCCESS_DEPTH)
        & (lateral_error(env, name) <= half)
        & (yaw_error(env, name) <= SUCCESS_YAW)
        & (object_pos_local(env, name)[:, 2] > SLOT_FLOOR_Z - 0.005)
    )
