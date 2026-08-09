# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the reconstructed-workstation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .common import CLUTTER_NAMES, TARGET_NAME, box_centers_local, box_yaws, object_pos_local, placed_mask


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object root pose (pos + quat, 7-d) in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    return torch.cat([pos_b, quat_b], dim=1)


def target_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The cube's pose in the robot root frame (7-d)."""
    return object_pose_in_robot_root_frame(env, SceneEntityCfg(TARGET_NAME))


def box_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Box centre xy plus its yaw as (cos, sin) (4-d).

    Yaw goes in as a **cos/sin pair**, not an angle: the box yaw is drawn near +/-pi/2 and a
    raw angle wraps there, so a network would see two nearly-identical boxes as maximally
    different inputs.
    """
    yaw = box_yaws(env)
    return torch.cat([box_centers_local(env), torch.cos(yaw).unsqueeze(1),
                      torch.sin(yaw).unsqueeze(1)], dim=1)


def clutter_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Each clutter object's xy **relative to the cube**, plus its height (3 per object, 6-d).

    Relative rather than absolute, for the same reason the clutter task uses relative offsets:
    what matters to a grasp is whether an obstacle is in the way of *this* object, and an
    absolute position makes the network re-derive that difference in every spawn.
    """
    tgt = object_pos_local(env, TARGET_NAME)
    out = []
    for name in CLUTTER_NAMES:
        p = object_pos_local(env, name)
        out.append(torch.cat([p[:, :2] - tgt[:, :2], p[:, 2:3]], dim=1))
    return torch.cat(out, dim=1)


def target_placed_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 once the cube is at rest inside the box. Shape (N, 1)."""
    return placed_mask(env).float().unsqueeze(1)
