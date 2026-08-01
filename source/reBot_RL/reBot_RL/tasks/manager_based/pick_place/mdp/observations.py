# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the pick-and-place MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .common import basket_centers_local, placed_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def basket_center_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Env-local xy of the basket center (2-d). v1: per-env randomized; v0: constant."""
    return basket_centers_local(env)


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


def objects_placed_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One flag per object: 1.0 if it is inside the basket. Shape (N, num_objects)."""
    return placed_mask(env).float()


def objects_canonical(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Both object poses + placed flags, ordered TARGET-first (16-d).

    The target slot always holds the nearest not-yet-placed object (the same selection
    the reach reward uses), the other slot the remaining object; the two placed flags
    are permuted identically. This makes the grasp policy slot-agnostic: every grasp
    ever practiced trains the same input pathway, for either object in either order --
    with fixed slots, run 11 mastered object_a and had to learn object_b's grasp from
    scratch through a pathway that had never driven one.
    """
    from .common import OBJECT_NAMES

    ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    placed = placed_mask(env)
    poses = []
    dists = []
    for i, name in enumerate(OBJECT_NAMES):
        poses.append(object_pose_in_robot_root_frame(env, SceneEntityCfg(name), robot_cfg))
        d = torch.linalg.norm(env.scene[name].data.root_pos_w.torch - ee_w, dim=1)
        dists.append(torch.where(placed[:, i], torch.full_like(d, torch.inf), d))
    # (N,) index of the target object; all-placed envs fall back to spawn order
    target_is_b = (dists[1] < dists[0]).unsqueeze(1)
    first = torch.where(target_is_b, poses[1], poses[0])
    second = torch.where(target_is_b, poses[0], poses[1])
    flags = placed.float()
    flags_first = torch.where(target_is_b[:, 0], flags[:, 1], flags[:, 0]).unsqueeze(1)
    flags_second = torch.where(target_is_b[:, 0], flags[:, 0], flags[:, 1]).unsqueeze(1)
    return torch.cat([first, second, flags_first, flags_second], dim=1)
