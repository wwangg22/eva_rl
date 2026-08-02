# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the challenge tasks."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from . import common


def block_pose_in_root(env: ManagerBasedRLEnv, name: str = "block",
                       robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Block pose (pos + quat, 7-D) expressed in the robot root frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[name]
    pos, quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        obj.data.root_pos_w.torch,
        obj.data.root_quat_w.torch,
    )
    return torch.cat([pos, quat], dim=-1)


def slot_frame(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Task-frame error the policy actually has to null: (depth, lateral, yaw, engaged).

    Giving the error in the slot's own frame rather than raw world poses is what Factory
    does (``fingertip_pos_rel_fixed``) and it is why a 21-dim observation suffices there.
    """
    return torch.stack(
        [
            common.insertion_depth(env, name),
            common.lateral_error(env, name),
            common.yaw_error(env, name),
            common.is_inserted(env, name).float(),
        ],
        dim=-1,
    )
