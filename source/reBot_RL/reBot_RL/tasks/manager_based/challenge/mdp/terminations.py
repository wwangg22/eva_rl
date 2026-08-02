# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the challenge tasks."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import common


def block_dropped(env: ManagerBasedRLEnv, minimum_height: float = -0.05,
                  name: str = "block") -> torch.Tensor:
    """Block pushed off the table."""
    return common.object_pos_local(env, name)[:, 2] < minimum_height


def block_toppled(env: ManagerBasedRLEnv, max_tilt: float = 0.6,
                  name: str = "block") -> torch.Tensor:
    """Block lying on its side -- it can no longer be inserted, so the episode is over.

    Measured as the world-z component of the block's own +z axis.
    """
    from isaaclab.utils.math import quat_apply

    quat = common.object_quat(env, name)
    up = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(quat.shape[0], 3))
    return up[:, 2] < max_tilt
