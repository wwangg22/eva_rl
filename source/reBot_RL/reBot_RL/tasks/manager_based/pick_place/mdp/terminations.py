# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the pick-and-place MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .common import OBJECT_NAMES, object_pos_local

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def any_object_dropped(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """Terminate when any object falls below the table surface (pushed off the edge)."""
    dropped = [object_pos_local(env, name)[:, 2] < minimum_height for name in OBJECT_NAMES]
    return torch.stack(dropped, dim=1).any(dim=1)
