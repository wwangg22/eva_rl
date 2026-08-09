# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the reconstructed-workstation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .common import CLUTTER_NAMES, TARGET_NAME, object_pos_local

#: displacement that counts as "disturbed" [m]. Calibrated the way ``clutter``'s DISTURB_TOL
#: was: against the solver's own noise floor, measured under a null action, not guessed.
#: Verified by ``scripts/test_workstation_env.py``'s null-action control.
DISTURB_TOL = 0.002


def target_dropped(env: ManagerBasedRLEnv, minimum_height: float = -0.05) -> torch.Tensor:
    """The cube left the desk."""
    return object_pos_local(env, TARGET_NAME)[:, 2] < minimum_height


def clutter_disturbed_term(env: ManagerBasedRLEnv, tol: float = DISTURB_TOL) -> torch.Tensor:
    """Any clutter object moved further than ``tol`` from where it spawned.

    Only wired up in the ``-Strict-v0`` variant. In the default task the clutter is an
    obstacle, not a constraint, and this stays off -- a constraint the task does not actually
    impose should not be silently costing reward.
    """
    buf = getattr(env, "_clutter_spawn_xy", None)
    if buf is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, name in enumerate(CLUTTER_NAMES):
        d = torch.linalg.norm(object_pos_local(env, name)[:, :2] - buf[:, i], dim=1)
        out = out | (d > tol)
    return out
