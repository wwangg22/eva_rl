# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared task geometry for the pick-and-place MDP.

Single source of truth for the basket pose/dimensions and the "object is placed"
predicate -- the scene config builds the basket walls from these constants and every
reward/termination/observation term checks placement against the same numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# env-local xy of the basket center -- v0 only (v1 randomizes the basket per env per
# episode; read mdp.basket_centers_local instead of this constant). Radius from the robot
# base is ~0.25 m, well inside
# the workspace the lift teacher covered (goal commands out to r ~0.34 m). Run 3 with the
# basket at (0.24, -0.16)/r=0.29 plateaued hovering 15 cm short of it -- keep the basket
# comfortably reachable. Objects spawn on +y, basket sits on -y.
BASKET_CENTER = (0.22, -0.12)
# inner wall faces sit at +/- this distance from the center
BASKET_INNER_HALF = 0.055
BASKET_WALL_THICKNESS = 0.006
# wall top at z = 0.04: high enough that objects cannot be pushed in (can root z 0.015,
# box root z 0.028), low enough that the arm's side-on gripper clears it easily
BASKET_WALL_HEIGHT = 0.04
BASKET_FLOOR_THICKNESS = 0.004

# "placed" predicate: object root inside the basket in xy (5 mm margin off the wall
# faces) and below the rim. z > -0.01 excludes anything that somehow clipped through.
PLACED_XY_HALF = 0.05
PLACED_Z_MAX = 0.048

# scene entity names of the two task objects, in observation order
OBJECT_NAMES = ("object_a", "object_b")


def object_pos_local(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Object root position in the env-local frame (table surface at z=0). Shape (N, 3)."""
    return env.scene[name].data.root_pos_w.torch - env.scene.env_origins


def basket_centers_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env basket center xy in the env-local frame. Shape (N, 2).

    v1 scenes have a movable kinematic ``basket`` rigid object: return the per-env buffer
    the reset event (``mdp.reset_basket``) writes, lazily initialized to the cfg default.
    v0 scenes (static basket cuboids baked at config time): the ``BASKET_CENTER``
    constant expanded per env, so every consumer of this function serves both versions.
    """
    if "basket" in env.scene.rigid_objects:
        buf = getattr(env, "_basket_center_local", None)
        if buf is None:
            buf = torch.tensor(BASKET_CENTER, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            env._basket_center_local = buf
        return buf
    return torch.tensor(BASKET_CENTER, device=env.device).expand(env.num_envs, 2)


def placed_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Boolean mask of shape (N, num_objects): object i is inside the (per-env) basket."""
    centers = basket_centers_local(env)
    masks = []
    for name in OBJECT_NAMES:
        pos = object_pos_local(env, name)
        inside = (
            ((pos[:, 0] - centers[:, 0]).abs() < PLACED_XY_HALF)
            & ((pos[:, 1] - centers[:, 1]).abs() < PLACED_XY_HALF)
            & (pos[:, 2] > -0.01)
            & (pos[:, 2] < PLACED_Z_MAX)
        )
        masks.append(inside)
    return torch.stack(masks, dim=1)
