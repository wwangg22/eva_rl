# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the drawer-ordering task.

Two skills the pick-place task never asks for:

* **articulated interaction** -- the gripper has to follow a kinematic constraint it does
  not know, applying force along the drawer's slide axis and nowhere else;
* **irreversible precedence** -- the block cannot go in until the drawer is open, so the
  reward for the second stage is identically zero until the first is done. There is no
  smooth distance gradient bridging them, unlike carrying an object toward a basket.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

from . import common

#: cabinet origin in the env frame [m] -- mirrors drawer_env_cfg
#: x moved in from 0.290 and the whole cabinet lifted onto a plinth, both forced by
#: measurement: the usable TCP band is x ~ 0.22..0.26 and nothing below z = 44 mm can be
#: reached at all (scripts/analysis/tcp_floor.py). At the original height the handle sat at
#: z = 26 mm, i.e. the gripper could never have touched it.
CABINET_XY = (0.265, 0.0)
#: height of the plinth the cabinet stands on [m]
CABINET_Z = 0.045
#: full travel of the prismatic joint [m]; joint position is negative when pulled out
DRAWER_TRAVEL = 0.070
#: open enough to admit the block
OPEN_FRAC = 0.70
#: block must end up inside the drawer cavity
CAVITY_HALF = (0.033, 0.053)
#: cavity floor is 12 mm above the cabinet base, so on the plinth it sits at 57 mm
CAVITY_Z_MIN = CABINET_Z + 0.005
CAVITY_Z_MAX = CABINET_Z + 0.055


def drawer_opening(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """How far the drawer is pulled out, 0 (shut) .. 1 (fully out)."""
    q = env.scene[asset_cfg.name].data.joint_pos.torch[:, 0]
    return (-q / DRAWER_TRAVEL).clamp(0.0, 1.0)


def drawer_is_open(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return drawer_opening(env, asset_cfg) > OPEN_FRAC


def block_in_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                    name: str = "block") -> torch.Tensor:
    """Block inside the drawer cavity, tracked in the *drawer's own frame*.

    The cavity moves with the drawer, so a fixed world-space box would either miss it once
    it is pulled out or -- worse -- accept a block sitting on the table in front of a shut
    drawer.
    """
    p = common.object_pos_local(env, name)
    q = env.scene[asset_cfg.name].data.joint_pos.torch[:, 0]
    cx = CABINET_XY[0] + q  # cavity centre travels with the joint
    return (
        ((p[:, 0] - cx).abs() < CAVITY_HALF[0])
        & ((p[:, 1] - CABINET_XY[1]).abs() < CAVITY_HALF[1])
        & (p[:, 2] < CAVITY_Z_MAX)
        & (p[:, 2] > CAVITY_Z_MIN)
    )


def stowed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
           name: str = "block") -> torch.Tensor:
    """The task: drawer open AND block in the cavity."""
    return drawer_is_open(env, asset_cfg) & block_in_drawer(env, asset_cfg, name)


def reach_handle(env: ManagerBasedRLEnv, std: float, ee_frame_cfg: SceneEntityCfg,
                 asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reach the handle -- gated off once the drawer is open, so it stops competing.

    Left ungated it becomes a comfortable local optimum: hovering at the handle of an
    already-open drawer pays forever and the block never gets picked up.
    """
    handle_x = CABINET_XY[0] - 0.061 + env.scene[asset_cfg.name].data.joint_pos.torch[:, 0]
    tgt = torch.stack([
        handle_x,
        torch.full_like(handle_x, CABINET_XY[1]),
        torch.full_like(handle_x, CABINET_Z + 0.026),   # handle centre in the cabinet frame
    ], dim=-1)
    d = (common.ee_pos_local(env, ee_frame_cfg) - tgt).norm(dim=1)
    return (~drawer_is_open(env, asset_cfg)).float() * (1.0 - torch.tanh(d / std))


def opening_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Dense credit for pulling the drawer out. Stage one of the precedence chain."""
    return drawer_opening(env, asset_cfg)


def block_to_drawer(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg,
                    name: str = "block") -> torch.Tensor:
    """Carry reward, hard-gated on the drawer being open.

    This gate *is* the precedence constraint: before the drawer is open the term is
    exactly zero, so there is no gradient encouraging the policy to shove the block at a
    shut drawer.
    """
    p = common.object_pos_local(env, name)
    q = env.scene[asset_cfg.name].data.joint_pos.torch[:, 0]
    cx = CABINET_XY[0] + q
    tgt = torch.stack([cx, torch.full_like(cx, CABINET_XY[1])], dim=-1)
    d = (p[:, :2] - tgt).norm(dim=1)
    return drawer_is_open(env, asset_cfg).float() * (1.0 - torch.tanh(d / std))


def drawer_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(opening fraction, is_open, block_in_drawer). Shape (N, 3)."""
    return torch.stack([
        drawer_opening(env, asset_cfg),
        drawer_is_open(env, asset_cfg).float(),
        block_in_drawer(env, asset_cfg).float(),
    ], dim=-1)


class drawer_success(ManagerTermBase):
    """Sparse per-step success; logs the episode success rate on reset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/drawer_success_rate"] = (
            self._ever[env_ids].float().mean()
        )
        self._ever[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                 name: str = "block") -> torch.Tensor:
        ok = stowed(env, asset_cfg, name)
        self._ever.copy_(self._ever | ok)
        return ok.float()
