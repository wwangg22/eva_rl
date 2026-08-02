# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the clutter-extraction task.

A target block sits in a tight row of four distractors. Toppling any distractor ends the
episode. The row spacing is narrower than the gripper's outer width when open, so the
policy either threads the fingers in precisely or first pushes a neighbour aside -- the
correct first action is often *not* a grasp, which is the credit-assignment problem plain
pick-and-place never poses.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from . import common

#: names of the four distractor rigid objects, in row order
DISTRACTOR_NAMES = ("distractor_0", "distractor_1", "distractor_2", "distractor_3")
#: block half-extents (x, y, z) [m] -- 30 mm across the fingers, in the measured sweet spot
CL_BLOCK_HALF = (0.018, 0.015, 0.035)
#: a block counts as toppled once its own +z axis falls this far from world +z
TOPPLE_DOT = 0.75
#: goal: carry the target here (env-local xy) and set it down
GOAL_XY = (0.185, -0.185)
GOAL_RADIUS = 0.045
#: the target must clear the row before it counts as extracted
EXTRACT_Z = 0.090


def _up_z(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    q = common.object_quat(env, name)
    up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3))
    return up[:, 2]


def any_distractor_toppled(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination: the constraint that gives the task its teeth."""
    return torch.stack([_up_z(env, n) < TOPPLE_DOT for n in DISTRACTOR_NAMES], dim=1).any(dim=1)


def distractors_disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Total planar displacement of the distractors from their spawn poses [m].

    A shaping signal for "be gentle" that is softer than the termination: nudging a
    neighbour aside is allowed (and sometimes necessary), knocking it over is not.
    """
    if not hasattr(env, "_clutter_spawn_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    cur = torch.stack([common.object_pos_local(env, n)[:, :2] for n in DISTRACTOR_NAMES], dim=1)
    return (cur - env._clutter_spawn_xy).norm(dim=-1).sum(dim=1)


def target_extracted(env: ManagerBasedRLEnv, name: str = "target") -> torch.Tensor:
    """Target lifted clear of the row, with every distractor still standing."""
    return (common.object_pos_local(env, name)[:, 2] > EXTRACT_Z) & ~any_distractor_toppled(env)


def target_at_goal(env: ManagerBasedRLEnv, name: str = "target") -> torch.Tensor:
    """Target set down inside the goal zone, nothing toppled."""
    p = common.object_pos_local(env, name)
    goal = torch.tensor(GOAL_XY, device=p.device)
    return (
        ((p[:, :2] - goal).norm(dim=1) < GOAL_RADIUS)
        & (p[:, 2] < 0.055)
        & ~any_distractor_toppled(env)
    )


def reach_target(env: ManagerBasedRLEnv, std: float, ee_frame_cfg: SceneEntityCfg,
                 name: str = "target") -> torch.Tensor:
    d = (common.ee_pos_local(env, ee_frame_cfg) - common.object_pos_local(env, name)).norm(dim=1)
    return 1.0 - torch.tanh(d / std)


def target_to_goal(env: ManagerBasedRLEnv, std: float, name: str = "target") -> torch.Tensor:
    """Gated carry reward: only pays once the target is clear of the row."""
    p = common.object_pos_local(env, name)
    goal = torch.tensor(GOAL_XY, device=p.device)
    d = (p[:, :2] - goal).norm(dim=1)
    return (p[:, 2] > 0.070).float() * (1.0 - torch.tanh(d / std))


def clutter_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distractor xy offsets from the target, plus their up-axes. Shape (N, 12)."""
    tp = common.object_pos_local(env, "target")
    feats = []
    for n in DISTRACTOR_NAMES:
        feats.append(common.object_pos_local(env, n)[:, :2] - tp[:, :2])
        feats.append(_up_z(env, n).unsqueeze(-1))
    return torch.cat(feats, dim=-1)


class clutter_success(ManagerTermBase):
    """Sparse per-step success; logs the episode success rate on reset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/clutter_success_rate"] = (
            self._ever[env_ids].float().mean()
        )
        self._ever[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, name: str = "target") -> torch.Tensor:
        ok = target_at_goal(env, name)
        self._ever.copy_(self._ever | ok)
        return ok.float()


def record_spawn_xy(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset event: remember where the distractors started, for ``distractors_disturbed``."""
    cur = torch.stack([common.object_pos_local(env, n)[:, :2] for n in DISTRACTOR_NAMES], dim=1)
    if not hasattr(env, "_clutter_spawn_xy"):
        env._clutter_spawn_xy = cur.clone()
    else:
        env._clutter_spawn_xy[env_ids] = cur[env_ids]
