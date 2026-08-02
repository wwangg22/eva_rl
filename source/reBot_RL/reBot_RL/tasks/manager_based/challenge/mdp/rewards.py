# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the challenge tasks.

The shaping follows NVIDIA's Factory recipe (arXiv 2205.03532, reward detailed in
arXiv 2408.04587 App. B) rather than the pick-place task's hand-tuned ladder: a
multi-scale squashed keypoint distance plus staged engaged/success bonuses. The
multi-scale kernel *is* the curriculum -- a coarse term pulls the block toward the slot
from anywhere, a fine term only pays inside the last few millimetres -- which is what a
tight-tolerance task needs and what a single tanh cannot provide.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

from . import common


def squashing_fn(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Factory's bounded reward kernel ``1 / (exp(a x) + b + exp(-a x))``.

    ``a`` sets the length scale (a = 100 pays out over ~1 cm, a = 1000 over ~1 mm) and
    ``b`` the ceiling. Verbatim from ``isaaclab_tasks/direct/factory/factory_utils.py``.
    """
    return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))


def _keypoint_dist(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Mean distance from the block's corner keypoints to their fully-inserted poses.

    Corner keypoints rather than Factory's collinear ones: this arm cannot lock roll and
    pitch out of its action space the way Factory does, so the reward has to see all six
    DoF of the error or a rolled block reads as perfectly placed.
    """
    pos = common.object_pos_local(env, name)
    quat = common.object_quat(env, name)
    hx, hy, hz = common.BLOCK_HALF
    corners = torch.tensor(
        [[+hx, +hy, +hz], [+hx, -hy, -hz], [-hx, +hy, -hz], [-hx, -hy, +hz]],
        device=pos.device,
    )
    goal = torch.tensor(
        [common.SLOT_CENTER[0] + common.SUCCESS_DEPTH - common.SLOT_DEPTH / 2,
         common.SLOT_CENTER[1],
         common.SLOT_FLOOR_Z + common.BLOCK_HALF[2]],
        device=pos.device,
    )
    from isaaclab.utils.math import quat_apply

    n = corners.shape[0]
    cur = pos.unsqueeze(1) + quat_apply(quat.unsqueeze(1).expand(-1, n, -1), corners.expand(pos.shape[0], -1, -1))
    tgt = goal.unsqueeze(0).unsqueeze(0) + corners.unsqueeze(0)
    return (cur - tgt).norm(dim=-1).mean(dim=1)


def keypoint_baseline(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Coarse pull toward the slot, pays from tens of centimetres away."""
    return squashing_fn(_keypoint_dist(env, name), 5.0, 4.0)


def keypoint_coarse(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Alignment term, pays over roughly a centimetre."""
    return squashing_fn(_keypoint_dist(env, name), 50.0, 2.0)


def keypoint_fine(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Last-millimetre term. This is the one that makes the task a precision task."""
    return squashing_fn(_keypoint_dist(env, name), 300.0, 0.0)


def reach_block(env: ManagerBasedRLEnv, std: float, ee_frame_cfg: SceneEntityCfg,
                name: str = "block") -> torch.Tensor:
    """Get the gripper to the block at all -- without this nothing ever gets grasped."""
    d = (common.ee_pos_local(env, ee_frame_cfg) - common.object_pos_local(env, name)).norm(dim=1)
    return 1.0 - torch.tanh(d / std)


def block_lifted(env: ManagerBasedRLEnv, minimal_height: float, ee_max_dist: float,
                 ee_frame_cfg: SceneEntityCfg, name: str = "block") -> torch.Tensor:
    """1 while the block is off the table *and* still near the gripper.

    The proximity gate is not optional: without it, batting the block into the air scores
    the same as carrying it (the lift teacher in this repo found exactly that exploit).
    """
    pos = common.object_pos_local(env, name)
    near = (common.ee_pos_local(env, ee_frame_cfg) - pos).norm(dim=1) < ee_max_dist
    return ((pos[:, 2] > minimal_height) & near).float()


def engaged(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Staged bonus: the block's nose is between the walls, even if not fully home."""
    return ((common.insertion_depth(env, name) >= 0.010)
            & (common.lateral_error(env, name) <= 0.030)).float()


class inserted(ManagerTermBase):
    """Sparse per-step success reward, and the place the success metric is logged.

    Per-step rather than terminal on purpose: terminating on success makes a fast sloppy
    insert reward-rate-optimal, which is the failure this repo already hit on the lift
    task. Paying every step the block *stays* inserted makes a stable insert optimal.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        ever = self._ever[env_ids]
        self._env.extras.setdefault("log", {})["Metrics/insert_success_rate"] = ever.float().mean()
        self._ever[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
        ok = common.is_inserted(env, name)
        # in-place: rebinding under torch.inference_mode() turns the buffer into an
        # inference tensor and the next reset()'s write raises
        self._ever.copy_(self._ever | ok)
        return ok.float()
