# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the pre-grasp (extrinsic dexterity) task.

The block spawns lying flat, in a pose where **no grasp exists**, and the whole task rests
on that being literally true rather than merely plausible. Two measured numbers make it so
(``scripts/analysis/gripper_stroke.py``, ``scripts/challenge/pregrasp_probe.py``):

* the gripper's clear opening is **89.1 mm** -- ``gap = 1.0035 * (q_left + q_right) - 1.25``
  mm, fitted on 30 / 45 / 70 mm gauges with a 0.035 mm residual;
* lying, the block's horizontal shadow is **100 mm** across at its narrowest, so the fingers
  cannot span it from any heading. Tipped up on edge it presents **60 mm**.

And the arm cannot come at it from above either -- 0.00 % of table-height voxels admit a
finger axis within 26 deg of vertical (docs/CHALLENGE_SUITE.md C1). So the only route is to
push the block against the back wall until it pivots up onto its edge, then grasp it.

This is the setup from "Learning to Grasp the Ungraspable with Emergent Extrinsic
Dexterity" (CoRL 2022), which is verified to work with a plain parallel-jaw gripper and no
tactile or force sensing.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from . import common

#: block half-extents (x, y, z) [m] as authored, i.e. standing on edge: 100 x 60 x 100 mm.
#: y is the graspable thickness; the block is spawned rotated 90 deg about x, which lays it
#: flat and presents 100 x 100 mm to the gripper, standing 60 mm tall.
PG_BLOCK_HALF = (0.050, 0.030, 0.050)
#: back wall inner face, x [m]
WALL_X = 0.320
#: measured clear opening of the fully-open gripper [m] -- see scripts/analysis/gripper_stroke.py
GRIPPER_OPENING = 0.0891
#: narrowest horizontal width the block presents when lying / when up on edge [m]
W_LYING = 2 * PG_BLOCK_HALF[0]   # 0.100 -- 10.9 mm wider than the gripper can open
W_UP = 2 * PG_BLOCK_HALF[1]      # 0.060
#: below this the block is comfortably graspable; used as the "reoriented" test
W_GRASPABLE = 0.080
#: lifted clear of the table [m]. Standing on edge the block's centre is at 0.050.
LIFT_Z = 0.075
#: a block is "up" when its own long (+z) axis is within ~25 deg of world +z
UPRIGHT_DOT = 0.90

#: directions sampled when minimising the shadow width, on top of the exact kink directions
_N_DIRS = 128


def block_up_axis(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """World-z component of the block's own +z (long) axis. 1 = standing, 0 = lying."""
    q = common.object_quat(env, name)
    up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3))
    return up[:, 2]


def min_grasp_width(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Narrowest width the block presents to a horizontally-closing gripper [m].

    Formally the minimum width of the block's shadow on the horizontal plane. For a box
    with half-extents ``h`` and rotation ``R``, the width seen by fingers closing along a
    horizontal unit direction ``d`` is ``sum_i 2 h_i |R_i . d|``; this minimises that over
    ``d``.

    The obvious-looking alternative -- project each body axis onto the plane and take the
    smallest -- is **wrong**, and was the first implementation here. For a block lying flat
    the vertical body axis projects to ~0, so it reports a ~0 mm width exactly when the
    block is at its most ungraspable.
    """
    q = common.object_quat(env, name)
    n = q.shape[0]
    dev = q.device
    h = torch.tensor(PG_BLOCK_HALF, device=dev)

    # world directions of the three body axes, (n, 3, 3)
    eye = torch.eye(3, device=dev)
    axes = quat_apply(q.unsqueeze(1).expand(-1, 3, -1), eye.expand(n, -1, -1))

    # A uniform sweep alone is not accurate enough: with 64 directions a lying 115 x 105 mm
    # block reported 107.8 mm instead of 105.0, because the true minimum sits between two
    # samples. The minima of sum_i c_i |u_i . d| lie at the kinks, where one term vanishes,
    # so the per-env directions perpendicular to each axis's horizontal projection are
    # evaluated exactly, alongside a fine uniform sweep as a guard.
    t = torch.linspace(0.0, torch.pi, _N_DIRS + 1, device=dev)[:-1]  # d and -d agree
    d_uni = torch.stack([t.cos(), t.sin(), torch.zeros_like(t)], dim=-1)  # (D, 3)
    d_uni = d_uni.unsqueeze(0).expand(n, -1, -1)

    ax = axes[..., :2]                                   # horizontal projections, (n, 3, 2)
    perp = torch.stack([-ax[..., 1], ax[..., 0]], dim=-1)
    nrm = perp.norm(dim=-1, keepdim=True)
    # A body axis pointing straight up has NO horizontal projection, so its kink direction
    # is undefined -- and naively normalising the zero vector leaves a zero "direction"
    # whose width evaluates to 0 mm. That made an axis-aligned upright block report a 0 mm
    # graspable width, i.e. maximally graspable, which is as wrong as it gets. Those cases
    # fall back to +x; they are redundant with the uniform sweep anyway.
    fallback = torch.tensor([1.0, 0.0], device=dev).expand_as(perp)
    perp = torch.where(nrm > 1e-6, perp / nrm.clamp(min=1e-9), fallback)
    d_kink = torch.cat([perp, torch.zeros_like(perp[..., :1])], dim=-1)  # (n, 3, 3)

    d = torch.cat([d_uni, d_kink], dim=1)                # (n, D + 3, 3)
    # width for every direction: sum over body axes of 2 h_i |R_i . d|
    proj = torch.einsum("nai,ndi->nda", axes, d).abs()   # (n, D + 3, 3)
    widths = 2.0 * (proj * h).sum(dim=-1)
    return widths.min(dim=1).values


def is_reoriented(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """The block now presents a graspable width -- the precondition the policy must create."""
    return min_grasp_width(env, name) < W_GRASPABLE


def is_upright(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    return block_up_axis(env, name) > UPRIGHT_DOT


def uprighting_progress(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """Dense credit for tipping the block up, before any grasp is possible.

    Measured as how far the presented width has fallen from lying (100 mm) toward on-edge
    (60 mm), so it is orientation-agnostic: tipping about either horizontal axis pays. That
    matters because the block can legitimately come up on either of two edges.

    Without this the task has no gradient at all: the block starts ungraspable, so every
    reach-and-close reward the pick-place task relies on is flat until the reorientation has
    already happened.
    """
    w = min_grasp_width(env, name)
    return ((W_LYING - w) / (W_LYING - W_UP)).clamp(0.0, 1.0)


def is_lifted_upright(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg,
                      name: str = "block") -> torch.Tensor:
    """The task: block reoriented AND picked up AND still in the gripper.

    No separate "is it up" test is needed beyond the width: the block cannot be held at all
    unless it was reoriented first, so reoriented-and-lifted *is* the task.
    """
    pos = common.object_pos_local(env, name)
    near = (common.ee_pos_local(env, ee_frame_cfg) - pos).norm(dim=1) < 0.08
    return is_reoriented(env, name) & (pos[:, 2] > LIFT_Z) & near


# -- observation wrappers ---------------------------------------------------
# The reward/termination helpers above return (N,); the ObservationManager concatenates
# on the last axis and needs (N, 1), so the observation forms are separate.


def obs_block_up_axis(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    return block_up_axis(env, name).unsqueeze(-1)


def obs_min_grasp_width(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    return min_grasp_width(env, name).unsqueeze(-1)


class pregrasp_success(ManagerTermBase):
    """Sparse per-step success, and where the success metric is logged."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/pregrasp_success_rate"] = (
            self._ever[env_ids].float().mean()
        )
        self._ever[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg,
                 name: str = "block") -> torch.Tensor:
        ok = is_lifted_upright(env, ee_frame_cfg, name)
        self._ever.copy_(self._ever | ok)
        return ok.float()
