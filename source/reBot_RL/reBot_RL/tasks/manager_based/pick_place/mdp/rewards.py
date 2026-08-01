# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the pick-and-place MDP.

Staging mirrors the proven lift shaping, generalized to two objects and a fixed basket
goal: reach the nearest not-yet-placed object, lift it, carry it over the basket, and
hold placements. There is deliberately no success termination -- the lift teacher taught
us that terminating on success makes flinging reward-rate optimal; here the per-step
``objects_placed`` stream (weighted above the whole carry-shaping stream) makes *stable*
placement the best policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .common import OBJECT_NAMES, basket_centers_local, object_pos_local, placed_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def reach_nearest_unplaced(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh reward for keeping the end-effector near the closest not-yet-placed object.

    While an object is carried it is still "unplaced", so the term stays satisfied during
    transport; once it lands in the basket the target switches to the other object. With
    both placed the reward is maxed out (no incentive to disturb the basket).
    """
    ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    placed = placed_mask(env)
    dists = []
    for i, name in enumerate(OBJECT_NAMES):
        d = torch.linalg.norm(env.scene[name].data.root_pos_w.torch - ee_w, dim=1)
        dists.append(torch.where(placed[:, i], torch.full_like(d, torch.inf), d))
    dmin = torch.stack(dists, dim=1).min(dim=1).values
    reward = 1 - torch.tanh(dmin / std)
    return torch.where(placed.all(dim=1), torch.ones_like(reward), reward)


def unplaced_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_max_dist: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1.0 while any not-yet-placed object is held above ``minimal_height``.

    "Held" = the end-effector is within ``ee_max_dist`` of the object. Without this gate
    the policy learns to swat objects airborne instead of grasping them (observed in run
    2: 15% lift time with near-zero reaching reward and 90-step episodes ending in
    knocked-off-table terminations).
    """
    ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    placed = placed_mask(env)
    lifted = []
    for i, name in enumerate(OBJECT_NAMES):
        pos_w = env.scene[name].data.root_pos_w.torch
        z = pos_w[:, 2] - env.scene.env_origins[:, 2]
        held = torch.linalg.norm(pos_w - ee_w, dim=1) < ee_max_dist
        lifted.append((z > minimal_height) & held & ~placed[:, i])
    return torch.stack(lifted, dim=1).any(dim=1).float()


def unplaced_to_basket(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    ee_max_dist: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh reward for carrying a held, lifted, not-yet-placed object toward the basket.

    Gated on lift height (pushing along the table earns nothing; the walls block pushing
    in anyway) and on end-effector proximity (thrown objects earn nothing). The max over
    objects tracks whichever object is being carried.
    """
    ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    placed = placed_mask(env)
    centers = basket_centers_local(env)
    best = None
    for i, name in enumerate(OBJECT_NAMES):
        pos_w = env.scene[name].data.root_pos_w.torch
        pos = pos_w - env.scene.env_origins
        held = torch.linalg.norm(pos_w - ee_w, dim=1) < ee_max_dist
        gate = (pos[:, 2] > minimal_height) & held & ~placed[:, i]
        d = torch.linalg.norm(pos[:, :2] - centers, dim=1)
        r = gate.float() * (1 - torch.tanh(d / std))
        best = r if best is None else torch.maximum(best, r)
    return best


def held_over_basket(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1.0 while a held, not-yet-placed object hovers over the basket footprint.

    Bridges the gap run 3 got stuck in (hovering 15 cm from the basket, never crossing
    the wall): being *over* the basket now pays before the release does.
    """
    from .common import PLACED_XY_HALF

    ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    placed = placed_mask(env)
    centers = basket_centers_local(env)
    over = []
    for i, name in enumerate(OBJECT_NAMES):
        pos_w = env.scene[name].data.root_pos_w.torch
        pos = pos_w - env.scene.env_origins
        held = torch.linalg.norm(pos_w - ee_w, dim=1) < 0.07
        in_xy = ((pos[:, 0] - centers[:, 0]).abs() < PLACED_XY_HALF) & (
            (pos[:, 1] - centers[:, 1]).abs() < PLACED_XY_HALF
        )
        over.append(held & in_xy & (pos[:, 2] > minimal_height) & ~placed[:, i])
    return torch.stack(over, dim=1).any(dim=1).float()


def falling_into_basket(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """1.0 while an unheld, not-yet-placed object is descending over the basket footprint.

    This is the ~0.1 s window right after a release (or throw) over the basket -- the
    probe on run 4 showed the policy reaches over the basket but opens the gripper there
    in 0.00% of steps; this credits the release at the exact decision step. The state is
    physically transient (an unheld object over the footprint must be falling; resting on
    the wall top is outside the footprint), so it cannot be farmed.
    """
    from .common import PLACED_XY_HALF

    ee_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :]
    placed = placed_mask(env)
    centers = basket_centers_local(env)
    falling = []
    for i, name in enumerate(OBJECT_NAMES):
        pos_w = env.scene[name].data.root_pos_w.torch
        pos = pos_w - env.scene.env_origins
        held = torch.linalg.norm(pos_w - ee_w, dim=1) < 0.07
        in_xy = ((pos[:, 0] - centers[:, 0]).abs() < PLACED_XY_HALF) & (
            (pos[:, 1] - centers[:, 1]).abs() < PLACED_XY_HALF
        )
        falling.append(in_xy & ~held & ~placed[:, i] & (pos[:, 2] > 0.03) & (pos[:, 2] < 0.15))
    return torch.stack(falling, dim=1).any(dim=1).float()


def task_incomplete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 every step until both objects are placed -- constant time pressure."""
    return (~placed_mask(env).all(dim=1)).float()


def gripper_toggle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1.0 on steps where the binary gripper command flips sign (open <-> close).

    A committed episode needs 4 flips (close/open per object); the v4 policy chattered
    ~34 per episode (17.15 close-edges, 15.63 failed grasps -- eval 2026-07-31), which
    the L2 action-rate term barely prices. Penalizing the flip itself makes first-try
    grasps the cheap strategy.
    """
    a = env.action_manager.action[:, 6]
    prev = env.action_manager.prev_action[:, 6]
    return ((a < 0) != (prev < 0)).float()


class objects_placed(ManagerTermBase):
    """Per-step count of objects inside the basket (0, 1 or 2) -- the placement stream.

    Its weight must exceed the summed carry-shaping stream (lift + both transport terms),
    otherwise hovering over the basket without releasing out-earns placing.

    Also tracks the task metrics and logs them on reset, so they show up in tensorboard:
    ``Metrics/success_rate`` (both objects in the basket when the episode ended -- the
    training-time proxy for the >90% completion gate) and ``Metrics/ever_both_placed``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._both_placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._ever_both_placed = torch.zeros_like(self._both_placed)

    def reset(self, env_ids: torch.Tensor):
        log = self._env.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._both_placed[env_ids].float().mean().item()
        log["Metrics/ever_both_placed"] = self._ever_both_placed[env_ids].float().mean().item()
        self._both_placed[env_ids] = False
        self._ever_both_placed[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        placed = placed_mask(env)
        # copy_ (not rebind): under torch.inference_mode() stepping, rebinding would turn
        # the buffer into an inference tensor and the next reset()'s in-place write crashes
        self._both_placed.copy_(placed.all(dim=1))
        self._ever_both_placed |= self._both_placed
        return placed.float().sum(dim=1)
