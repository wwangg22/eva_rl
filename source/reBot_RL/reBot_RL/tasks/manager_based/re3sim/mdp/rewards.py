# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the reconstructed-workstation task.

Shaped as a staircase -- reach, lift, carry, place -- because the terminal event (a 56 mm cube
at rest inside a box 250 mm away) is far too sparse to find by exploration. Each rung only
pays once the one below it is satisfied, so the agent cannot collect carry reward for an
object it is not holding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .common import (
    CLUTTER_NAMES,
    TARGET_NAME,
    box_centers_local,
    object_pos_local,
    placed_mask,
)


def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w.torch[..., 0, :] - env.scene.env_origins


def reach_target(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """1 - tanh(d / std) between the TCP and the cube. Zero once the cube is placed."""
    d = torch.linalg.norm(_ee_pos(env, ee_frame_cfg) - object_pos_local(env, TARGET_NAME), dim=1)
    return (1.0 - torch.tanh(d / std)) * (~placed_mask(env)).float()


def target_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.075,
    ee_max_dist: float = 0.08,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The cube is off the desk **and** still near the TCP.

    The ``ee_max_dist`` half is not optional: without it a cube batted across the desk and
    bouncing scores lift reward, which is how a policy learns to swat instead of grasp.
    """
    pos = object_pos_local(env, TARGET_NAME)
    near = torch.linalg.norm(_ee_pos(env, ee_frame_cfg) - pos, dim=1) < ee_max_dist
    return ((pos[:, 2] > minimal_height) & near).float()


def target_to_box(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    minimal_height: float = 0.075,
    ee_max_dist: float = 0.08,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Carry: pays for closing the horizontal gap to the box, but only while lifted."""
    pos = object_pos_local(env, TARGET_NAME)
    d = torch.linalg.norm(pos[:, :2] - box_centers_local(env), dim=1)
    return (1.0 - torch.tanh(d / std)) * target_lifted(env, minimal_height, ee_max_dist, ee_frame_cfg)


def place_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The task: cube at rest inside the box, against the strengthened predicate."""
    return placed_mask(env).float()


def clutter_disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Total planar displacement of the clutter from where it spawned [m].

    Used with a small negative weight. It is a *nudge* toward tidy behaviour, not the
    constraint -- the strict variant is what actually forbids disturbance, and it does so with
    a termination. Keeping the shaping term and the constraint separate is deliberate:
    ``clutter``'s ``distractors_disturbed`` was wired only to a shaping reward and nothing
    else, and a 73.3 % expert turned out to be 16.4 % once displacement was really tested.
    """
    buf = getattr(env, "_clutter_spawn_xy", None)
    if buf is None:
        return torch.zeros(env.num_envs, device=env.device)
    total = torch.zeros(env.num_envs, device=env.device)
    for i, name in enumerate(CLUTTER_NAMES):
        total = total + torch.linalg.norm(object_pos_local(env, name)[:, :2] - buf[:, i], dim=1)
    return total
