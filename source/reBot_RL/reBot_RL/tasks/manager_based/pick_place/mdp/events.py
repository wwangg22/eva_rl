# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the pick-and-place MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .common import OBJECT_NAMES, PLACED_XY_HALF, basket_centers_local


def nudge_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    dist_range: tuple[float, float] = (0.01, 0.03),
    max_height: float = 0.03,
    max_radius: float = 0.33,
):
    """Displace an on-table, unplaced object by a small random planar offset (interval mode).

    Robustness perturbation: the policy must re-align to a moved target instead of
    replaying a memorized approach. Objects that are lifted/held (z above ``max_height``)
    or already inside the basket are left alone -- yanking a gripped or placed object
    would be unrecoverable, not a correction exercise.
    """
    obj = env.scene[asset_cfg.name]
    pos = obj.data.root_pos_w.torch[env_ids] - env.scene.env_origins[env_ids]
    centers = basket_centers_local(env)[env_ids]
    in_basket = ((pos[:, 0] - centers[:, 0]).abs() < PLACED_XY_HALF) & (
        (pos[:, 1] - centers[:, 1]).abs() < PLACED_XY_HALF
    )
    chosen = env_ids[(pos[:, 2] < max_height) & ~in_basket]
    if len(chosen) == 0:
        return
    pose = obj.data.root_pose_w.torch[chosen].clone()
    dist = dist_range[0] + torch.rand(len(chosen), device=env.device) * (dist_range[1] - dist_range[0])
    ang = torch.rand(len(chosen), device=env.device) * (2 * torch.pi)
    pose[:, 0] += dist * torch.cos(ang)
    pose[:, 1] += dist * torch.sin(ang)
    # keep the nudged object reachable: cumulative nudges on a waiting object (2-3 per
    # episode while the other is fetched) can otherwise drift it past the workspace
    # radius (~0.34), creating unwinnable episodes (v6c diag: 24/62 failures ended with
    # an upright, unplaced can A)
    local_xy = pose[:, :2] - env.scene.env_origins[chosen, :2]
    r = torch.linalg.norm(local_xy, dim=1)
    scale = (max_radius / r).clamp(max=1.0)
    pose[:, :2] = env.scene.env_origins[chosen, :2] + local_xy * scale.unsqueeze(1)
    obj.write_root_pose_to_sim(pose, env_ids=chosen)


def spawn_lying(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prob: float,
    z: float = 0.015,
):
    """With probability ``prob`` per env, tip the object onto its side at reset.

    Lying-grasp practice (reverse-curriculum, like prestage_in_basket): 25/62 failed
    episodes in the v6c diagnosis ended with can A knocked flat -- a state the policy
    never trained to grasp from. Keeps the reset xy; orientation = random yaw composed
    with a 90-degree roll (quats XYZW). Must be defined AFTER the plain reset events and
    BEFORE prestage terms (prestage overrides lying, never the reverse).
    """
    obj = env.scene[asset_cfg.name]
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(chosen) == 0:
        return
    pose = obj.data.root_pose_w.torch[chosen].clone()
    pose[:, 2] = env.scene.env_origins[chosen, 2] + z
    half_yaw = torch.rand(len(chosen), device=env.device) * torch.pi
    c = 0.7071068
    # XYZW of Rz(yaw) * Rx(90 deg)
    pose[:, 3] = c * torch.cos(half_yaw)
    pose[:, 4] = c * torch.sin(half_yaw)
    pose[:, 5] = c * torch.sin(half_yaw)
    pose[:, 6] = c * torch.cos(half_yaw)
    obj.write_root_pose_to_sim(pose, env_ids=chosen)
    vel = obj.data.root_state_w.torch[chosen, 7:].clone()
    vel[:] = 0.0
    obj.write_root_velocity_to_sim(vel, env_ids=chosen)


def prestage_in_basket(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    prob: float,
    pos: tuple[float, float, float],
    xy_jitter: float = 0.015,
):
    """With probability ``prob`` per env, start the episode with the object already in the basket.

    Reverse-curriculum-lite: the critic sees the placed-state return stream from step 0
    (runs 3/4 never released because that value was unknown), and episodes where the
    *other* object must be fetched get practiced. Must be defined AFTER the plain
    reset_root_state events so it overrides them.

    ``pos`` is the env-local target (inside the basket, at the object's rest height).
    """
    obj = env.scene[asset_cfg.name]
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(chosen) == 0:
        return
    root_state = obj.data.default_root_state[chosen].clone()
    target = torch.tensor(pos, device=env.device).expand(len(chosen), 3).clone()
    target[:, :2] += (torch.rand(len(chosen), 2, device=env.device) * 2 - 1) * xy_jitter
    root_state[:, :3] = target + env.scene.env_origins[chosen]
    root_state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(root_state[:, :7], env_ids=chosen)
    obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids=chosen)


def reset_objects_wide(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float] = (0.20, 0.32),
    azimuth_range: tuple[float, float] = (-0.7853982, 0.7853982),
    min_separation: float = 0.06,
    min_basket_clearance: float = 0.14,
    max_radius: float = 0.33,
    lying_prob: float = 0.25,
    upright_z: float = 0.020,
    lying_z: float = 0.025,
    max_tries: int = 50,
):
    """v1 spawn sampler: both cans in an annulus sector around the robot base.

    Replaces the v0 per-object box regions. Radius/azimuth are sampled uniformly in the
    env-local frame (0 deg = +x, the robot's forward); radii are hard-clamped to
    ``max_radius``. Each can is rejection-resampled (sequentially, can A then can B, so
    an accepted can is never thrown away) until it is at least ``min_basket_clearance``
    from the basket center and ``min_separation`` from the other can. Must run AFTER
    ``reset_basket`` (cfg definition order): the basket sector is small enough that two
    object-clearance discs can cover it entirely, so sampling objects first can leave
    the basket no valid spot -- the object sector always keeps free corners instead.
    The basket center comes from the ``basket_centers_local`` buffer, NOT from sim data
    (``data.root_pos_w`` is lazily refreshed and still holds PRE-reset poses here).

    Each can independently spawns lying on its side with probability ``lying_prob``.
    The cans are authored Y-up (identity rotation = lying on the side), so upright is
    the +90 deg X stand-up rotation and lying is identity roll with a random yaw
    (quats XYZW).
    """
    n = len(env_ids)
    device = env.device
    centers = basket_centers_local(env)[env_ids]
    xy = torch.zeros(n, len(OBJECT_NAMES), 2, device=device)
    for i in range(len(OBJECT_NAMES)):
        pending = torch.ones(n, dtype=torch.bool, device=device)
        for _ in range(max_tries):
            if not pending.any():
                break
            idx = pending.nonzero(as_tuple=False).squeeze(1)
            m = len(idx)
            r = radius_range[0] + torch.rand(m, device=device) * (radius_range[1] - radius_range[0])
            az = azimuth_range[0] + torch.rand(m, device=device) * (azimuth_range[1] - azimuth_range[0])
            cand = torch.stack([r * torch.cos(az), r * torch.sin(az)], dim=1)
            xy[idx, i] = cand
            ok = torch.linalg.norm(cand - centers[idx], dim=1) >= min_basket_clearance
            for j in range(i):
                ok &= torch.linalg.norm(cand - xy[idx, j], dim=1) >= min_separation
            pending[idx] = ~ok
    # hard workspace clamp (graspable-envelope ceiling)
    r = torch.linalg.norm(xy, dim=2)
    xy = xy * (max_radius / r).clamp(max=1.0).unsqueeze(2)

    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = env.scene.env_origins[env_ids]
        pose[:, :2] += xy[:, i]
        lying = torch.rand(n, device=device) < lying_prob
        pose[:, 2] += torch.where(
            lying, torch.full((n,), lying_z, device=device), torch.full((n,), upright_z, device=device)
        )
        # Y-up asset: upright = +90 deg about X; lying = identity roll + random yaw
        # (Rz(yaw) alone -- adding Rx(90) on top would stand the can back up), XYZW
        half_yaw = torch.rand(n, device=device) * torch.pi
        c = 0.7071068
        pose[:, 3] = torch.where(lying, torch.zeros(n, device=device), torch.full((n,), c, device=device))
        pose[:, 5] = torch.where(lying, torch.sin(half_yaw), torch.zeros(n, device=device))
        pose[:, 6] = torch.where(lying, torch.cos(half_yaw), torch.full((n,), c, device=device))
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def reset_basket(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    radius_range: tuple[float, float] = (0.20, 0.27),
    azimuth_range: tuple[float, float] = (-0.8726646, 0.8726646),
):
    """v1 basket randomization: sample the kinematic basket's center per env per episode.

    The center is sampled env-locally in radius/azimuth (0 deg = +x; the range spans
    BOTH sides of the workspace, unlike the fixed v0 basket on -y), unconstrained. It
    must be defined BEFORE ``reset_objects_wide``, which keeps the cans clear of the
    center sampled here (basket-first is the feasible order -- see that docstring).
    Writes the root pose to sim (identity rotation, XYZW) and stores the env-local xy in
    the per-env buffer behind ``mdp.basket_centers_local``, which the placed predicate,
    rewards, nudges, the object sampler and the ``basket_center_xy`` observation read.
    """
    basket = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device
    r = radius_range[0] + torch.rand(n, device=device) * (radius_range[1] - radius_range[0])
    az = azimuth_range[0] + torch.rand(n, device=device) * (azimuth_range[1] - azimuth_range[0])
    centers = torch.stack([r * torch.cos(az), r * torch.sin(az)], dim=1)

    pose = torch.zeros(n, 7, device=device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, :2] += centers
    pose[:, 6] = 1.0  # identity quaternion, XYZW
    basket.write_root_pose_to_sim(pose, env_ids=env_ids)
    basket.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
    basket_centers_local(env)[env_ids] = centers
