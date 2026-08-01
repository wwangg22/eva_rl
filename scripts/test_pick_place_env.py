# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the pick-and-place env.

Checks: env builds, observation shapes, objects settle at sane poses (validates YCB
scale/orientation), the in-basket predicate + placed-flag observations fire when an
object is teleported into the basket, the drop termination fires, and all reward terms
are finite. Saves workspace-camera stills to ``--out_dir`` for visual review.

.. code-block:: bash

    python scripts/test_pick_place_env.py --headless

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the pick-and-place env.")
parser.add_argument("--task", type=str, default="Rebot-PickPlace-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out_dir", type=str, default="logs/camera_previews/pick_place_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# the preview camera cannot render without this
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import imageio.v2 as imageio
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG
from reBot_RL.tasks.manager_based.pick_place import mdp

EXPECTED_POLICY_DIM = 8 + 8 + 7 + 7 + 2 + 7  # 39
FLAG_SLICE = slice(30, 32)  # placed flags inside the policy vector


def grab_frame(cam) -> "object":
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[0, ..., :3].detach().cpu().numpy().astype("uint8")


def zero_step(env, n=1):
    obs = None
    for _ in range(n):
        actions = torch.zeros(env.unwrapped.num_envs, 7, device=env.unwrapped.device)
        obs, _, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
    return obs, terminated, truncated


def teleport(env, name, pos_local, quat=(0.0, 0.0, 0.0, 1.0)):  # xyzw identity
    obj = env.unwrapped.scene[name]
    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    pos_w = env.unwrapped.scene.env_origins + torch.tensor(pos_local, device=device)
    quat_t = torch.tensor(quat, device=device).repeat(n, 1)
    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_t], dim=1))
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))


def main():
    failures = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=640, height=360)
    env = gym.make(args_cli.task, cfg=env_cfg)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    cam = env.unwrapped.scene["workspace_cam"]

    with torch.inference_mode():
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        print(f"[shapes] policy obs: {tuple(obs.shape)} (expected (*, {EXPECTED_POLICY_DIM}))")
        if obs.shape[1] != EXPECTED_POLICY_DIM:
            failures.append(f"policy obs dim {obs.shape[1]} != {EXPECTED_POLICY_DIM}")

        # -- settle: objects should come to rest near their spawn at sane heights
        obs, _, _ = zero_step(env, n=25)
        for name in mdp.OBJECT_NAMES:
            pos = mdp.object_pos_local(env.unwrapped, name)
            z = pos[:, 2]
            print(
                f"[settle] {name}: xy env0 = ({pos[0, 0]:.3f}, {pos[0, 1]:.3f}),"
                f" z min/mean/max = {z.min():.4f}/{z.mean():.4f}/{z.max():.4f}"
            )
            if not torch.all((z > 0.004) & (z < 0.08)):
                failures.append(f"{name} settled z out of range [0.004, 0.08]: {z.min():.4f}..{z.max():.4f}")
            # uprightness: the YCB assets are Y-up, spawned with a +90 deg X-rotation, so
            # a standing object's body-Y axis points at world +Z (invariant to yaw spin)
            import isaaclab.utils.math as math_utils

            quat = env.unwrapped.scene[name].data.root_quat_w.torch
            body_y = torch.tensor([0.0, 1.0, 0.0], device=quat.device).expand(quat.shape[0], 3)
            up = math_utils.quat_apply(quat, body_y)[:, 2]
            print(f"[settle] {name}: uprightness (body-Y . world-Z) min = {up.min():.3f}")
            if not torch.all(up > 0.9):
                failures.append(f"{name} not standing upright after settle (min uprightness {up.min():.3f})")
        imageio.imwrite(os.path.join(args_cli.out_dir, "settled.png"), grab_frame(cam))

        # -- measure real object dimensions. NOTE: this reads the spawn-authored USD pose
        # (sim poses are not written back to USD), so only the SIZES are meaningful, not
        # which axis is up. Graspability needs one cross-section dim within the ~0.045 m
        # finger stroke. Uprightness is instead implied by the root quat staying at the
        # spawn rotation through settling (a fallen object's root quat must change).
        from pxr import Usd, UsdGeom

        stage = env.unwrapped.scene.stage
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for prim_name, obj_name in [("ObjectA", "object_a"), ("ObjectB", "object_b")]:
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/{prim_name}")
            size = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
            print(f"[dims] {obj_name} world AABB: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}) m")
            if min(size) > 0.04:
                failures.append(f"{obj_name} smallest dimension {min(size):.3f} m exceeds graspable ~0.04 m")

        # -- object_b must never start placed (object_a may: the prestage event puts it
        # in the basket in ~30% of envs by design)
        placed = mdp.placed_mask(env.unwrapped)
        print(f"[spawn] prestaged object_a in {int(placed[:, 0].sum())}/{placed.shape[0]} envs")
        if placed[:, 1].any():
            failures.append(f"object_b placed at spawn in {int(placed[:, 1].sum())} envs")

        # -- reward terms should be finite and sane at rest
        for fname, kwargs in [
            ("reach_nearest_unplaced", {"std": 0.1}),
            ("unplaced_is_lifted", {"minimal_height": 0.06, "ee_max_dist": 0.07}),
            ("unplaced_to_basket", {"std": 0.3, "minimal_height": 0.06, "ee_max_dist": 0.07}),
            ("held_over_basket", {"minimal_height": 0.05}),
            ("falling_into_basket", {}),
        ]:
            val = getattr(mdp, fname)(env.unwrapped, **kwargs)
            print(f"[reward] {fname}: mean {val.mean():.4f}")
            if not torch.isfinite(val).all():
                failures.append(f"{fname} returned non-finite values")

        # -- teleport object_a into the basket: predicate + obs flags must fire
        cx, cy = mdp.BASKET_CENTER
        teleport(env, "object_a", (cx, cy, 0.03))
        obs, _, _ = zero_step(env, n=3)
        placed = mdp.placed_mask(env.unwrapped)
        print(f"[predicate] object_a placed after teleport-in: {placed[:, 0].float().mean():.2f} of envs")
        if not placed[:, 0].all():
            failures.append("object_a not detected as placed after teleport into basket")
        if placed[:, 1].any():
            failures.append("object_b wrongly detected as placed")
        # flags are canonically ordered (target-first), so compare counts, not order
        flags = obs[:, FLAG_SLICE]
        if not torch.allclose(flags.sum(dim=1), placed.float().sum(dim=1)):
            failures.append(f"obs placed-flag count does not match predicate (obs env0: {flags[0].tolist()})")
        pos_a = mdp.object_pos_local(env.unwrapped, "object_a")
        print(f"[predicate] object_a in-basket z env0: {pos_a[0, 2]:.4f} (floor ~{mdp.BASKET_FLOOR_THICKNESS:.3f})")
        imageio.imwrite(os.path.join(args_cli.out_dir, "object_a_in_basket.png"), grab_frame(cam))

        # -- basket must contain the object (walls collide): it stayed inside during settle
        if not ((pos_a[:, 0] - cx).abs() < mdp.BASKET_INNER_HALF).all():
            failures.append("object_a escaped the basket in x -- wall collision broken?")

        # -- drop termination: teleport object_a below the table
        teleport(env, "object_a", (cx, cy, -0.5))
        _, terminated, _ = zero_step(env, n=1)
        print(f"[termination] object_dropping fired in {terminated.float().mean():.2f} of envs")
        if not terminated.all():
            failures.append("object_dropping termination did not fire for all envs")

    print(f"[stills] wrote settled.png / object_a_in_basket.png to {args_cli.out_dir}")
    if failures:
        print("[result] FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        print("[result] PASS")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
