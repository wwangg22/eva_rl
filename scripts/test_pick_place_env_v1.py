# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate 0 smoke test for the pick-and-place v1 env (randomized basket, 41-D obs).

Programmatic checks: obs is 41-D, basket root poses differ across envs and change across
resets (and match the ``basket_centers_local`` buffer), spawn-sampler geometry (radius /
azimuth / separation / basket clearance), the placed predicate follows the RANDOMIZED
basket (fires at the per-env center, not at the old fixed v0 location), objects settle
upright/lying per the 0.25 probability, per-env object scales differ (prestartup USD
randomization), nudges are disabled by default, and no NaNs anywhere. Saves
workspace-camera stills to ``--out_dir`` for visual review.

.. code-block:: bash

    python scripts/test_pick_place_env_v1.py --headless

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gate 0 smoke test for the pick-and-place v1 env.")
parser.add_argument("--task", type=str, default="Rebot-PickPlace-v1")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=25, help="settle steps after each reset (bounds the run)")
parser.add_argument("--out_dir", type=str, default="logs/camera_previews/pick_place_v1_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# the preview camera cannot render without this
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import imageio.v2 as imageio
import torch

import isaaclab.utils.math as math_utils

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG
from reBot_RL.tasks.manager_based.pick_place import mdp

EXPECTED_POLICY_DIM = 8 + 8 + 16 + 2 + 7  # 41 (v0's 39 + basket center xy)
_STAND_UP_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)  # XYZW, Y-up can stood upright


def grab_frame(cam) -> "object":
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[0, ..., :3].detach().cpu().numpy().astype("uint8")


def zero_step(env, n=1):
    obs = reward = terminated = truncated = None
    for _ in range(n):
        actions = torch.zeros(env.unwrapped.num_envs, 7, device=env.unwrapped.device)
        obs, reward, terminated, truncated, _ = env.step(actions)
        obs = obs["policy"]
    return obs, reward, terminated, truncated


def teleport(env, name, pos_local, quat=_STAND_UP_QUAT):
    """Teleport an object; ``pos_local`` is (3,) or a per-env (N, 3) env-local tensor."""
    obj = env.unwrapped.scene[name]
    n = env.unwrapped.num_envs
    device = env.unwrapped.device
    pos_local_t = torch.as_tensor(pos_local, dtype=torch.float, device=device)
    if pos_local_t.dim() == 1:
        pos_local_t = pos_local_t.expand(n, 3)
    pos_w = env.unwrapped.scene.env_origins + pos_local_t
    quat_t = torch.tensor(quat, device=device).repeat(n, 1)
    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_t], dim=1))
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=device))


def basket_xy(env) -> torch.Tensor:
    """Env-local basket root xy straight from sim (not the buffer)."""
    return (env.unwrapped.scene["basket"].data.root_pos_w.torch - env.unwrapped.scene.env_origins)[:, :2]


def uprightness(env, name) -> torch.Tensor:
    """Body-Y . world-Z: ~1 standing (Y-up asset stood up), ~0 lying on the side."""
    quat = env.unwrapped.scene[name].data.root_quat_w.torch
    body_y = torch.tensor([0.0, 1.0, 0.0], device=quat.device).expand(quat.shape[0], 3)
    return math_utils.quat_apply(quat, body_y)[:, 2]


def main():
    failures = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=640, height=360)
    env = gym.make(args_cli.task, cfg=env_cfg)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    cam = env.unwrapped.scene["workspace_cam"]
    n = env.unwrapped.num_envs

    # -- nudges must be off in the nominal v1 cfg
    print(f"[cfg] nudge_object_a is {env_cfg.events.nudge_object_a} (expected None: disabled by default)")
    if env_cfg.events.nudge_object_a is not None or env_cfg.events.nudge_object_b is not None:
        failures.append("interval nudge events are enabled in the nominal v1 cfg")

    with torch.inference_mode():
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        print(f"[shapes] policy obs: {tuple(obs.shape)} (expected (*, {EXPECTED_POLICY_DIM}))")
        if obs.shape[1] != EXPECTED_POLICY_DIM:
            failures.append(f"policy obs dim {obs.shape[1]} != {EXPECTED_POLICY_DIM}")
        if not torch.isfinite(obs).all():
            failures.append("NaN/inf in policy obs after first reset")

        # -- basket randomization: poses differ across envs and match the buffer
        centers_sim = basket_xy(env)
        centers_buf = mdp.basket_centers_local(env.unwrapped)
        distinct = len(torch.unique((centers_sim * 1000).round(), dim=0))
        print(f"[basket] distinct centers: {distinct}/{n}, xy std = ({centers_sim.std(0)[0]:.4f}, {centers_sim.std(0)[1]:.4f})")
        if distinct < 8:
            failures.append(f"basket centers only take {distinct} distinct values across {n} envs")
        if not torch.allclose(centers_sim, centers_buf, atol=1e-4):
            failures.append("basket root pose in sim does not match the basket_centers_local buffer")
        # obs slice 32:34 must be the buffer (basket term sits after objects_canonical)
        if not torch.allclose(obs[:, 32:34], centers_buf, atol=1e-5):
            failures.append(f"obs[:, 32:34] != basket center buffer (env0 obs {obs[0, 32:34].tolist()})")

        # -- sampler geometry right after reset (before physics moves anything)
        r_b = torch.linalg.norm(centers_sim, dim=1)
        az_b = torch.atan2(centers_sim[:, 1], centers_sim[:, 0]).abs()
        print(f"[basket] r min/max = {r_b.min():.3f}/{r_b.max():.3f}, |azimuth| max = {math.degrees(az_b.max()):.1f} deg")
        if not torch.all((r_b > 0.199) & (r_b < 0.271)):
            failures.append(f"basket radius outside [0.20, 0.27]: {r_b.min():.3f}..{r_b.max():.3f}")
        if not torch.all(az_b < math.radians(50.1)):
            failures.append(f"basket azimuth outside +/-50 deg: {math.degrees(az_b.max()):.1f}")
        obj_xy = torch.stack([mdp.object_pos_local(env.unwrapped, name)[:, :2] for name in mdp.OBJECT_NAMES], dim=1)
        r_o = torch.linalg.norm(obj_xy, dim=2)
        az_o = torch.atan2(obj_xy[..., 1], obj_xy[..., 0]).abs()
        sep = torch.linalg.norm(obj_xy[:, 0] - obj_xy[:, 1], dim=1)
        clear = torch.linalg.norm(obj_xy - centers_sim.unsqueeze(1), dim=2)
        print(
            f"[spawn] object r min/max = {r_o.min():.3f}/{r_o.max():.3f}, |azimuth| max = "
            f"{math.degrees(az_o.max()):.1f} deg, separation min = {sep.min():.3f}, basket clearance min = {clear.min():.3f}"
        )
        if not torch.all((r_o > 0.199) & (r_o < 0.331)):
            failures.append(f"object radius outside [0.20, 0.33]: {r_o.min():.3f}..{r_o.max():.3f}")
        if not torch.all(az_o < math.radians(45.1)):
            failures.append(f"object azimuth outside +/-45 deg: {math.degrees(az_o.max()):.1f}")
        if not torch.all(sep > 0.0599):
            failures.append(f"object pairwise separation below 0.06: {sep.min():.4f}")
        if not torch.all(clear > 0.1399):
            failures.append(f"basket-object clearance below 0.14: {clear.min():.4f}")

        # -- basket centers must change across resets
        centers_first = centers_sim.clone()
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        centers_second = basket_xy(env)
        moved = (torch.linalg.norm(centers_second - centers_first, dim=1) > 0.01).float().mean()
        print(f"[basket] centers moved >1 cm after 2nd reset in {moved:.2f} of envs")
        if moved < 0.8:
            failures.append(f"basket centers only moved in {moved:.2f} of envs across resets")
        if not torch.allclose(obs[:, 32:34], mdp.basket_centers_local(env.unwrapped), atol=1e-5):
            failures.append("obs basket slice stale after 2nd reset")

        # -- settle: objects at sane heights, upright/lying split per the 0.25 prob
        obs, reward, _, _ = zero_step(env, n=args_cli.steps)
        if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
            failures.append("NaN/inf in obs or reward after settle")
        ups = []
        for name in mdp.OBJECT_NAMES:
            pos = mdp.object_pos_local(env.unwrapped, name)
            z = pos[:, 2]
            up = uprightness(env, name)
            ups.append(up)
            print(f"[settle] {name}: z min/mean/max = {z.min():.4f}/{z.mean():.4f}/{z.max():.4f}, up min/max = {up.min():.2f}/{up.max():.2f}")
            if not torch.all((z > 0.004) & (z < 0.08)):
                failures.append(f"{name} settled z out of range [0.004, 0.08]: {z.min():.4f}..{z.max():.4f}")
        up_all = torch.cat(ups)
        upright = up_all > 0.9
        lying = up_all.abs() < 0.5
        frac_lying = lying.float().mean().item()
        print(f"[settle] lying fraction = {frac_lying:.2f} ({int(lying.sum())}/{len(up_all)}), expected ~0.25")
        if not torch.all(upright | lying):
            failures.append("some objects settled neither upright nor lying (uprightness in [0.5, 0.9])")
        if not (0.05 <= frac_lying <= 0.50):
            failures.append(f"lying fraction {frac_lying:.2f} outside [0.05, 0.50] (prob 0.25)")
        imageio.imwrite(os.path.join(args_cli.out_dir, "settled.png"), grab_frame(cam))

        # -- per-env object scale randomization (prestartup USD event)
        stage = env.unwrapped.scene.stage
        scales = []
        for i in range(n):
            attr = stage.GetPrimAtPath(f"/World/envs/env_{i}/ObjectA").GetAttribute("xformOp:scale")
            scales.append(float(attr.Get()[0]))
        scales_t = torch.tensor(scales)
        distinct_scales = len(torch.unique((scales_t * 1000).round()))
        print(
            f"[scale] object_a scale min/max = {scales_t.min():.4f}/{scales_t.max():.4f}, "
            f"distinct = {distinct_scales}/{n} (authored range 0.28..0.4375)"
        )
        if distinct_scales < max(4, n // 4):
            failures.append(f"object scales barely differ across envs ({distinct_scales} distinct)")
        if not ((scales_t.min() > 0.279) and (scales_t.max() < 0.438)):
            failures.append(f"object scales outside [0.28, 0.4375]: {scales_t.min():.4f}..{scales_t.max():.4f}")

        # -- placed predicate must follow the RANDOMIZED basket
        centers = mdp.basket_centers_local(env.unwrapped)
        target = torch.cat([centers, torch.full((n, 1), 0.03, device=centers.device)], dim=1)
        teleport(env, "object_a", target)
        obs, _, _, _ = zero_step(env, n=3)
        placed = mdp.placed_mask(env.unwrapped)
        print(f"[predicate] object_a placed after teleport to per-env basket: {placed[:, 0].float().mean():.2f} of envs")
        if not placed[:, 0].all():
            failures.append("object_a not detected as placed after teleport into the randomized basket")
        flags = obs[:, 30:32]  # canonical placed flags (unchanged position inside objects term)
        if not torch.allclose(flags.sum(dim=1), placed.float().sum(dim=1)):
            failures.append(f"obs placed-flag count does not match predicate (env0 flags {flags[0].tolist()})")
        imageio.imwrite(os.path.join(args_cli.out_dir, "object_a_in_basket.png"), grab_frame(cam))

        # -- ... and must NOT fire at the OLD fixed v0 basket location (except in envs
        # whose randomized basket happens to sit there). Envs within a 4 cm ambiguity
        # ring of the boundary are skipped (the can may clip a basket wall and wander).
        old = torch.tensor(mdp.BASKET_CENTER, device=centers.device)
        d_old = (centers - old).abs()
        inside = (d_old < mdp.PLACED_XY_HALF - 0.01).all(dim=1)
        outside = (d_old > mdp.PLACED_XY_HALF + 0.04).any(dim=1)
        teleport(env, "object_a", (old[0].item(), old[1].item(), 0.03))
        obs, _, _, _ = zero_step(env, n=3)
        placed = mdp.placed_mask(env.unwrapped)
        print(
            f"[predicate] teleport to OLD fixed center: placed in {int(placed[:, 0].sum())}/{n} envs "
            f"(basket actually there in {int(inside.sum())}, clearly elsewhere in {int(outside.sum())})"
        )
        if placed[outside, 0].any():
            failures.append(f"placed fired at the OLD fixed location in {int(placed[outside, 0].sum())} envs whose basket is elsewhere")
        if inside.any() and not placed[inside, 0].all():
            failures.append("placed did not fire in envs whose randomized basket covers the old location")
        if not torch.isfinite(obs).all():
            failures.append("NaN/inf in obs after teleports")
        imageio.imwrite(os.path.join(args_cli.out_dir, "object_a_at_old_center.png"), grab_frame(cam))

    print(f"[stills] wrote settled.png / object_a_in_basket.png / object_a_at_old_center.png to {args_cli.out_dir}")
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
