# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the precision-slot env -- rungs V1, V3, V5, V6 of the validation ladder.

Checks the env builds, the authored slot geometry matches the configured clearance to the
micron (V1), the block settles without interpenetrating (V3), the observation and reward
plumbing is finite and correctly shaped (V6), and -- the part that matters -- that the
success predicate has teeth: it fires when the block is teleported home, and does *not*
fire for a block that is deep enough but laterally offset, yawed, or only part-way in (V5).

Every check appends to ``failures`` instead of asserting, so one run reports everything.

.. code-block:: bash

    python scripts/test_precision_slot_env.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the precision-slot env.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.challenge import mdp

EXPECTED_DIM = 8 + 8 + 7 + 4 + 7  # 34


def zero_step(env, n: int) -> None:
    a = torch.zeros(env.unwrapped.num_envs, 7, device=env.unwrapped.device)
    for _ in range(n):
        env.step(a)


def teleport(env, pos, quat=None, settle: int = 0) -> None:
    """Write the block root pose in env-local coordinates.

    ``settle=0`` refreshes the data buffers without advancing physics, which is what the
    predicate-logic checks need: with ``settle>0`` the slot walls physically shove a
    misaligned block back into alignment before the predicate is ever read, so the test
    would be measuring the geometry rather than the predicate.
    """
    e = env.unwrapped
    n = e.num_envs
    if quat is None:
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=e.device).repeat(n, 1)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).repeat(n, 1)
    state = torch.cat([pos + e.scene.env_origins, quat, torch.zeros((n, 6), device=e.device)], dim=1)
    e.scene["block"].write_root_state_to_sim(state)
    if settle:
        zero_step(env, settle)
    else:
        e.sim.forward()
        e.scene.update(e.physics_dt)


def main() -> None:
    failures: list[str] = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    clearance = env_cfg.clearance

    with torch.inference_mode():
        obs, _ = env.reset()
        pol = obs["policy"]

        # ---- V6: observation plumbing ---------------------------------------
        if pol.shape[1] != EXPECTED_DIM:
            failures.append(f"obs dim {pol.shape[1]} != {EXPECTED_DIM}")
        if not torch.isfinite(pol).all():
            failures.append("observation contains non-finite values")

        # ---- V1: the authored geometry is what the config claims -------------
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        stage = e.sim.stage
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

        def extent(path: str):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return None
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            return [r.GetMax()[i] - r.GetMin()[i] for i in range(3)], [r.GetMin()[i] for i in range(3)], [
                r.GetMax()[i] for i in range(3)]

        py = extent("/World/envs/env_0/SlotWallPY")
        ny = extent("/World/envs/env_0/SlotWallNY")
        blk = extent("/World/envs/env_0/Block")
        if py is None or ny is None or blk is None:
            failures.append("slot walls or block prim missing from the stage")
        else:
            # inner gap between the walls, read back off the stage
            gap = py[1][1] - ny[2][1]
            want = 2 * (mdp.BLOCK_HALF[1] + clearance)
            if abs(gap - want) > 1e-5:
                failures.append(f"authored slot gap {gap * 1000:.3f} mm != configured {want * 1000:.3f} mm")
            block_w = blk[0][1]
            if abs(block_w - 2 * mdp.BLOCK_HALF[1]) > 1e-5:
                failures.append(f"block width {block_w * 1000:.3f} mm != {2 * mdp.BLOCK_HALF[1] * 1000:.3f} mm")
            print(f"[V1] slot gap {gap * 1000:.3f} mm, block {block_w * 1000:.3f} mm, "
                  f"per-side clearance {(gap - block_w) / 2 * 1000:.3f} mm")
            if py[0][2] >= 2 * mdp.BLOCK_HALF[2]:
                failures.append("walls are not shorter than the block -- fingers will hit them")

        # ---- V3: the block settles sanely ------------------------------------
        zero_step(env, 30)
        z = mdp.object_pos_local(e, "block")[:, 2]
        if not ((z > 0.020) & (z < 0.050)).all():
            failures.append(f"block did not settle upright on the table: z in [{z.min():.4f}, {z.max():.4f}]")
        up = mdp.object_quat(e, "block")
        from isaaclab.utils.math import quat_apply

        upz = quat_apply(up, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(up.shape[0], 3))[:, 2]
        if not (upz > 0.95).all():
            failures.append(f"block is not upright after settling: min up-z {upz.min():.3f}")

        # ---- V6: rewards are finite ------------------------------------------
        for fn, kw in [
            (mdp.keypoint_baseline, {}), (mdp.keypoint_coarse, {}), (mdp.keypoint_fine, {}),
            (mdp.engaged, {}),
            (mdp.reach_block, {"std": 0.1, "ee_frame_cfg": mdp.SceneEntityCfg("ee_frame")}
             if hasattr(mdp, "SceneEntityCfg") else {}),
        ]:
            try:
                v = fn(e, **kw) if kw else fn(e)
                if not torch.isfinite(v).all():
                    failures.append(f"reward {fn.__name__} produced non-finite values")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"reward {fn.__name__} raised {exc!r}")

        # ---- V5: the success predicate, positive case ------------------------
        cx, cy = mdp.SLOT_CENTER
        home_x = cx + mdp.SUCCESS_DEPTH - mdp.SLOT_DEPTH / 2
        home_z = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]
        home = torch.tensor([home_x, cy, home_z], device=dev)
        teleport(env, home)
        ok = mdp.is_inserted(e)
        if not ok.all():
            failures.append(f"inserted predicate did not fire at the home pose ({int(ok.sum())}/{ok.numel()})")
        print(f"[V5+] home pose: inserted {int(ok.sum())}/{ok.numel()}, "
              f"depth {float(mdp.insertion_depth(e).mean()) * 1000:.1f} mm")

        # ---- V5: negative controls -------------------------------------------
        # (a) deep enough, but pushed sideways past the wall line
        teleport(env, torch.tensor([home_x, cy + 0.030, home_z], device=dev))
        if mdp.is_inserted(e).any():
            failures.append("negative control (a): predicate fired for a laterally offset block")

        # (b) on the centreline and deep, but yawed well past tolerance
        import math

        yaw = 0.6
        q = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=dev).repeat(e.num_envs, 1)
        teleport(env, home, q)
        if mdp.is_inserted(e).any():
            failures.append(f"negative control (b): predicate fired for a block yawed {yaw:.2f} rad")

        # (b2) and the geometry itself rejects that yaw: let physics run and the walls
        # must shove the block back square (a 45 mm block cannot sit at 0.6 rad in a
        # 33 mm slot). This is the property that makes the task a precision task.
        teleport(env, home, q, settle=25)
        resid = mdp.yaw_error(e)
        print(f"[V5+] a block forced in at {yaw:.2f} rad relaxes to {float(resid.mean()):.3f} rad "
              f"(max {float(resid.max()):.3f}) -- the walls do reject yaw")
        if float(resid.max()) > yaw:
            failures.append("slot geometry did not resist a yawed block at all")

        # (c) squarely aligned but only just entering the mouth
        teleport(env, torch.tensor([cx - mdp.SLOT_DEPTH / 2 + 0.005, cy, home_z], device=dev))
        if mdp.is_inserted(e).any():
            failures.append("negative control (c): predicate fired for a barely-entered block")

        # (d) still on the table where it spawned
        teleport(env, torch.tensor([0.220, -0.130, 0.035], device=dev))
        if mdp.is_inserted(e).any():
            failures.append("negative control (d): predicate fired for a block still on the table")

        # ---- V6: terminations ------------------------------------------------
        # evaluate the term functions directly: stepping would auto-reset the terminated
        # envs and put the block back on the table before it could be observed
        teleport(env, torch.tensor([0.220, -0.130, -0.50], device=dev))
        if not mdp.block_dropped(e, minimum_height=-0.05).all():
            failures.append("block_dropped did not fire for a block below the table")

        lay = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=dev).repeat(e.num_envs, 1)  # 90 deg about x
        teleport(env, torch.tensor([0.220, -0.130, 0.015], device=dev), lay)
        if not mdp.block_toppled(e, max_tilt=0.6).all():
            failures.append("block_toppled did not fire for a block lying on its side")

    print("\n" + "=" * 70)
    if failures:
        print("[result] FAIL")
        for f in failures:
            print("  - " + f)
    else:
        print("[result] PASS -- geometry, physics, observations, rewards and the success")
        print("         predicate (with four negative controls) all check out.")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
