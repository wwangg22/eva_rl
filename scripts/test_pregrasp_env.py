# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the pre-grasp env.

Checks the block spawns lying and settles there, that ``min_grasp_width`` reports the
block's true horizontal shadow in both poses (it has been wrong twice, in opposite
directions), that the uprighting shaping term spans 0..1 between them, and that the success
predicate needs all three of reoriented, lifted and held.

It also reports the env's **known unresolved issue** rather than quietly passing over it:
the design premise is that a block lying flat is too wide for the gripper to span, and
``scripts/challenge/pregrasp_probe.py`` disproved that. See the banner at the end.

Every check appends to ``failures`` instead of asserting, so one run reports everything.

.. code-block:: bash

    python scripts/test_pregrasp_env.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the pre-grasp env.")
parser.add_argument("--task", type=str, default="Rebot-PreGrasp-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.challenge import mdp

EXPECTED_DIM = 8 + 8 + 7 + 1 + 1 + 7  # 32
LYING = [0.7071, 0.0, 0.0, 0.7071]
UPRIGHT = [0.0, 0.0, 0.0, 1.0]
EE = SceneEntityCfg("ee_frame")
#: the fingers can be forced apart to the sum of their joint limits, not just to the value
#: the binary open command asks for -- measured in scripts/analysis/gripper_stroke.py
FORCED_OPEN = 0.050 + 0.0715


def main() -> None:
    failures: list[str] = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    n = e.num_envs
    H = mdp.PG_BLOCK_HALF

    def put(pos, quat, settle: int = 0):
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).repeat(n, 1)
        e.scene["block"].write_root_state_to_sim(
            torch.cat([pos + e.scene.env_origins, quat.repeat(n, 1),
                       torch.zeros((n, 6), device=dev)], dim=1))
        if settle:
            a = torch.zeros(n, 7, device=dev)
            for _ in range(settle):
                env.step(a)
        else:
            e.sim.forward()
            e.scene.update(e.physics_dt)

    lying_q = torch.tensor(LYING, device=dev).unsqueeze(0)
    up_q = torch.tensor(UPRIGHT, device=dev).unsqueeze(0)

    with torch.inference_mode():
        obs, _ = env.reset()
        pol = obs["policy"]

        # ---- V6: observation plumbing ---------------------------------------
        if pol.shape[1] != EXPECTED_DIM:
            failures.append(f"obs dim {pol.shape[1]} != {EXPECTED_DIM}")
        if not torch.isfinite(pol).all():
            failures.append("observation contains non-finite values")
        for fn in (mdp.obs_min_grasp_width, mdp.obs_block_up_axis):
            if fn(e).shape != (n, 1):
                failures.append(f"{fn.__name__} returns {tuple(fn(e).shape)}, want {(n, 1)}")

        # ---- V3: the block settles lying flat --------------------------------
        a = torch.zeros(n, 7, device=dev)
        for _ in range(40):
            env.step(a)
        z = mdp.object_pos_local(e, "block")[:, 2]
        if not ((z > H[1] - 0.006) & (z < H[1] + 0.006)).all():
            failures.append(f"block did not settle lying at z={H[1]:.4f}: "
                            f"z in [{float(z.min()):.4f}, {float(z.max()):.4f}]")
        if float(mdp.block_up_axis(e).abs().max()) > 0.25:
            failures.append("block did not settle lying down -- its long axis is not horizontal")
        print(f"[V3] settled lying at z = {float(z.median()) * 1000:.1f} mm "
              f"(expect {H[1] * 1000:.1f}), up-axis {float(mdp.block_up_axis(e).median()):+.3f}")

        # ---- V1: min_grasp_width is the true horizontal shadow ----------------
        # This function has been wrong twice: once projecting each body axis separately
        # (reported ~0 mm for a lying block, i.e. maximally graspable when it is least so),
        # and once normalising a degenerate kink direction (reported 0 mm for an
        # axis-aligned upright block). Both poses are pinned here.
        put(torch.tensor([0.265, 0.0, H[1]], device=dev), lying_q)
        w_lying = float(mdp.min_grasp_width(e).median())
        put(torch.tensor([0.265, 0.0, H[2]], device=dev), up_q)
        w_up = float(mdp.min_grasp_width(e).median())
        print(f"[V1] presented width: lying {w_lying * 1000:.2f} mm, on edge {w_up * 1000:.2f} mm")
        if abs(w_lying - mdp.W_LYING) > 5e-4:
            failures.append(f"lying width {w_lying * 1000:.2f} mm != {mdp.W_LYING * 1000:.2f}")
        if abs(w_up - mdp.W_UP) > 5e-4:
            failures.append(f"on-edge width {w_up * 1000:.2f} mm != {mdp.W_UP * 1000:.2f}")

        # ---- V5: the shaping term spans the reorientation ---------------------
        put(torch.tensor([0.265, 0.0, H[1]], device=dev), lying_q)
        u_lying = float(mdp.uprighting_progress(e).median())
        if mdp.is_reoriented(e).any():
            failures.append("is_reoriented fired for a block lying flat")
        put(torch.tensor([0.265, 0.0, H[2]], device=dev), up_q)
        u_up = float(mdp.uprighting_progress(e).median())
        if not mdp.is_reoriented(e).all():
            failures.append("is_reoriented did not fire for a block up on edge")
        print(f"[V5] uprighting reward: lying {u_lying:.3f} -> on edge {u_up:.3f}")
        if u_lying > 0.02:
            failures.append(f"uprighting reward is {u_lying:.3f} at the start pose, want ~0")
        if u_up < 0.98:
            failures.append(f"uprighting reward is {u_up:.3f} when fully up, want ~1")

        # ---- V5: success predicate, positive case ------------------------------
        ee = mdp.ee_pos_local(e, EE)[0]
        held = torch.tensor([float(ee[0]), float(ee[1]), max(float(ee[2]), mdp.LIFT_Z + 0.010)],
                            device=dev)
        put(held, up_q)
        if not mdp.is_lifted_upright(e, EE).all():
            failures.append("success predicate did not fire for a reoriented block held aloft")
        print(f"[V5+] reoriented + lifted + in the gripper: "
              f"{int(mdp.is_lifted_upright(e, EE).sum())}/{n}")

        # ---- V5: negative controls ---------------------------------------------
        # (a) reoriented and high, but nowhere near the gripper
        put(torch.tensor([float(held[0]), float(held[1]) - 0.40, float(held[2])], device=dev), up_q)
        if mdp.is_lifted_upright(e, EE).any():
            failures.append("negative control (a): success fired for a block far from the gripper")

        # (b) at the gripper and high, but still lying flat -- not reoriented
        put(held, lying_q)
        if mdp.is_lifted_upright(e, EE).any():
            failures.append("negative control (b): success fired for a block that is still flat")

        # (c) reoriented and at the gripper, but never picked up
        put(torch.tensor([float(held[0]), float(held[1]), H[2]], device=dev), up_q)
        if mdp.is_lifted_upright(e, EE).any():
            failures.append("negative control (c): success fired for a block still on the table")

        # ---- V6: rewards finite, and no termination at reset --------------------
        for fn, kw in ((mdp.uprighting_progress, {}),
                       (mdp.reach_block, {"std": 0.12, "ee_frame_cfg": EE}),
                       (mdp.block_lifted, {"minimal_height": 0.075, "ee_max_dist": 0.08,
                                           "ee_frame_cfg": EE})):
            v = fn(e, **kw)
            if not torch.isfinite(v).all():
                failures.append(f"{fn.__name__} produced non-finite values")

        put(torch.tensor([0.265, 0.0, -0.50], device=dev), lying_q)
        if not mdp.block_dropped(e, minimum_height=-0.05).all():
            failures.append("block_dropped did not fire for a block below the table")

        env.reset()
        _, _, term, _, _ = env.step(torch.zeros(n, 7, device=dev))
        if bool(term.any()):
            failures.append(f"{int(term.sum())}/{n} envs terminated on the first step -- a "
                            "toppled-style termination would fire immediately here, since "
                            "lying down is the START state")

    print("\n" + "=" * 74)
    if failures:
        print("[result] FAIL")
        for f in failures:
            print("  - " + f)
    else:
        print("[result] PASS -- geometry, the shadow-width term, the uprighting gradient and")
        print("         the success predicate (with three negative controls) all check out.")
    print("=" * 74)

    # The MDP is self-consistent, which is what the checks above establish. The env's
    # premise is a separate question, and it does not currently hold.
    print("\n" + "!" * 74)
    print("KNOWN ISSUE -- this env's premise is disproven and it needs a redesign.")
    print(f"  The design assumes a block presenting {mdp.W_LYING * 1000:.0f} mm cannot be grasped,")
    print(f"  because the binary open command asks for {mdp.GRIPPER_OPENING * 1000:.1f} mm.")
    print(f"  But the finger joint limits sum to {FORCED_OPEN * 1000:.1f} mm, and a wide block simply")
    print("  forces the fingers past the commanded opening. pregrasp_probe.py measures a")
    print("  100 % lift of the lying block, so the start state IS graspable.")
    print("  No object that fits this arm's usable workspace can be too wide to grasp.")
    print("  See docs/envs/pregrasp.md for the measurements and the redesign options.")
    print("!" * 74)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
