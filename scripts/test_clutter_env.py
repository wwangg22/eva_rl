# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the clutter-extraction env.

Checks the row geometry is the pitch the config claims (measured off the settled blocks, not
read back out of the config), that everything starts upright and undisturbed, and that the
no-topple constraint -- the thing that makes this a clutter task rather than a pick task --
actually fires, and fires only when it should.

Every check appends to ``failures`` instead of asserting, so one run reports everything.

.. code-block:: bash

    python scripts/test_clutter_env.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the clutter-extraction env.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
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

EXPECTED_DIM = 8 + 8 + 7 + 12 + 7  # 42
#: 90 deg about x -- lays a block on its side
LYING = [0.7071, 0.0, 0.0, 0.7071]


def main() -> None:
    failures: list[str] = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    n = e.num_envs
    names = ("target",) + mdp.DISTRACTOR_NAMES

    def put(name, pos, quat=None):
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).repeat(n, 1)
        if quat is None:
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        e.scene[name].write_root_state_to_sim(
            torch.cat([pos + e.scene.env_origins, quat, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def zero_step(k):
        a = torch.zeros(n, 7, device=dev)
        for _ in range(k):
            env.step(a)

    with torch.inference_mode():
        obs, _ = env.reset()
        pol = obs["policy"]

        # ---- V6: observation plumbing ---------------------------------------
        if pol.shape[1] != EXPECTED_DIM:
            failures.append(f"obs dim {pol.shape[1]} != {EXPECTED_DIM}")
        if not torch.isfinite(pol).all():
            failures.append("observation contains non-finite values")
        if mdp.clutter_obs(e).shape != (n, 12):
            failures.append(f"clutter_obs shape {tuple(mdp.clutter_obs(e).shape)} != {(n, 12)}")

        # ---- V3 / V1: the row settles at the pitch the config claims ----------
        zero_step(30)
        ys = torch.stack([mdp.object_pos_local(e, m)[:, 1] for m in names], dim=1)
        zs = torch.stack([mdp.object_pos_local(e, m)[:, 2] for m in names], dim=1)
        order = ys[0].argsort()
        spacing = (ys[0][order][1:] - ys[0][order][:-1])
        free_gap = spacing - 2 * mdp.CL_BLOCK_HALF[1]
        print(f"[V1] settled row pitch {[f'{float(s) * 1000:.1f}' for s in spacing]} mm")
        print(f"[V1] free gap between 30 mm blocks {[f'{float(g) * 1000:.1f}' for g in free_gap]} mm")
        if float(free_gap.min()) <= 0.0:
            failures.append("blocks overlap in the row -- the pitch is narrower than the blocks")
        if not ((zs > 0.020) & (zs < 0.050)).all():
            failures.append(f"blocks did not settle on the table: z in [{float(zs.min()):.4f}, "
                            f"{float(zs.max()):.4f}]")

        # ---- V5: the no-topple constraint, negative case first ----------------
        if mdp.any_distractor_toppled(e).any():
            failures.append("a distractor read as toppled at reset")
        dist0 = float(mdp.distractors_disturbed(e).max())
        print(f"[V5-] at reset: nothing toppled, disturbance {dist0 * 1000:.2f} mm")
        if dist0 > 0.005:
            failures.append(f"distractors_disturbed is {dist0 * 1000:.2f} mm at reset, want ~0 "
                            "-- record_spawn probably did not run last")

        # ---- V5: and the positive case -- it must actually be reachable -------
        d0 = mdp.DISTRACTOR_NAMES[0]
        p0 = mdp.object_pos_local(e, d0)[0].clone()
        put(d0, torch.tensor([float(p0[0]), float(p0[1]), 0.018], device=dev),
            torch.tensor(LYING, device=dev).repeat(n, 1))
        if not mdp.any_distractor_toppled(e).all():
            failures.append(f"any_distractor_toppled did not fire for a {d0} lying on its side "
                            f"-- TOPPLE_DOT = {mdp.TOPPLE_DOT} may be unreachable")
        print(f"[V5+] a distractor laid on its side: toppled "
              f"{int(mdp.any_distractor_toppled(e).sum())}/{n}")

        # a distractor merely SHOVED aside, still upright, must NOT count as toppled
        put(d0, torch.tensor([float(p0[0]), float(p0[1]) - 0.030, float(p0[2])], device=dev))
        if mdp.any_distractor_toppled(e).any():
            failures.append("negative control: a shoved-but-upright distractor read as toppled")
        shove = float(mdp.distractors_disturbed(e).max())
        print(f"[V5-] shoved 30 mm but upright: not toppled, disturbance {shove * 1000:.1f} mm")
        if shove < 0.020:
            failures.append("distractors_disturbed did not register a 30 mm shove")

        env.reset()
        zero_step(20)

        # ---- V5: extraction and goal predicates -------------------------------
        tp = mdp.object_pos_local(e, "target")[0].clone()
        put("target", torch.tensor([float(tp[0]), float(tp[1]), 0.120], device=dev))
        if not mdp.target_extracted(e).all():
            failures.append("target_extracted did not fire for a target lifted clear of the row")

        gx, gy = mdp.GOAL_XY
        put("target", torch.tensor([gx, gy, 0.035], device=dev))
        if not mdp.target_at_goal(e).all():
            failures.append("target_at_goal did not fire for a target set down in the goal zone")
        print(f"[V5+] target in the goal zone: {int(mdp.target_at_goal(e).sum())}/{n}")

        # (a) at the goal in xy but still held high
        put("target", torch.tensor([gx, gy, 0.120], device=dev))
        if mdp.target_at_goal(e).any():
            failures.append("negative control (a): target_at_goal fired for a target held above the goal")

        # (b) set down, but well outside the goal radius
        put("target", torch.tensor([gx + 3 * mdp.GOAL_RADIUS, gy, 0.035], device=dev))
        if mdp.target_at_goal(e).any():
            failures.append("negative control (b): target_at_goal fired outside the goal radius")

        # (c) perfectly at the goal, but a distractor is down -- the constraint must dominate
        put("target", torch.tensor([gx, gy, 0.035], device=dev))
        put(d0, torch.tensor([float(p0[0]), float(p0[1]), 0.018], device=dev),
            torch.tensor(LYING, device=dev).repeat(n, 1))
        if mdp.target_at_goal(e).any():
            failures.append("negative control (c): target_at_goal fired while a distractor was "
                            "toppled -- the no-topple constraint does not bind the success test")
        if mdp.target_extracted(e).any():
            failures.append("negative control (c): target_extracted ignores a toppled distractor")
        print("[V5-] target at the goal but a distractor down: correctly not a success")

        # ---- V6: rewards are finite -------------------------------------------
        for fn, kw in ((mdp.reach_target, {"std": 0.10, "ee_frame_cfg": SceneEntityCfg("ee_frame")}),
                       (mdp.target_to_goal, {"std": 0.12}),
                       (mdp.distractors_disturbed, {}),
                       (mdp.clutter_obs, {})):
            v = fn(e, **kw)
            if not torch.isfinite(v).all():
                failures.append(f"{fn.__name__} produced non-finite values")

        # ---- V6: terminations --------------------------------------------------
        put("target", torch.tensor([0.250, 0.0, -0.50], device=dev))
        if not mdp.block_dropped(e, minimum_height=-0.05, name="target").all():
            failures.append("target_dropped did not fire for a target below the table")

        env.reset()
        _, _, term, _, _ = env.step(torch.zeros(n, 7, device=dev))
        if bool(term.any()):
            failures.append(f"{int(term.sum())}/{n} envs terminated on the first step after reset "
                            "-- the row probably spawns already toppled")

    print("\n" + "=" * 70)
    if failures:
        print("[result] FAIL")
        for f in failures:
            print("  - " + f)
    else:
        print("[result] PASS -- row geometry, the topple constraint (positive and negative),")
        print("         the goal predicate and three negative controls all check out.")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
