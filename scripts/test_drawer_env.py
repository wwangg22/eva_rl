# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the drawer-ordering env.

Checks the cabinet articulation parses with the joint and travel it claims, that the drawer
starts shut, that the handle is high enough for this arm to actually reach, and -- the part
that matters -- that the precedence gate has teeth: the carry reward and the success
predicate must be *exactly zero* while the drawer is shut, no matter where the block is.

Every check appends to ``failures`` instead of asserting, so one run reports everything.

.. code-block:: bash

    python scripts/test_drawer_env.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the drawer-ordering env.")
parser.add_argument("--task", type=str, default="Rebot-DrawerOrder-Play-v0")
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

EXPECTED_DIM = 8 + 8 + 7 + 3 + 7  # 33
#: measured usable TCP floor above the table -- scripts/analysis/tcp_floor.py
TCP_FLOOR = 0.044

DRAWER = SceneEntityCfg("drawer")
EE = SceneEntityCfg("ee_frame")


def main() -> None:
    failures: list[str] = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    n = e.num_envs
    drawer = e.scene["drawer"]

    def set_drawer(q: float) -> None:
        pos = torch.full((n, 1), q, device=dev)
        drawer.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def put_block(pos) -> None:
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).repeat(n, 1)
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        e.scene["block"].write_root_state_to_sim(
            torch.cat([pos + e.scene.env_origins, quat, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    with torch.inference_mode():
        obs, _ = env.reset()
        pol = obs["policy"]

        # ---- V6: observation plumbing ---------------------------------------
        if pol.shape[1] != EXPECTED_DIM:
            failures.append(f"obs dim {pol.shape[1]} != {EXPECTED_DIM}")
        if not torch.isfinite(pol).all():
            failures.append("observation contains non-finite values")

        # ---- V1: the articulation is what the config claims ------------------
        if drawer.joint_names != ["DrawerJoint"]:
            failures.append(f"drawer joints {drawer.joint_names} != ['DrawerJoint']")
        lim = torch.as_tensor(drawer.data.joint_pos_limits[0], device=dev)[0]
        want_lo = -mdp.DRAWER_TRAVEL
        if abs(float(lim[0]) - want_lo) > 1e-4 or abs(float(lim[1])) > 1e-4:
            failures.append(f"drawer joint limits {lim.tolist()} != [{want_lo}, 0.0]")
        print(f"[V1] DrawerJoint limits {float(lim[0]) * 1000:.1f} .. {float(lim[1]) * 1000:.1f} mm")

        # ---- V1: the handle is reachable at all ------------------------------
        # This is the check that would have caught the original design: the handle sat at
        # z = 26 mm, below the measured TCP floor, so no policy could ever have touched it.
        handle_z = mdp.CABINET_Z + 0.026
        print(f"[V1] handle centre at z = {handle_z * 1000:.1f} mm, TCP floor {TCP_FLOOR * 1000:.0f} mm")
        if handle_z < TCP_FLOOR:
            failures.append(f"handle at z={handle_z * 1000:.1f} mm is below the {TCP_FLOOR * 1000:.0f} mm "
                            "TCP floor -- the gripper can never reach it")

        # ---- V3: the drawer starts shut --------------------------------------
        frac = mdp.drawer_opening(e, DRAWER)
        if float(frac.max()) > 1e-3:
            failures.append(f"drawer did not start shut: max opening {float(frac.max()):.3f}")
        if mdp.drawer_is_open(e, DRAWER).any():
            failures.append("drawer_is_open fired at reset")

        # ---- V5: opening fraction tracks the joint ---------------------------
        set_drawer(-mdp.DRAWER_TRAVEL)
        frac = mdp.drawer_opening(e, DRAWER)
        if abs(float(frac.mean()) - 1.0) > 1e-3:
            failures.append(f"opening fraction at full travel is {float(frac.mean()):.3f}, want 1.0")
        if not mdp.drawer_is_open(e, DRAWER).all():
            failures.append("drawer_is_open did not fire at full travel")
        set_drawer(-mdp.DRAWER_TRAVEL * mdp.OPEN_FRAC * 0.5)
        if mdp.drawer_is_open(e, DRAWER).any():
            failures.append("drawer_is_open fired at half the open threshold")

        # ---- V5: block in the cavity, positive case --------------------------
        set_drawer(-mdp.DRAWER_TRAVEL)
        cx = mdp.CABINET_XY[0] - mdp.DRAWER_TRAVEL
        cavity_z = (mdp.CAVITY_Z_MIN + mdp.CAVITY_Z_MAX) / 2
        put_block(torch.tensor([cx, mdp.CABINET_XY[1], cavity_z], device=dev))
        if not mdp.block_in_drawer(e, DRAWER).all():
            failures.append("block_in_drawer did not fire for a block at the open cavity centre")
        if not mdp.stowed(e, DRAWER).all():
            failures.append("stowed did not fire with the drawer open and the block in the cavity")
        print(f"[V5+] open cavity centre ({cx:.3f}, 0, {cavity_z:.3f}): "
              f"stowed {int(mdp.stowed(e, DRAWER).sum())}/{n}")

        # ---- V5: negative controls -------------------------------------------
        # (a) THE precedence gate: drawer shut, block placed exactly where the cavity would
        #     be. This is the one that decides whether the task can be short-circuited.
        set_drawer(0.0)
        put_block(torch.tensor([mdp.CABINET_XY[0], mdp.CABINET_XY[1], cavity_z], device=dev))
        if mdp.stowed(e, DRAWER).any():
            failures.append("negative control (a): stowed fired with the drawer SHUT -- the "
                            "precedence constraint can be short-circuited")
        if float(mdp.block_to_drawer(e, std=0.12, asset_cfg=DRAWER).abs().max()) > 0.0:
            failures.append("negative control (a): the carry reward is non-zero while shut, so "
                            "there is a gradient toward stowing before opening")

        # (b) drawer open, block still on the table where it spawns
        set_drawer(-mdp.DRAWER_TRAVEL)
        put_block(torch.tensor([0.235, -0.135, 0.025], device=dev))
        if mdp.block_in_drawer(e, DRAWER).any():
            failures.append("negative control (b): block_in_drawer fired for a block on the table")

        # (c) drawer open, block hovering above the cavity rather than in it
        put_block(torch.tensor([cx, mdp.CABINET_XY[1], 0.200], device=dev))
        if mdp.block_in_drawer(e, DRAWER).any():
            failures.append("negative control (c): block_in_drawer fired for a block above the cavity")

        # (d) drawer open, block below the cavity floor (i.e. under the plinth line)
        put_block(torch.tensor([cx, mdp.CABINET_XY[1], 0.010], device=dev))
        if mdp.block_in_drawer(e, DRAWER).any():
            failures.append("negative control (d): block_in_drawer fired for a block below the cavity")

        # ---- V6: the reward gates behave as the design claims -----------------
        set_drawer(0.0)
        put_block(torch.tensor([0.235, -0.135, 0.025], device=dev))
        shut_carry = float(mdp.block_to_drawer(e, std=0.12, asset_cfg=DRAWER).max())
        shut_reach = float(mdp.reach_handle(e, std=0.10, ee_frame_cfg=EE, asset_cfg=DRAWER).max())
        set_drawer(-mdp.DRAWER_TRAVEL)
        open_carry = float(mdp.block_to_drawer(e, std=0.12, asset_cfg=DRAWER).max())
        open_reach = float(mdp.reach_handle(e, std=0.10, ee_frame_cfg=EE, asset_cfg=DRAWER).max())
        print(f"[V6] carry reward  shut {shut_carry:.4f} -> open {open_carry:.4f}  (gate on open)")
        print(f"[V6] handle reward shut {shut_reach:.4f} -> open {open_reach:.4f}  (gate off open)")
        if shut_carry != 0.0:
            failures.append("block_to_drawer is not exactly zero while shut")
        if open_carry <= 0.0:
            failures.append("block_to_drawer is still zero once open -- stage 2 has no gradient")
        if open_reach != 0.0:
            failures.append("reach_handle is not zero once open -- hovering at the handle stays "
                            "a local optimum")
        if shut_reach <= 0.0:
            failures.append("reach_handle is zero while shut -- stage 1 has no gradient")

        for fn, kw in ((mdp.opening_progress, {"asset_cfg": DRAWER}),
                       (mdp.drawer_obs, {"asset_cfg": DRAWER})):
            v = fn(e, **kw)
            if not torch.isfinite(v).all():
                failures.append(f"{fn.__name__} produced non-finite values")

        # ---- V6: terminations -------------------------------------------------
        put_block(torch.tensor([0.235, -0.135, -0.50], device=dev))
        if not mdp.block_dropped(e, minimum_height=-0.05).all():
            failures.append("block_dropped did not fire for a block below the table")

        # nothing should terminate at reset
        env.reset()
        _, _, term, trunc, _ = env.step(torch.zeros(n, 7, device=dev))
        if bool(term.any()):
            failures.append(f"{int(term.sum())}/{n} envs terminated on the first step after reset")

    print("\n" + "=" * 70)
    if failures:
        print("[result] FAIL")
        for f in failures:
            print("  - " + f)
    else:
        print("[result] PASS -- articulation, handle reachability, the opening predicate and")
        print("         the precedence gate (with four negative controls) all check out.")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
