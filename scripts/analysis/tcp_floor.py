# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How low can this arm actually hold its TCP above the table?

``reachability_map.py`` answers where the TCP can be placed *kinematically* -- it writes
joint states and reads link poses, with no physics and therefore no table. That is the right
tool for asking about the arm's joint limits, and the wrong one for asking whether a pose is
usable: the gripper is a bulky body, and well before the TCP reaches the table the gripper
bottoms out on it.

This measures the difference. For a grid of (x, z) targets it searches for an arm config
reaching that TCP, commands it **through the env's own action manager** (so the actuator
gains, decimation and action scaling are the real ones), lets it settle, and reports the
achieved TCP. Where the achieved z sits well above the commanded z, the pose is blocked.

The number matters for every challenge task: it sets the lowest feature the gripper can
touch, which decides whether a handle is graspable, whether a block can be pushed on its
upper half, and how tall fixtures have to be.

.. code-block:: bash

    python scripts/analysis/tcp_floor.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the usable TCP floor above the table.")
parser.add_argument("--task", type=str, default="Rebot-PreGrasp-Play-v0")
parser.add_argument("--num_envs", type=int, default=128, help="CEM population")
parser.add_argument("--cem_iters", type=int, default=60)
parser.add_argument("--settle", type=int, default=90)
parser.add_argument("--out", type=str, default="logs/analysis/tcp_floor/tcp_floor.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401

TCP_OFFSET = (-0.0419, 0.0, 0.0)  # measured; see mdp/common.TCP_OFFSET
XS = (0.18, 0.22, 0.26, 0.30)
ZS = (0.020, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100)
#: a pose counts as usable if the achieved TCP is within this of the commanded one
TOL = 0.010


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    n = e.num_envs
    env.reset()

    body_idx = robot.body_names.index("gripper_end")
    li = robot.body_names.index("gripper_left")
    ri = robot.body_names.index("gripper_right")
    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
    fing_dof = [robot.joint_names.index(x) for x in ("joint_left", "joint_right")]
    lim = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)
    lo, hi = lim[arm_dof, 0], lim[arm_dof, 1]
    q_default = torch.as_tensor(robot.data.default_joint_pos[0], device=dev).clone()
    q_arm0 = q_default[arm_dof].clone()
    offs = torch.tensor(TCP_OFFSET, device=dev).repeat(n, 1)
    Y = torch.tensor([0.0, 1.0, 0.0], device=dev)

    # park the block far out of the way -- this measures the arm against the table, nothing else
    blk = e.scene["block"]
    blk.write_root_state_to_sim(torch.cat([
        torch.tensor([0.20, -0.60, 0.10], device=dev).repeat(n, 1) + e.scene.env_origins,
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1),
        torch.zeros((n, 6), device=dev)], dim=1))

    def fk(q_arm):
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = 0.0
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        bp = robot.data.body_pos_w.torch
        lq = robot.data.body_quat_w.torch[:, body_idx, :]
        tcp = bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins
        sep = bp[:, li, :] - bp[:, ri, :]
        return tcp, sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)

    def cem(target, seed):
        mean, std = seed.clone(), torch.full((6,), 0.45, device=dev)
        best_q, best, best_p = seed.clone(), 1e9, 1e9
        for _ in range(args_cli.cem_iters):
            cand = (mean + std * torch.randn(n, 6, device=dev)).clamp(lo, hi)
            cand[0] = mean
            tcp, sep = fk(cand)
            pos = (tcp - target).norm(dim=1)
            cost = pos + 0.25 * (1.0 - (sep @ Y).abs())
            elite = cand[cost.topk(max(8, n // 8), largest=False).indices]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.01)
            j = int(cost.argmin())
            if float(cost[j]) < best:
                best, best_q, best_p = float(cost[j]), cand[j].clone(), float(pos[j])
        return best_q, best_p

    def act(q_arm):
        a = torch.zeros((n, 7), device=dev)
        a[:, :6] = (q_arm - q_arm0.unsqueeze(0)) / 0.5
        a[:, 6] = -1.0  # fingers closed: the widest part of the gripper is what bottoms out
        return a

    rows = []
    print(f"\n{'x [m]':>7} {'z cmd':>7} | {'CEM':>8} | {'z achieved':>11} | {'track err':>10} | usable")
    print("-" * 66)
    for x in XS:
        for z in ZS:
            t = torch.tensor([x, 0.0, z], device=dev)
            q_t, cem_err = cem(t, q_arm0)
            q = q_default.unsqueeze(0).repeat(n, 1)
            q[:, arm_dof] = q_t
            q[:, fing_dof] = 0.0
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            e.sim.forward()
            e.scene.update(e.physics_dt)
            for _ in range(args_cli.settle):
                env.step(act(q_t.unsqueeze(0).repeat(n, 1)))
            bp = robot.data.body_pos_w.torch
            lq = robot.data.body_quat_w.torch[:, body_idx, :]
            got = (bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins)
            got = got.median(dim=0).values
            err = float((got - t).norm())
            usable = err < TOL and cem_err < 0.005
            rows.append({"x": x, "z_cmd": z, "cem_err_m": cem_err,
                         "z_got": float(got[2]), "track_err_m": err, "usable": bool(usable)})
            print(f"{x:7.3f} {z:7.3f} | {cem_err * 1000:7.2f}mm | {float(got[2]):10.4f}m | "
                  f"{err * 1000:9.2f}mm | {'yes' if usable else 'NO'}")
        print("-" * 66)

    print(f"\n{'x [m]':>7} | lowest usable TCP z")
    print("-" * 34)
    floors = {}
    for x in XS:
        ok = [r["z_cmd"] for r in rows if r["x"] == x and r["usable"]]
        floors[x] = min(ok) if ok else None
        print(f"{x:7.3f} | {f'{floors[x] * 1000:.0f} mm' if ok else 'none of the sampled heights'}")

    # The floor itself is best read off the ACHIEVED heights: every command below it lands
    # at the same place, because that is where the gripper rests on the table. Taking the
    # lowest *usable* z per x conflates this with reach limits at the far edge of the
    # envelope, where nothing tracks well at any height.
    blocked = [r["z_got"] for r in rows if r["track_err_m"] >= TOL and r["z_cmd"] <= 0.040]
    print()
    if blocked:
        blocked.sort()
        mid = blocked[len(blocked) // 2]
        print(f"[floor] Commands below the floor all land at z = {min(blocked):.4f} .. "
              f"{max(blocked):.4f} m (median {mid:.4f})")
        print(f"[floor] => TCP FLOOR ~ {mid * 1000:.0f} mm above the table")
    usable = [(r['x'], r['z_cmd']) for r in rows if r["usable"]]
    if usable:
        xs = sorted({x for x, _ in usable})
        print(f"[floor] usable x range {min(xs):.2f} .. {max(xs):.2f} m; "
              f"lowest usable z overall {min(z for _, z in usable) * 1000:.0f} mm")
    print("[floor] Consequence: the gripper cannot be positioned to act on any feature")
    print("        below the floor. Handles, and the part of a block that must be pushed")
    print("        above its centre of mass, have to sit above it.")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump({"tol_m": TOL, "grid": rows,
                   "floor_by_x": {str(k): v for k, v in floors.items()}}, f, indent=2)
    print(f"\n[floor] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
