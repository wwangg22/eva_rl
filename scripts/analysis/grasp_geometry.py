# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What can this gripper actually pick up, as a function of object height and grip height?

Three separate probes kept producing confidently wrong numbers because this was never
measured. The pieces that were known -- an 89.1 mm opening (``gripper_stroke.py``) and a
44 mm TCP floor (``tcp_floor.py``) -- do not by themselves say whether a given object can be
grasped: the fingers are bulky bodies, so a short object can sit entirely below them even
when the TCP is nominally in the right place.

So this sweeps a free-standing block over **height** and **grip height** and, for each cell,
runs the whole manoeuvre the way a policy would: approach along a Cartesian waypoint path
with the fingers wide, close, lift 90 mm, and check the object came along.

A grasp counts only if the fingers ended up **around** the block. The clear finger gap is
``1.0035 * (q_left + q_right) - 1.25`` mm, so fingers resting on the block's faces stall at
its width, fingers that closed on nothing go to ~0 mm, and fingers jammed outside a too-wide
block sit near 89 mm. Without that check a gripper that merely shoves a block up and along
scores as a successful grasp -- which is exactly what happened at first.

.. code-block:: bash

    python scripts/analysis/grasp_geometry.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the usable grasp geometry envelope.")
parser.add_argument("--task", type=str, default="Rebot-PreGrasp-Play-v0")
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--cem_iters", type=int, default=45)
parser.add_argument("--block_h", type=float, required=True, help="block height [m]")
parser.add_argument("--out_dir", type=str, default="logs/analysis/grasp_geometry")
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
BLOCK_X = 0.245
#: the block is rebuilt at each height by writing a new collision cuboid; the width across
#: the fingers is held at 40 mm, comfortably inside the 89.1 mm opening
BLOCK_W = 0.040
BLOCK_D = 0.040
#: swept by the caller, one process per height (`--block_h`; `--height` collides with a
#: SimulationApp argument): a rigid body's collider cannot be resized
#: after the scene is built (a USD xformOp:scale written post-build is ignored by PhysX),
#: and burying the block to expose only part of it just makes the table eject it
HEIGHTS = (args_cli.block_h,)
GRIP_ZS = (0.045, 0.055, 0.070, 0.090)
STANDOFF = 0.070


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    # rebuild the block as a plain cuboid of the size this sweep needs
    env_cfg.scene.block.spawn.size = (BLOCK_D, BLOCK_W, args_cli.block_h)
    env_cfg.scene.block.init_state.pos = [BLOCK_X, 0.0, args_cli.block_h / 2]
    env_cfg.scene.block.init_state.rot = [0.0, 0.0, 0.0, 1.0]
    env_cfg.scene.block.spawn.mass_props.mass = 0.05
    # the wall is irrelevant here and would let a failed grasp turn into a wedge
    env_cfg.scene.wall.init_state.pos = (1.5, 0.0, 0.070)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    block = e.scene["block"]
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
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)

    def fk(q_arm):
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = 0.045
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        bp = robot.data.body_pos_w.torch
        lq = robot.data.body_quat_w.torch[:, body_idx, :]
        tcp = bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins
        sep = bp[:, li, :] - bp[:, ri, :]
        sep = sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)
        fwd = quat_apply(lq, torch.tensor([-1.0, 0.0, 0.0], device=dev).repeat(n, 1))
        return tcp, sep, fwd

    def cem(target, seed, std0=0.45):
        mean, std = seed.clone(), torch.full((6,), std0, device=dev)
        best_q, best, best_p = seed.clone(), 1e9, 1e9
        for _ in range(args_cli.cem_iters):
            cand = (mean + std * torch.randn(n, 6, device=dev)).clamp(lo, hi)
            cand[0] = mean
            tcp, sep, _ = fk(cand)
            pos = (tcp - target).norm(dim=1)
            cost = pos + 0.25 * (1.0 - (sep @ Y).abs())
            elite = cand[cost.topk(max(8, n // 8), largest=False).indices]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.01)
            j = int(cost.argmin())
            if float(cost[j]) < best:
                best, best_q, best_p = float(cost[j]), cand[j].clone(), float(pos[j])
        return best_q, best_p

    def act(q_arm, close):
        a = torch.zeros((n, 7), device=dev)
        a[:, :6] = (q_arm - q_arm0.unsqueeze(0)) / 0.5
        a[:, 6] = -1.0 if close else 1.0
        return a

    def run(q_from, q_to, steps, close):
        for s in range(steps):
            f = min(1.0, (s + 1) / (steps * 0.8))
            env.step(act(((1 - f) * q_from + f * q_to).unsqueeze(0).repeat(n, 1), close))

    def path(targets, seed):
        qs, q = [], seed
        for i, t in enumerate(targets):
            q, err = cem(t, q, std0=0.45 if i == 0 else 0.12)
            qs.append(q)
        return qs, err

    def lerp(a, b, k):
        return [a + (b - a) * (i + 1) / k for i in range(k)]

    rows = []
    print(f"\n  block is {BLOCK_D * 1000:.0f} x {BLOCK_W * 1000:.0f} mm in plan at x = {BLOCK_X}")
    print(f"\n  {'h [mm]':>7} {'grip z':>7} | {'CEM':>7} | {'gap [mm]':>9} | {'encl':>5} "
          f"| {'rose':>5} | {'HELD':>6}")
    print("  " + "-" * 62)
    for h in HEIGHTS:
        for gz in GRIP_ZS:
            if gz > h - 0.005:            # the grip point must be on the block
                continue
            pos = torch.tensor([BLOCK_X, 0.0, h / 2], device=dev).repeat(n, 1)
            block.write_root_state_to_sim(torch.cat(
                [pos + e.scene.env_origins, ident, torch.zeros((n, 6), device=dev)], dim=1))
            e.sim.forward()
            e.scene.update(e.physics_dt)

            grip = torch.tensor([BLOCK_X, 0.0, gz], device=dev)
            q_grip, perr = cem(grip, q_arm0)
            _, _, fwd = fk(q_grip.unsqueeze(0).repeat(n, 1))
            f = fwd[0].clone()
            f[2] = 0.0
            f = f / f.norm().clamp(min=1e-9)
            q_back, _ = path(lerp(grip, grip - STANDOFF * f, 3), q_grip)
            approach = list(reversed(q_back)) + [q_grip]
            q_up, _ = path(lerp(grip, grip + torch.tensor([0.0, 0.0, 0.090], device=dev), 3), q_grip)

            block.write_root_state_to_sim(torch.cat(
                [pos + e.scene.env_origins, ident, torch.zeros((n, 6), device=dev)], dim=1))
            q = q_default.unsqueeze(0).repeat(n, 1)
            q[:, arm_dof] = approach[0]
            q[:, fing_dof] = 0.045
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            e.sim.forward()
            e.scene.update(e.physics_dt)

            z0 = float(pos[0, 2])
            for _ in range(15):
                env.step(act(approach[0].unsqueeze(0).repeat(n, 1), False))
            for i in range(len(approach) - 1):
                run(approach[i], approach[i + 1], 25, False)
            for _ in range(70):
                env.step(act(q_grip.unsqueeze(0).repeat(n, 1), True))
            seq = [q_grip] + q_up
            for i in range(len(seq) - 1):
                run(seq[i], seq[i + 1], 25, True)
            for _ in range(50):
                env.step(act(q_up[-1].unsqueeze(0).repeat(n, 1), True))

            bpos = block.data.root_pos_w.torch - e.scene.env_origins
            bp = robot.data.body_pos_w.torch
            lq = robot.data.body_quat_w.torch[:, body_idx, :]
            tcp = bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins
            gap = 1.0035 * robot.data.joint_pos.torch[:, fing_dof].sum(dim=1) - 0.00125
            encl = (gap - BLOCK_W).abs() < 0.012
            rose = bpos[:, 2] > z0 + 0.045
            near = (tcp - bpos).norm(dim=1) < 0.09
            held = rose & near & encl
            rows.append({"height_m": h, "grip_z": gz, "cem_err_m": perr,
                         "gap_m": float(gap.median()), "enclose_rate": float(encl.float().mean()),
                         "rose_rate": float(rose.float().mean()),
                         "held_rate": float(held.float().mean())})
            print(f"  {h * 1000:7.0f} {gz * 1000:7.0f} | {perr * 1000:6.2f}mm | "
                  f"{float(gap.median()) * 1000:9.1f} | {float(encl.float().mean()):5.0%} | "
                  f"{float(rose.float().mean()):5.0%} | {float(held.float().mean()):6.0%}")
        print("  " + "-" * 62)

    print(f"\n  {'h [mm]':>7} | best grip z | best HELD")
    print("  " + "-" * 38)
    best_by_h = {}
    for h in HEIGHTS:
        sub = [r for r in rows if r["height_m"] == h]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["held_rate"])
        best_by_h[h] = b
        print(f"  {h * 1000:7.0f} | {b['grip_z'] * 1000:11.0f} | {b['held_rate']:9.0%}")

    good = [h for h, b in best_by_h.items() if b["held_rate"] > 0.5]
    print()
    if good:
        print(f"[geom] MINIMUM GRASPABLE OBJECT HEIGHT = {min(good) * 1000:.0f} mm")
        print(f"[geom] best grip height at that size   = {best_by_h[min(good)]['grip_z'] * 1000:.0f} mm")
        print("[geom] Anything shorter cannot be picked up at all, whatever the policy does:")
        print("       the fingers sit above it. That is a usable ungraspability mechanism.")
    else:
        print("[geom] NOTHING was grasped reliably at any height -- investigate before")
        print("       drawing conclusions about any task built on this gripper.")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    out = os.path.join(args_cli.out_dir, f"h{round(args_cli.block_h * 1000):03d}.json")
    with open(out, "w") as f:
        json.dump({"block_w_m": BLOCK_W, "block_d_m": BLOCK_D, "x": BLOCK_X, "grid": rows}, f, indent=2)
    print(f"\n[geom] wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
