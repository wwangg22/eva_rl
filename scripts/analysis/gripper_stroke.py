# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How wide does this gripper actually open, in metres of clear space between the fingers?

This is a load-bearing number that was never measured. ``_GRIPPER_OPEN = 0.045`` is a
*per-finger prismatic joint value*, not an opening, and the two finger joints do not even
share a limit (left 0..0.050, right 0..0.0715). The pre-grasp task's whole premise -- that
a block presenting 50 mm cannot be grasped -- is false if the real opening is 90 mm.

Measured two independent ways, which must agree:

1. **Kinematic.** Finger body origins are read from the physics view (NOT from
   ``UsdGeom.BBoxCache``, which reads the USD xform hierarchy that PhysX never writes back
   to -- it reports a constant bound at every joint value). This gives d(separation)/dq.
2. **Contact.** A block of known width is pinned in place between the fingers and the
   gripper is commanded shut. The joints stall when the finger faces touch the block, so
   the stalled joint value is exactly the one whose clear gap equals the block width. This
   calibrates the offset that the kinematic measurement cannot supply, and pins down the
   widest block that can be admitted at all.

.. code-block:: bash

    python scripts/analysis/gripper_stroke.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the true gripper opening.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401

OUT = "logs/analysis/gripper_stroke/gripper_stroke.json"


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    robot = e.scene["robot"]
    block = e.scene["block"]
    dev = e.device
    n = e.num_envs
    env.reset()

    li = robot.body_names.index("gripper_left")
    ri = robot.body_names.index("gripper_right")
    fing = [robot.joint_names.index(x) for x in ("joint_left", "joint_right")]
    lim = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)
    q_lo = lim[fing, 0].cpu().numpy()
    q_hi = lim[fing, 1].cpu().numpy()
    q_default = torch.as_tensor(robot.data.default_joint_pos[0], device=dev).clone()

    print(f"[stroke] joint_left  limits {q_lo[0]:+.4f} .. {q_hi[0]:+.4f} m")
    print(f"[stroke] joint_right limits {q_lo[1]:+.4f} .. {q_hi[1]:+.4f} m")

    # ---- 1. kinematic: finger body separation vs joint value -----------------
    print("\n[stroke] 1. kinematic sweep (physics body poses)")
    print(f"{'q [m]':>8} | {'|sep| [mm]':>10} | separation vector (x, y, z) [mm]")
    print("-" * 66)
    qs, seps = [], []
    for q in (0.0, 0.010, 0.020, 0.030, 0.040, 0.050):
        js = q_default.unsqueeze(0).repeat(n, 1)
        js[:, fing] = q
        robot.write_joint_state_to_sim(js, torch.zeros_like(js))
        e.sim.forward()
        robot.update(0.0)
        bp = robot.data.body_pos_w.torch
        v = (bp[:, li, :] - bp[:, ri, :])[0].cpu().numpy()
        qs.append(q)
        seps.append(float(np.linalg.norm(v)))
        print(f"{q:8.3f} | {seps[-1] * 1000:10.3f} | {v[0] * 1000:8.2f} {v[1] * 1000:8.2f} {v[2] * 1000:8.2f}")

    slope = np.polyfit(qs, seps, 1)[0]
    print(f"\n[stroke] d(separation)/dq = {slope:.3f}  "
          f"({'both fingers move' if slope > 1.5 else 'one finger moves' if slope > 0.5 else 'NEITHER MOVES'})")

    # ---- 2. contact: stall the fingers on a pinned block of known width -------
    # Each env gets a different block width. The block is re-pinned every step so it acts
    # as an immovable gauge; the fingers stall where their inner faces touch it.
    print("\n[stroke] 2. contact calibration (fingers stall on a pinned gauge block)")
    # Widths are obtained by ROTATING the 45 x 30 x 70 mm block, not by scaling it: the
    # block presents 30 mm across y at identity, 45 mm rotated 90 deg about z, and 70 mm
    # rotated 90 deg about x. Scaling was tried first and every width stalled the fingers
    # at the same joint value -- see `scale_fidelity` below for why.
    ROTS = [
        ("identity   (30 mm)", 0.030, [0.0, 0.0, 0.0, 1.0]),
        ("yaw 90 deg (45 mm)", 0.045, [0.0, 0.0, 0.70711, 0.70711]),
        ("roll 90 deg (70 mm)", 0.070, [0.70711, 0.0, 0.0, 0.70711]),
    ]
    # env 0 is the control: gauge parked far away, so we learn where the fingers stall on
    # their own. Without it a constant stall value looks like a measurement.
    plan = [("control", 0.0, [0.0, 0.0, 0.0, 1.0])]
    while len(plan) < n:
        plan.append(ROTS[(len(plan) - 1) % len(ROTS)])
    widths = np.array([p[1] for p in plan])
    quats = torch.tensor([p[2] for p in plan], device=dev, dtype=torch.float32)

    # park the arm at its default pose and put the gauge between the fingers
    js = q_default.unsqueeze(0).repeat(n, 1)
    js[:, fing] = 0.050
    robot.write_joint_state_to_sim(js, torch.zeros_like(js))
    e.sim.forward()
    robot.update(0.0)
    bp = robot.data.body_pos_w.torch
    mid = (bp[:, li, :] + bp[:, ri, :]) / 2.0
    # gauge axis = the finger separation direction, so the block's y axis must lie along it
    sep_dir = (bp[:, li, :] - bp[:, ri, :])
    sep_dir = sep_dir / sep_dir.norm(dim=1, keepdim=True)
    print(f"[stroke] gripper mid-point (env 0, local) "
          f"{(mid[0] - e.scene.env_origins[0]).cpu().numpy().round(4)}, "
          f"sep dir {sep_dir[0].cpu().numpy().round(3)}")

    mid = mid.clone()
    mid[0, 2] += 0.30  # env 0 control: gauge parked well clear of the gripper
    state = torch.cat([mid, quats, torch.zeros((n, 6), device=dev)], dim=1)

    # command the fingers shut and hold the gauge pinned every physics step
    tgt = js.clone()
    tgt[:, fing] = 0.0
    for _ in range(400):
        block.write_root_state_to_sim(state)
        robot.set_joint_position_target(tgt)
        e.scene.write_data_to_sim()
        e.sim.step()
        e.scene.update(e.physics_dt)

    q_final = robot.data.joint_pos.torch[:, fing].cpu().numpy()
    free = float(q_final[0].sum())
    print(f"\n[stroke] control (no gauge): fingers close to q_l+q_r = {free * 1000:.3f} mm")
    print(f"\n{'gauge':>20} | {'width [mm]':>11} | {'q_l+q_r [mm]':>13} | verdict")
    print("-" * 70)
    rows = []
    for i, (label, w, _) in enumerate(plan):
        tot = float(q_final[i].sum())
        held = tot > free + 0.002
        rows.append({"label": label, "width_m": float(w), "q_sum": tot, "stalled": bool(held)})
        tag = "control" if i == 0 else ("stalled on gauge" if held else "closed past it")
        print(f"{label:>20} | {w * 1000:11.2f} | {tot * 1000:13.3f} | {tag}")

    stalled = [r for r in rows[1:] if r["stalled"]]
    print()
    if len({r["width_m"] for r in stalled}) >= 2:
        # gap(q) is affine; fit it from the widths that actually stalled the fingers
        w_arr = np.array([r["width_m"] for r in stalled])
        q_arr = np.array([r["q_sum"] for r in stalled])
        a, b = np.polyfit(q_arr, w_arr, 1)
        resid = np.abs(np.polyval([a, b], q_arr) - w_arr).max()
        print(f"[stroke] gap(q_l + q_r) = {a:.4f} * q {b * 1000:+.3f} mm   (max resid {resid * 1000:.3f} mm)")
        print(f"[stroke] MAX OPENING at q = 0.050 each  = {(a * 0.100 + b) * 1000:.2f} mm")
        print(f"[stroke] OPENING at the commanded 0.045 = {(a * 0.090 + b) * 1000:.2f} mm")
        print(f"[stroke] widest gauge that stalled the fingers = {max(w_arr) * 1000:.2f} mm")
    else:
        print("[stroke] too few distinct gauge widths stalled the fingers to fit a calibration")

    # ---- 3. does a USD xformOp:scale reach the COLLISION geometry? ------------
    # The grasp-envelope sweep varied object width by scaling the prim. If scale only
    # affects the render mesh, that whole sweep measured one width four times over.
    # Test: halve the block's z, drop it, and see whether it rests at 17.5 mm or 35 mm.
    print("\n[stroke] 3. does xformOp:scale reach the collision geometry?")
    from pxr import Gf, UsdGeom

    stage = e.sim.stage
    for i in range(n):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Block")
        xf = UsdGeom.Xformable(prim)
        ops = [op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
        (ops[0] if ops else xf.AddScaleOp()).Set(Gf.Vec3f(1.0, 1.0, 0.5))
    e.sim.reset()

    drop = torch.tensor([0.30, -0.20, 0.12], device=dev).repeat(n, 1) + e.scene.env_origins
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
    block.write_root_state_to_sim(torch.cat([drop, ident, torch.zeros((n, 6), device=dev)], dim=1))
    for _ in range(400):
        e.scene.write_data_to_sim()
        e.sim.step()
        e.scene.update(e.physics_dt)
    rest_z = float((block.data.root_pos_w.torch - e.scene.env_origins)[:, 2].median())
    print(f"[stroke] block scaled z x0.5, dropped, rests at centre z = {rest_z * 1000:.2f} mm")
    print(f"[stroke]   expected  {0.035 * 1000:.1f} mm if scale is IGNORED by collision")
    print(f"[stroke]   expected  {0.0175 * 1000:.1f} mm if scale IS applied to collision")
    scale_ok = abs(rest_z - 0.0175) < abs(rest_z - 0.035)
    print(f"[stroke]   => scale {'IS' if scale_ok else 'is NOT'} applied to the collider")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"kinematic": {"q": qs, "sep_m": seps, "slope": float(slope)},
                   "joint_limits": {"left": q_hi[0].tolist(), "right": q_hi[1].tolist()},
                   "contact": rows,
                   "scale_reaches_collider": bool(scale_ok), "scaled_rest_z_m": rest_z}, f, indent=2)
    print(f"\n[stroke] wrote {OUT}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
