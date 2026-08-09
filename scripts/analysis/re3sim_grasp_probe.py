# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can this gripper pick up the three *real* objects captured for the re3sim workstation?

This is the B1 gate for ``docs/envs/re3sim`` and it exists because
``scripts/analysis/grasp_geometry.py`` cannot answer the question. That sweep starts its
grip heights at 45 mm and skips any cell with ``grip_z > h - 5 mm``, so for the two flat
objects in this capture -- a 24 mm roll of tape and a 36 mm tape measure -- it emits **zero
rows**. The 44 mm TCP floor (C9) was likewise measured with the fingers *shut*, which is not
the configuration a grasp approaches in.

So the numbers that rule these objects out have never actually been measured against them.
This does measure them: real footprint, real height, grip heights from 8 mm up, and the full
manoeuvre (approach wide along a Cartesian path, close, lift 90 mm) run through the env's own
action manager so the gains and decimation are the real ones.

A grasp counts only if the fingers ended up **around** the object -- ``grasp_geometry``'s
lesson, kept verbatim. The clear finger gap is ``1.0035 * (q_left + q_right) - 1.25`` mm, so
fingers resting on the object stall near its width, fingers that closed on nothing go to
~0 mm, and fingers jammed outside a too-wide object sit near 89 mm. Without that check a
gripper that merely shoves the object along scores as a successful grasp.

.. code-block:: bash

    python scripts/analysis/re3sim_grasp_probe.py --object rubixcube
    python scripts/analysis/re3sim_grasp_probe.py --object rolloftape
    python scripts/analysis/re3sim_grasp_probe.py --object rolloftape_onedge
    python scripts/analysis/re3sim_grasp_probe.py --object tapemeasure
"""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Object table. Dimensions are the user's caliper measurements from
# data/captures/2026-08-05/measurements.txt; `width` is the span the fingers must
# close across and `depth` the span along the approach.
#
# NOTE the tape measure's third dimension was not measured (the capture sheet records
# height + longest only). 64 mm is an estimate and is flagged in the output.
# ---------------------------------------------------------------------------
OBJECTS = {
    # name:            (shape,      height, width_across_fingers, depth_along_approach, mass)
    "rubixcube":       ("cuboid",   0.056, 0.056, 0.056, 0.073),
    "rolloftape":      ("cylinder", 0.024, 0.091, 0.091, 0.042),
    # a roll of tape stood up on its rim: the 91 mm diameter becomes the height and the
    # 24 mm width becomes the span across the fingers. This is the pose in which it is
    # trivially graspable, and it is a physically natural resting pose for a taped roll.
    "rolloftape_onedge": ("cylinder_edge", 0.091, 0.024, 0.091, 0.042),
    "tapemeasure":     ("cuboid",   0.036, 0.064, 0.0715, 0.184),
    # controls: a block at the height grasp_geometry would have tested, so this probe's
    # numbers can be compared against a configuration known to work
    "control_56":      ("cuboid",   0.056, 0.040, 0.040, 0.050),
    "control_24":      ("cuboid",   0.024, 0.040, 0.040, 0.050),
}

parser = argparse.ArgumentParser(description="Grasp feasibility for the re3sim capture objects.")
parser.add_argument("--task", type=str, default="Rebot-PreGrasp-Play-v0")
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--cem_iters", type=int, default=45)
parser.add_argument("--object", type=str, required=True, choices=sorted(OBJECTS))
parser.add_argument("--object_x", type=float, default=0.245, help="radial placement [m]")
parser.add_argument("--out_dir", type=str, default="logs/analysis/re3sim_grasp")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401

TCP_OFFSET = (-0.0419, 0.0, 0.0)  # measured; see mdp/common.TCP_OFFSET
#: the whole point of this probe -- heights the existing sweep never reached
GRIP_ZS = (0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060)
STANDOFF = 0.075
#: how far above the table the object must end up to count as lifted
LIFT_MARGIN = 0.045


def main() -> None:
    shape, obj_h, obj_w, obj_d, obj_m = OBJECTS[args_cli.object]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5

    # rebuild the "block" as this object's collision proxy
    common = dict(
        rigid_props=env_cfg.scene.block.spawn.rigid_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=obj_m),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.9, restitution=0.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.10)),
    )
    if shape == "cuboid":
        env_cfg.scene.block.spawn = sim_utils.CuboidCfg(size=(obj_d, obj_w, obj_h), **common)
        rest_rot = [0.0, 0.0, 0.0, 1.0]
    elif shape == "cylinder":
        # lying flat: axis vertical, so radius spans both horizontal directions
        env_cfg.scene.block.spawn = sim_utils.CylinderCfg(
            radius=obj_w / 2, height=obj_h, axis="Z", **common
        )
        rest_rot = [0.0, 0.0, 0.0, 1.0]
    else:  # cylinder_edge -- stood up on its rim, axis horizontal along the approach
        env_cfg.scene.block.spawn = sim_utils.CylinderCfg(
            radius=obj_h / 2, height=obj_w, axis="X", **common
        )
        rest_rot = [0.0, 0.0, 0.0, 1.0]

    env_cfg.scene.block.init_state.pos = [args_cli.object_x, 0.0, obj_h / 2]
    env_cfg.scene.block.init_state.rot = rest_rot
    # the wall would let a failed grasp turn into a wedge
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
    ident = torch.tensor(rest_rot, device=dev).repeat(n, 1)

    def fk(q_arm, finger_q=0.045):
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = finger_q
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        bp = robot.data.body_pos_w.torch
        lq = robot.data.body_quat_w.torch[:, body_idx, :]
        tcp = bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins
        sep = bp[:, li, :] - bp[:, ri, :]
        sep = sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)
        fwd = quat_apply(lq, torch.tensor([-1.0, 0.0, 0.0], device=dev).repeat(n, 1))
        # lowest finger-body origin, a cheap proxy for how far the fingers hang below the TCP
        fing_z = torch.minimum(bp[:, li, 2], bp[:, ri, 2]) - e.scene.env_origins[:, 2]
        return tcp, sep, fwd, fing_z

    def cem(target, seed, std0=0.45):
        mean, std = seed.clone(), torch.full((6,), std0, device=dev)
        best_q, best, best_p = seed.clone(), 1e9, 1e9
        for _ in range(args_cli.cem_iters):
            cand = (mean + std * torch.randn(n, 6, device=dev)).clamp(lo, hi)
            cand[0] = mean
            tcp, sep, _, _ = fk(cand)
            pos = (tcp - target).norm(dim=1)
            # A5: position alone is not a sufficient specification -- pin the finger axis
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
        qs, q, err = [], seed, 1e9
        for i, t in enumerate(targets):
            q, err = cem(t, q, std0=0.45 if i == 0 else 0.12)
            qs.append(q)
        return qs, err

    def lerp(a, b, k):
        return [a + (b - a) * (i + 1) / k for i in range(k)]

    def reset_block():
        pos = torch.tensor([args_cli.object_x, 0.0, obj_h / 2], device=dev).repeat(n, 1)
        block.write_root_state_to_sim(torch.cat(
            [pos + e.scene.env_origins, ident, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)
        return pos

    print(f"\n  object   : {args_cli.object}  ({shape})")
    print(f"  height   : {obj_h * 1000:.1f} mm")
    print(f"  across   : {obj_w * 1000:.1f} mm  (commanded opening is 89.1 mm; forced ~120 mm -- C3)")
    print(f"  mass     : {obj_m * 1000:.0f} g   (C2: the 2000 N/m finger drive is required)")
    print(f"  placed at: x = {args_cli.object_x:.3f} m")
    if args_cli.object == "tapemeasure":
        print("  NOTE: the 64 mm cross-finger span is an ESTIMATE -- not on the capture sheet.")

    rows = []
    print(f"\n  {'grip z':>7} | {'CEM':>8} | {'TCPz got':>9} | {'fing z':>7} | {'gap':>7} | "
          f"{'encl':>5} | {'rose':>5} | {'HELD':>6}")
    print("  " + "-" * 76)
    for gz in GRIP_ZS:
        if gz > obj_h - 0.004:  # the grip point has to be on the object
            continue
        pos = reset_block()

        grip = torch.tensor([args_cli.object_x, 0.0, gz], device=dev)
        q_grip, perr = cem(grip, q_arm0)
        _, _, fwd, fz = fk(q_grip.unsqueeze(0).repeat(n, 1))
        f = fwd[0].clone()
        f[2] = 0.0
        f = f / f.norm().clamp(min=1e-9)
        q_back, _ = path(lerp(grip, grip - STANDOFF * f, 3), q_grip)
        approach = list(reversed(q_back)) + [q_grip]
        q_up, _ = path(lerp(grip, grip + torch.tensor([0.0, 0.0, 0.090], device=dev), 3), q_grip)

        reset_block()
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
        # A6: the search is FINISHED before the fingers close. q_up was solved above.
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
        # for a cylinder grasped across its diameter the fingers stall at the diameter;
        # a 12 mm band accommodates squeeze and the calibration residual
        encl = (gap - obj_w).abs() < 0.012
        rose = bpos[:, 2] > z0 + LIFT_MARGIN
        near = (tcp - bpos).norm(dim=1) < 0.09
        held = rose & near & encl
        rows.append({
            "grip_z": gz, "cem_err_m": perr, "tcp_z_got": float(tcp[:, 2].median()),
            "finger_z_at_grip": float(fz.median()), "gap_m": float(gap.median()),
            "enclose_rate": float(encl.float().mean()), "rose_rate": float(rose.float().mean()),
            "near_rate": float(near.float().mean()), "held_rate": float(held.float().mean()),
        })
        print(f"  {gz * 1000:7.0f} | {perr * 1000:6.2f}mm | {float(tcp[:, 2].median()) * 1000:8.1f} | "
              f"{float(fz.median()) * 1000:6.1f} | {float(gap.median()) * 1000:6.1f} | "
              f"{float(encl.float().mean()):5.0%} | {float(rose.float().mean()):5.0%} | "
              f"{float(held.float().mean()):6.0%}")

    print("  " + "-" * 76)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    out = os.path.join(args_cli.out_dir, f"{args_cli.object}.json")
    with open(out, "w") as f:
        json.dump({"object": args_cli.object, "shape": shape, "height_m": obj_h,
                   "width_m": obj_w, "depth_m": obj_d, "mass_kg": obj_m,
                   "x": args_cli.object_x, "grid": rows}, f, indent=2)

    if rows:
        best = max(rows, key=lambda r: r["held_rate"])
        print(f"\n[b1] best grip z = {best['grip_z'] * 1000:.0f} mm -> HELD {best['held_rate']:.0%} "
              f"(enclosed {best['enclose_rate']:.0%}, rose {best['rose_rate']:.0%})")
        if best["held_rate"] > 0.5:
            print(f"[b1] VERDICT: {args_cli.object} IS GRASPABLE at grip z = "
                  f"{best['grip_z'] * 1000:.0f} mm")
        else:
            print(f"[b1] VERDICT: {args_cli.object} is NOT reliably graspable as posed.")
            print("[b1]   enclose vs rose separates the two failure modes: low enclose means the")
            print("[b1]   fingers never got around it (too wide, or too low for the fingers);")
            print("[b1]   high enclose with low rose means it was gripped but slipped (C2 force).")
    else:
        print(f"\n[b1] no cells ran -- every grip height exceeded {obj_h * 1000:.0f} mm - 4 mm")
    print(f"[b1] wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
