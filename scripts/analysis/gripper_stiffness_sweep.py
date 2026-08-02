# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does the simulated gripper reproduce the hardware's rated payload?

The RS-rebot-dev-arm gripper is rated to hold ~2.5 kg, and its URDF/USD carries the real
``maxForce = 500 N``. But the USD also authors ``drive:linear:physics:stiffness = 100``
(N/m -- the stage is ``metersPerUnit = 1``, so this is not a unit-scaling artefact). A
position drive only produces force proportional to its tracking error, and the finger can
only be in error by about half the object width, so 100 N/m yields roughly

    F = 100 N/m x 0.015 m = 1.5 N per finger

of squeeze, no matter what ``maxForce`` says. That is ~0.3 kg of friction hold, which is
consistent with the repo's note that 0.35 kg YCB masses were "far too heavy" -- and two
orders of magnitude below the hardware spec.

This sweeps finger stiffness against payload to find the stiffness that reproduces the
rated 2.5 kg, and reports the squeeze force actually generated.

.. code-block:: bash

    python scripts/analysis/gripper_stiffness_sweep.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gripper stiffness vs payload sweep.")
parser.add_argument("--out_dir", type=str, default="logs/analysis/gripper_stiffness")
parser.add_argument("--obj_width", type=float, default=0.034, help="object width [m]")
parser.add_argument("--repeats", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import itertools
import json
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from reBot_RL.assets import REBOT_ARM_CFG
from reBot_RL.tasks.manager_based.lift.rebot_lift_env_cfg import _START_POSE

TCP_OFFSET = (-0.075, 0.0, 0.0)
GRIPPER_LINK = "gripper_end"
OBJ_H = 0.070
SUPPORT_TOP = 0.170 - OBJ_H / 2

#: as authored in the USD, then decades above it
STIFFNESS = [100.0, 500.0, 2000.0, 8000.0, 30000.0]
#: 2.5 kg is the vendor's rated payload
MASSES = [0.05, 0.25, 1.0, 2.5]
GRID = list(itertools.product(STIFFNESS, MASSES))
NUM_ENVS = len(GRID) * args_cli.repeats


@configclass
class _SceneCfg(InteractiveSceneCfg):
    robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    support = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, SUPPORT_TOP / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, SUPPORT_TOP),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )
    obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obj",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.30)),
        spawn=sim_utils.CuboidCfg(
            size=(args_cli.obj_width, args_cli.obj_width, OBJ_H),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.8)),
        ),
    )


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1 / 400,
            device=args_cli.device,
            physics=PhysxCfg(bounce_threshold_velocity=0.01, gpu_max_rigid_patch_count=2**19),
        )
    )
    scene = InteractiveScene(_SceneCfg(num_envs=NUM_ENVS, env_spacing=2.5))
    sim.reset()

    robot, obj = scene["robot"], scene["obj"]
    device = robot.device
    body_idx = robot.body_names.index(GRIPPER_LINK)
    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
    fing_dof = [robot.joint_names.index(n) for n in ("joint_left", "joint_right")]

    masses = torch.tensor([[GRID[i % len(GRID)][1]] for i in range(NUM_ENVS)],
                          device=device, dtype=torch.float32)
    obj.set_masses_index(masses=masses, env_ids=torch.arange(NUM_ENVS, device=device))

    # per-env finger stiffness -- the variable under test
    stiff = torch.tensor([[GRID[i % len(GRID)][0]] * 2 for i in range(NUM_ENVS)],
                         device=device, dtype=torch.float32)
    robot.write_joint_stiffness_to_sim_index(
        stiffness=stiff, joint_ids=fing_dof, env_ids=torch.arange(NUM_ENVS, device=device)
    )

    start = torch.tensor([_START_POSE[f"joint{i}"] for i in range(1, 7)], device=device)
    q = torch.zeros((NUM_ENVS, robot.num_joints), device=device)
    q[:, arm_dof] = start
    q[:, fing_dof] = 0.045
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    sim.forward()
    robot.update(0.0)

    def tcp() -> torch.Tensor:
        lp = torch.as_tensor(robot.data.body_pos_w[:, body_idx, :], device=device)
        lq = torch.as_tensor(robot.data.body_quat_w[:, body_idx, :], device=device)
        return lp + quat_apply(lq, torch.tensor(TCP_OFFSET, device=device).repeat(NUM_ENVS, 1))

    p = tcp()
    p[:, 2] = SUPPORT_TOP + OBJ_H / 2
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(NUM_ENVS, 1)
    obj.write_root_state_to_sim(torch.cat([p, ident, torch.zeros((NUM_ENVS, 6), device=device)], dim=1))

    def run(n: int) -> None:
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(1 / 400)

    run(20)
    q[:, fing_dof] = 0.0
    robot.set_joint_position_target(q)
    run(60)

    # squeeze force actually produced: |applied finger torque| (prismatic -> newtons)
    squeeze = torch.as_tensor(robot.data.applied_torque[:, fing_dof], device=device).abs().mean(dim=1)

    # lift straight up off the support, then hold
    target = q.clone()
    for s in range(200):
        f = min(1.0, s / 120.0)
        target[:, arm_dof[1]] = start[1] + 0.40 * f
        target[:, arm_dof[3]] = start[3] - 0.30 * f
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 400)

    delta = torch.as_tensor(obj.data.root_pos_w, device=device) - tcp()
    err = delta.norm(dim=1)
    # A dropped object free-falls to dz ~ -2.2 m; a retained one sits within a few
    # centimetres of where the fingers closed on it (the baseline offset is ~+19 mm,
    # because the TCP lands below the object's centre). Anything in between is a slip.
    held = (delta[:, 2] > -0.10) & (delta[:, :2].norm(dim=1) < 0.15)

    # diagnosis: separate "slipped down out of the fingers" (dz very negative) from
    # "squirted out sideways" (large horizontal offset) from "arm never lifted it"
    print("\n  diagnosis -- final object-minus-TCP offset [mm], and arm joint2 tracking error [rad]")
    j2_err = (torch.as_tensor(robot.data.joint_pos[:, arm_dof[1]], device=device) - target[:, arm_dof[1]]).abs()
    for gi, (k, m) in enumerate(GRID):
        idx = [gi + j * len(GRID) for j in range(args_cli.repeats) if gi + j * len(GRID) < NUM_ENVS]
        dz = 1000 * float(delta[idx, 2].mean())
        dxy = 1000 * float(delta[idx, :2].norm(dim=1).mean())
        je = float(j2_err[idx].mean())
        print(f"    k={k:>7.0f} m={m:>5.2f}kg   dz={dz:+9.1f}  dxy={dxy:8.1f}  j2_err={je:.4f}")

    rows = []
    for gi, (k, m) in enumerate(GRID):
        idx = [gi + j * len(GRID) for j in range(args_cli.repeats) if gi + j * len(GRID) < NUM_ENVS]
        rows.append({
            "stiffness_N_per_m": k,
            "mass_kg": m,
            "n": len(idx),
            "held": int(sum(int(held[i]) for i in idx)),
            "squeeze_N": float(sum(float(squeeze[i]) for i in idx) / max(len(idx), 1)),
        })

    os.makedirs(args_cli.out_dir, exist_ok=True)
    with open(os.path.join(args_cli.out_dir, "stiffness_sweep.json"), "w") as f:
        json.dump({"width_m": args_cli.obj_width, "grid": rows}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"GRIPPER STIFFNESS vs PAYLOAD   (object {args_cli.obj_width * 1000:.0f} mm wide, mu = 1.0)")
    print("=" * 72)
    print(f"{'stiffness':>12} {'squeeze':>9} | " + " ".join(f"{m:>6}kg" for m in MASSES))
    print("-" * 72)
    for k in STIFFNESS:
        sel = [r for r in rows if r["stiffness_N_per_m"] == k]
        sq = sel[0]["squeeze_N"]
        cells = " ".join(f"{r['held']:>3}/{r['n']:<4}" for r in sel)
        tag = "  <- as authored in the USD" if k == 100.0 else ""
        print(f"{k:12.0f} {sq:8.2f}N | {cells}{tag}")
    print("\n(held = still between the fingers after being lifted off the support)")


if __name__ == "__main__":
    main()
    simulation_app.close()
