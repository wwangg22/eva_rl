# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Empirical grasp envelope: what can this gripper actually hold, and how heavy?

The repo overrides YCB masses to 0.04 kg with the note "far too heavy for this small
arm", but that number was never swept. Every new task design depends on it: a task that
needs a 0.2 kg object, or a 60 mm-wide one, is not worth building if the gripper drops it.

Method: one procedural box per env, teleported between the open fingers at a fixed arm
pose, gripper closed, then carried through a lift + wrist-rotation trajectory. An object
is "held" if it is still between the fingers at the end. Sweeps object width, mass and
friction across envs.

.. code-block:: bash

    python scripts/analysis/grasp_envelope.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grasp payload / width envelope sweep.")
parser.add_argument("--out_dir", type=str, default="logs/analysis/grasp_envelope")
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--carry_steps", type=int, default=160)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import itertools
import json
import os

import numpy as np
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

# swept factors -- widths bracket the 0.045 m finger stroke, masses bracket the repo's 0.04 kg
WIDTHS = [0.018, 0.026, 0.034, 0.042]
MASSES = [0.02, 0.05, 0.10, 0.20, 0.40]
FRICTIONS = [0.6, 1.0]
GRID = list(itertools.product(WIDTHS, MASSES, FRICTIONS))
REPEATS = 4  # per cell, with small pose jitter
NUM_ENVS = len(GRID) * REPEATS


# the object is *cloned* across envs (Isaac Lab requires one asset entry, N instances),
# so per-env variation is applied afterwards: width via a USD scale write before sim
# start, mass and friction via the physics tensor views after it.
BASE_W = 0.030
BASE_H = 0.070
# the pedestal top sits one half-object below the TCP at ``_START_POSE`` (fingertips
# hover at ~(0.35, 0, 0.17)), so the object rests exactly between the open fingers
SUPPORT_TOP = 0.170 - BASE_H / 2


@configclass
class _GraspSceneCfg(InteractiveSceneCfg):
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
            size=(BASE_W, BASE_W, BASE_H),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
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
    scene_cfg = _GraspSceneCfg(num_envs=NUM_ENVS, env_spacing=2.5)
    # per-env USD scale requires the physics NOT be replicated from a single prototype
    scene_cfg.replicate_physics = False
    scene = InteractiveScene(scene_cfg)

    # -- per-env width, applied on the USD stage BEFORE the sim starts ------
    from pxr import Gf, Sdf, Vt  # noqa: PLC0415

    stage = sim.stage
    prim_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/Obj")
    with Sdf.ChangeBlock():
        for i in range(NUM_ENVS):
            w = GRID[i % len(GRID)][0]
            spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_paths[i])
            scale_spec = spec.GetAttributeAtPath(prim_paths[i] + ".xformOp:scale")
            s = Gf.Vec3f(w / BASE_W, w / BASE_W, 1.0)
            if scale_spec is None:
                scale_spec = Sdf.AttributeSpec(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3)
                order = spec.GetAttributeAtPath(prim_paths[i] + ".xformOpOrder")
                if order is None:
                    order = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
                    order.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
            scale_spec.default = s

    sim.reset()

    robot = scene["robot"]
    obj = scene["obj"]
    device = robot.device
    body_idx = robot.body_names.index(GRIPPER_LINK)
    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
    fing_dof = [robot.joint_names.index(n) for n in ("joint_left", "joint_right")]

    # -- per-env mass and friction, applied through the physics views ------
    masses = torch.tensor(
        [[GRID[i % len(GRID)][1]] for i in range(NUM_ENVS)], device=obj.device, dtype=torch.float32
    )
    obj.set_masses_index(masses=masses, env_ids=torch.arange(NUM_ENVS, device=obj.device))

    import warp as wp  # noqa: PLC0415

    mats = wp.to_torch(obj.root_view.get_material_properties()).clone()
    for i in range(NUM_ENVS):
        mu = GRID[i % len(GRID)][2]
        mats[i, :, 0] = mu + 0.2  # static
        mats[i, :, 1] = mu        # dynamic
        mats[i, :, 2] = 0.0       # restitution
    obj.root_view.set_material_properties(
        wp.from_torch(mats.contiguous(), dtype=wp.float32),
        wp.from_torch(torch.arange(NUM_ENVS, dtype=torch.int32), dtype=wp.int32),
    )

    start = torch.tensor([_START_POSE[f"joint{i}"] for i in range(1, 7)], device=device)
    q = torch.zeros((NUM_ENVS, robot.num_joints), device=device)
    q[:, arm_dof] = start
    q[:, fing_dof] = 0.045
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    sim.forward()
    robot.update(0.0)

    # Place each object at the TCP, upright, resting on the support pedestal, with a
    # little jitter along the finger axis. The pedestal matters: without it a heavy
    # object free-falls during the ~0.1 s the gripper takes to close, so "fell before
    # the fingers arrived" would be scored as "too heavy to hold". Supporting it during
    # the close isolates the actual payload question, and matches how a real grasp works.
    link_pos = torch.as_tensor(robot.data.body_pos_w[:, body_idx, :], device=device)
    link_quat = torch.as_tensor(robot.data.body_quat_w[:, body_idx, :], device=device)
    tcp = link_pos + quat_apply(link_quat, torch.tensor(TCP_OFFSET, device=device).repeat(NUM_ENVS, 1))
    jitter = (torch.rand((NUM_ENVS, 3), device=device) - 0.5) * torch.tensor([0.006, 0.006, 0.004], device=device)
    obj_pos = tcp + jitter
    obj_pos[:, 2] = SUPPORT_TOP + BASE_H / 2  # rest on the pedestal, not floating
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(NUM_ENVS, 1)
    obj.write_root_state_to_sim(torch.cat([obj_pos, ident, torch.zeros((NUM_ENVS, 6), device=device)], dim=1))

    for _ in range(20):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 400)

    # --- close the gripper (object still supported) -------------------------
    q[:, fing_dof] = 0.0
    robot.set_joint_position_target(q)
    for _ in range(args_cli.settle_steps):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 400)

    def tcp_now() -> torch.Tensor:
        lp = torch.as_tensor(robot.data.body_pos_w[:, body_idx, :], device=device)
        lq = torch.as_tensor(robot.data.body_quat_w[:, body_idx, :], device=device)
        return lp + quat_apply(lq, torch.tensor(TCP_OFFSET, device=device).repeat(NUM_ENVS, 1))

    def obj_pos_now() -> torch.Tensor:
        return torch.as_tensor(obj.data.root_pos_w, device=device)

    grasped = (obj_pos_now() - tcp_now()).norm(dim=1) < 0.05

    # --- carry: lift the shoulder and roll the wrist ------------------------
    target = q.clone()
    for s in range(args_cli.carry_steps):
        f = min(1.0, s / max(args_cli.carry_steps * 0.6, 1))
        target[:, arm_dof[1]] = start[1] + 0.45 * f   # joint2 lift
        target[:, arm_dof[3]] = start[3] - 0.35 * f   # joint4 pitch
        target[:, arm_dof[5]] = start[5] + 1.20 * f   # joint6 roll
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(1 / 400)

    err = (obj_pos_now() - tcp_now()).norm(dim=1)
    held = (err < 0.05) & grasped

    # --- report -------------------------------------------------------------
    rows = []
    for gi, (w, m, mu) in enumerate(GRID):
        idx = [gi + k * len(GRID) for k in range(REPEATS)]
        idx = [i for i in idx if i < NUM_ENVS]
        rows.append({
            "width_m": w,
            "mass_kg": m,
            "dyn_friction": mu,
            "n": len(idx),
            "grasped_after_close": int(sum(int(grasped[i]) for i in idx)),
            "held_after_carry": int(sum(int(held[i]) for i in idx)),
            "mean_final_err_m": float(np.mean([float(err[i]) for i in idx])),
        })

    os.makedirs(args_cli.out_dir, exist_ok=True)
    with open(os.path.join(args_cli.out_dir, "grasp_envelope.json"), "w") as f:
        json.dump({"grid": rows, "repeats": REPEATS, "num_envs": NUM_ENVS}, f, indent=2)

    print("\n" + "=" * 78)
    print("GRASP ENVELOPE  (held = still between the fingers after lift + 69 deg wrist roll)")
    print("=" * 78)
    print(f"{'width':>7} {'mass':>7} {'mu':>5} | {'closed':>7} {'held':>6} | {'err[mm]':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['width_m']:7.3f} {r['mass_kg']:7.2f} {r['dyn_friction']:5.1f} | "
              f"{r['grasped_after_close']:3d}/{r['n']:<3d} {r['held_after_carry']:2d}/{r['n']:<3d} | "
              f"{1000 * r['mean_final_err_m']:8.1f}")

    print("\nheld-rate marginals:")
    for name, vals, key in [("width", WIDTHS, "width_m"), ("mass", MASSES, "mass_kg"), ("friction", FRICTIONS, "dyn_friction")]:
        print(f"  by {name}:")
        for v in vals:
            sel = [r for r in rows if r[key] == v]
            tot = sum(r["n"] for r in sel)
            hl = sum(r["held_after_carry"] for r in sel)
            print(f"    {v:>6}: {hl:3d}/{tot:<3d} ({100 * hl / max(tot, 1):5.1f}%)")


if __name__ == "__main__":
    main()
    simulation_app.close()
