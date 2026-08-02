# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Can this arm physically execute the slot insertion? (validation rung V4, isolated)

Separates "can it insert" from "can it grasp". The block is placed into the closed gripper
at a pre-insertion pose, then the arm executes ONLY the insertion stroke. If this fails,
no amount of policy learning will solve the task and the fixture geometry has to change.

No IK solver is used. Two joint configurations -- one at the slot mouth, one fully
inserted -- are found by cross-entropy search over the 6 arm joints, scored by forward
kinematics evaluated in the sim itself (write joint state, ``sim.forward()``, read the
gripper link pose). That sidesteps Jacobian handling entirely and cannot silently converge
to an unreachable pose: the achieved TCP error is reported.

.. code-block:: bash

    python scripts/challenge/slot_insertion_probe.py --task Rebot-PrecisionSlot-Play-v0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isolated insertion feasibility probe.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=512, help="CEM population")
parser.add_argument("--cem_iters", type=int, default=60)
parser.add_argument("--stroke_steps", type=int, default=140)
parser.add_argument("--out_dir", type=str, default="logs/analysis/slot_expert")
parser.add_argument("--video", action="store_true", help="record the stroke to logs/videos/")
parser.add_argument("--grip_z", type=float, default=None, help="override the grip height [m]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG

TCP_OFFSET = (-0.0419, 0.0, 0.0)  # measured; see mdp/common.TCP_OFFSET
GRIPPER_LINK = "gripper_end"

# TCP grips the block's upper half: block centre sits on the slot floor at
# z = SLOT_FLOOR_Z + BLOCK_HALF[2] = 0.055, and the walls top out at 0.050, so the fingers
# must ride above them.
GRIP_Z = 0.072  # overridable via --grip_z
MOUTH_X = mdp.SLOT_CENTER[0] - mdp.SLOT_DEPTH / 2                 # 0.210
# The block starts with its nose just inside the mouth, RESTING ON THE SLOT FLOOR. The
# floor only exists inside the slot (x in [0.210, 0.280]), so a start pose outside the
# mouth would leave the block floating 20 mm in mid-air; it then falls to the table during
# the settle and the gripper closes above it. This probe isolates the insertion stroke, so
# the block starts supported and already engaged.
PRE_X = MOUTH_X + 0.008
HOME_X = mdp.SLOT_CENTER[0] + mdp.SUCCESS_DEPTH - mdp.SLOT_DEPTH / 2   # depth == SUCCESS_DEPTH


def main() -> None:
    global GRIP_Z
    if args_cli.grip_z is not None:
        GRIP_Z = args_cli.grip_z
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.video:
        env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=640, height=360)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    block = e.scene["block"]
    n = e.num_envs

    body_idx = robot.body_names.index(GRIPPER_LINK)
    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
    fing_dof = [robot.joint_names.index(x) for x in ("joint_left", "joint_right")]
    lo = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)[arm_dof, 0]
    hi = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)[arm_dof, 1]
    q_default = torch.as_tensor(robot.data.default_joint_pos[0], device=dev).clone()

    offs = torch.tensor(TCP_OFFSET, device=dev).repeat(n, 1)

    left_idx = robot.body_names.index("gripper_left")
    right_idx = robot.body_names.index("gripper_right")

    def fk(q_arm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched TCP position (env-local) and finger-separation axis, for (n, 6) configs."""
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = 0.045  # fingers apart, so the separation axis is well defined
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        bp = torch.as_tensor(robot.data.body_pos_w.torch, device=dev)
        lp = bp[:, body_idx, :]
        lq = torch.as_tensor(robot.data.body_quat_w.torch[:, body_idx, :], device=dev)
        tcp = lp + quat_apply(lq, offs) - e.scene.env_origins
        sep = bp[:, left_idx, :] - bp[:, right_idx, :]
        sep = sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return tcp, sep

    def fk_tcp(q_arm: torch.Tensor) -> torch.Tensor:
        return fk(q_arm)[0]

    def cem(target: torch.Tensor, seed: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """Cross-entropy search for an arm config reaching ``target`` with a usable grasp.

        Position alone is NOT enough. The first version of this probe optimised only the
        TCP position, which left the wrist orientation free -- the fingers then closed
        along the block's 45 mm length instead of across its 30 mm width and squirted it
        backwards out of the slot (measured: the block moved 34 mm backwards and dropped
        onto the table during the close). The cost therefore also requires the finger
        separation axis to lie along world y, i.e. across the slot.
        """
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
        mean, std = seed.clone(), torch.full((6,), 0.45, device=dev)
        best_q, best_p, best_a = seed.clone(), 1e9, 1e9
        for _ in range(args_cli.cem_iters):
            q = (mean + std * torch.randn((n, 6), device=dev)).clamp(lo, hi)
            tcp, sep = fk(q)
            pos_err = (tcp - target).norm(dim=1)
            # |cos| so either finger may be on the +y side
            align_err = 1.0 - (sep @ y_axis).abs()
            cost = pos_err + 0.25 * align_err
            elite = q[cost.argsort()[: max(8, n // 20)]]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.01)
            i = int(cost.argmin())
            if float(cost[i]) < best_p + 0.25 * best_a:
                best_p, best_a, best_q = float(pos_err[i]), float(align_err[i]), q[i].clone()
        return best_q, best_p, best_a

    with torch.inference_mode():
        env.reset()
        seed = q_default[arm_dof].clone()

        t_pre = torch.tensor([PRE_X, 0.0, GRIP_Z], device=dev)
        t_home = torch.tensor([HOME_X, 0.0, GRIP_Z], device=dev)

        print("[probe] searching for the pre-insertion configuration ...")
        q_pre, e_pre, a_pre = cem(t_pre, seed)
        print(f"[probe]   TCP error {e_pre * 1000:.2f} mm, finger-axis misalignment {a_pre:.4f}")
        print("[probe] searching for the fully-inserted configuration ...")
        q_home, e_home, a_home = cem(t_home, q_pre)
        print(f"[probe]   TCP error {e_home * 1000:.2f} mm, finger-axis misalignment {a_home:.4f}")

        reachable = (e_pre < 0.008) and (e_home < 0.008) and (a_pre < 0.1) and (a_home < 0.1)
        if not reachable:
            print("\n[probe] FAIL at the kinematic stage -- the arm cannot place its TCP at "
                  "the required poses. The fixture must move; no controller will fix this.")

        # ---- dynamic stroke: block already held, execute only the insertion -----
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_pre
        q[:, fing_dof] = 0.045
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)

        tcp0 = fk_tcp(q_pre.unsqueeze(0).repeat(n, 1))
        # block centre directly under the TCP grip point, resting at slot-floor height
        bpos = tcp0.clone()
        bpos[:, 2] = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]
        # a little lateral scatter so this is not a single lucky alignment
        bpos[:, 1] += (torch.rand(n, device=dev) - 0.5) * 0.004
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        block.write_root_state_to_sim(
            torch.cat([bpos + e.scene.env_origins, ident, torch.zeros((n, 6), device=dev)], dim=1)
        )

        def act(q_arm: torch.Tensor, close: bool) -> torch.Tensor:
            a = torch.zeros((n, 7), device=dev)
            a[:, :6] = (q_arm - q_default[arm_dof].unsqueeze(0)) / 0.5
            a[:, 6] = -1.0 if close else 1.0
            return a

        frames = []
        cam = e.scene["workspace_cam"] if args_cli.video else None
        if cam is not None:
            eye = torch.tensor([0.60, -0.34, 0.30], device=dev).unsqueeze(0) + e.scene.env_origins
            tgt = torch.tensor([0.245, 0.0, 0.055], device=dev).unsqueeze(0) + e.scene.env_origins
            cam.set_world_poses_from_view(eye, tgt)

        def snap():
            if cam is None:
                return
            out = cam.data.output["rgb"]
            out = getattr(out, "torch", out)
            if callable(out):
                out = out()
            frames.append(out[0, ..., :3].detach().cpu().numpy().astype("uint8"))

        # close immediately -- any open-gripper settle just lets the block topple or slide
        for _ in range(50):
            env.step(act(q_pre.unsqueeze(0).repeat(n, 1), close=True))
            snap()

        bp = torch.as_tensor(block.data.root_pos_w.torch, device=dev) - e.scene.env_origins
        # strict: the block must still be at the grip point in xy AND at slot-floor height,
        # so a block that merely dropped to the table is not counted as grasped
        held = ((bp[:, :2] - tcp0[:, :2]).norm(dim=1) < 0.02) & (bp[:, 2] > mdp.SLOT_FLOOR_Z)
        print(f"[probe] block held at the grip point after closing: {int(held.sum())}/{n}  "
              f"(mean z {float(bp[:, 2].mean()) * 1000:.1f} mm, "
              f"start depth {float(mdp.insertion_depth(e).mean()) * 1000:+.1f} mm)")

        traj = []
        for s in range(args_cli.stroke_steps):
            f = min(1.0, s / (args_cli.stroke_steps * 0.7))
            q_cmd = (1 - f) * q_pre + f * q_home
            a = act(q_cmd.unsqueeze(0).repeat(n, 1), close=True)
            traj.append(a[0].cpu().numpy())
            env.step(a)
            snap()

        depth = mdp.insertion_depth(e)
        lat = mdp.lateral_error(e)
        yaw = mdp.yaw_error(e)
        ok = mdp.is_inserted(e)

        res = {
            "tcp_err_pre_mm": e_pre * 1000,
            "tcp_err_home_mm": e_home * 1000,
            "finger_axis_misalign_pre": a_pre,
            "finger_axis_misalign_home": a_home,
            "kinematically_reachable": bool(reachable),
            "n": int(n),
            "held_before_stroke": int(held.sum()),
            "inserted": int(ok.sum()),
            "insert_rate": float(ok.float().mean()),
            "depth_mm_mean": float(depth.mean()) * 1000,
            "depth_mm_p90": float(torch.quantile(depth, 0.9)) * 1000,
            "lateral_mm_mean": float(lat.mean()) * 1000,
            "yaw_rad_mean": float(yaw.mean()),
            "success_depth_mm": mdp.SUCCESS_DEPTH * 1000,
        }
        os.makedirs(args_cli.out_dir, exist_ok=True)
        with open(os.path.join(args_cli.out_dir, "insertion_probe.json"), "w") as f:
            json.dump(res, f, indent=2)
        np.save(os.path.join(args_cli.out_dir, "insertion_actions.npy"),
                np.stack(traj)[:, None, :])

        if cam is not None and frames:
            import imageio.v2 as imageio  # noqa: PLC0415
            os.makedirs("logs/videos", exist_ok=True)
            vp = "logs/videos/slot_insertion_probe.mp4"
            imageio.mimsave(vp, frames, fps=25, macro_block_size=1)
            print(f"[probe] wrote {vp} ({len(frames)} frames)")

        print("\n" + "=" * 68)
        print("ISOLATED INSERTION PROBE  (block pre-placed in the gripper)")
        print("=" * 68)
        for k, v in res.items():
            print(f"  {k:26s} {v}")
        print("=" * 68)
        if res["insert_rate"] > 0.5:
            print("  VERDICT: the arm CAN execute the insertion stroke. Remaining task")
            print("           difficulty is the grasp and the approach, not the geometry.")
        else:
            print("  VERDICT: the insertion stroke itself fails. Geometry needs changing --")
            print("           check depth_mm_mean against success_depth_mm to see how far it got.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
