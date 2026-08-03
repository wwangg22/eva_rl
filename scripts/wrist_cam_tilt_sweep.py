# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist-camera tilt sweep for Big Will's mount review.

Spawns ONE env with N wrist cameras at the current D405 mount position, each with a
different tilt (rotation about the camera's local pitch axis composed onto the current
mount rotation, applied via the spawn-time OffsetCfg -- the exact path the training
config uses). Saves one still per tilt plus summary.txt with the camera->grip
distances and per-tilt aim geometry. Frames are checked to differ numerically.

.. code-block:: bash

    python scripts/wrist_cam_tilt_sweep.py --headless

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wrist camera tilt sweep.")
parser.add_argument("--task", type=str, default="Rebot-PickPlace-Play-v1")
parser.add_argument("--out_dir", type=str, default="logs/camera_previews/wrist_tilt_sweep")
parser.add_argument("--tilts", type=float, nargs="+", default=[-60.0, -30.0, 30.0, 60.0],
                    help="tilt angles in degrees, applied about EACH camera-local axis (x/y/z); a 0-deg reference is always included")
parser.add_argument("--axes", type=str, nargs="+", default=["x", "y", "z"], choices=["x", "y", "z"],
                    help="camera-local axes to sweep")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG


def quat_mul_xyzw(a, b):
    """Hamilton product, xyzw tuples (Isaac Lab 3.0 quat layout)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def grab_frame(cam) -> np.ndarray:
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[0, ..., :3].detach().cpu().numpy().astype("uint8")


def cam_name(axis: str, tilt: float) -> str:
    return f"wrist_cam_{axis}_{'m' if tilt < 0 else 'p'}{round(abs(tilt)):03d}"


_AXIS_VEC = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    base_rot = WRIST_CAM_CFG.offset.rot  # xyzw, opengl convention (look -Z, up +Y, right +X)
    specs = [("x", 0.0)] + [(ax, t) for ax in args_cli.axes for t in args_cli.tilts]
    for axis, tilt in specs:
        half = math.radians(tilt) / 2.0
        v = _AXIS_VEC[axis]
        tilt_q = (math.sin(half) * v[0], math.sin(half) * v[1], math.sin(half) * v[2], math.cos(half))
        rot = quat_mul_xyzw(base_rot, tilt_q)  # rotate about the camera-LOCAL axis
        cfg = WRIST_CAM_CFG.replace(
            prim_path=WRIST_CAM_CFG.prim_path.replace("WristCam", f"WristCam{cam_name(axis, tilt)}"),
            offset=CameraCfg.OffsetCfg(pos=WRIST_CAM_CFG.offset.pos, rot=rot, convention="opengl"),
            width=640,
            height=360,
        )
        setattr(env_cfg.scene, cam_name(axis, tilt), cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    robot = env.unwrapped.scene["robot"]
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    with torch.inference_mode():
        env.reset()
        for _ in range(25):  # settle at the start pose
            env.step(torch.zeros(1, 7, device=env.unwrapped.device))

        cam0 = env.unwrapped.scene[cam_name(*specs[0])]
        cam_pos = cam0.data.pos_w[0].to(torch.float32)
        tcp = env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0].to(torch.float32)
        log(f"[mount] camera world pos: ({cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f})")
        log(f"[mount] ee_frame target (TCP) world pos: ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})")
        log(f"[mount] camera -> TCP distance: {torch.linalg.norm(tcp - cam_pos):.4f} m")
        for name in [n for n in robot.body_names if any(k in n.lower() for k in ("finger", "pad", "grip"))]:
            idx = robot.body_names.index(name)
            p = robot.data.body_pos_w.torch()[0, idx] if callable(getattr(robot.data.body_pos_w, "torch", None)) \
                else robot.data.body_pos_w[0, idx]
            log(f"[mount] camera -> {name}: {torch.linalg.norm(p.to(torch.float32) - cam_pos):.4f} m")
        log("[mount] D405 note: min depth range is ~0.07-0.10 m -- anything closer is out of focus/range on hardware.")

        prev = None
        for axis, tilt in specs:
            cam = env.unwrapped.scene[cam_name(axis, tilt)]
            # aim geometry: optical axis = camera-frame -Z (opengl) in world coords
            q = cam.data.quat_w_opengl[0].to(torch.float32)  # xyzw
            x, y, z, w = q
            fwd_w = torch.stack([  # R(q) @ (0,0,-1)
                -(2 * (x * z + w * y)),
                -(2 * (y * z - w * x)),
                -(1 - 2 * (x * x + y * y)),
            ])
            to_tcp = tcp - cam.data.pos_w[0].to(torch.float32)
            cosang = torch.dot(fwd_w, to_tcp) / (torch.linalg.norm(fwd_w) * torch.linalg.norm(to_tcp))
            off_deg = float(torch.rad2deg(torch.arccos(cosang.clamp(-1.0, 1.0))))
            frame = grab_frame(cam)
            fname = f"tilt_{axis}_{'m' if tilt < 0 else 'p'}{abs(tilt):04.1f}deg.png"
            imageio.imwrite(os.path.join(args_cli.out_dir, fname), frame)
            diff = float(np.abs(frame.astype(np.int16) - prev).mean()) if prev is not None else float("nan")
            log(f"[axis {axis} tilt {tilt:+6.1f} deg] {fname}  optical-axis-to-TCP offset: {off_deg:6.1f} deg"
                f"  frame std: {frame.std():6.1f}  (pixel diff vs previous: {diff:.1f})")
            if frame.std() < 2.0:
                log(f"[WARN] axis {axis} tilt {tilt}: frame is near-uniform (std {frame.std():.2f}) -- "
                    "likely inside geometry or looking at empty space")
            prev = frame.astype(np.int16)

    with open(os.path.join(args_cli.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] {len(args_cli.tilts)} stills + summary.txt -> {args_cli.out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
