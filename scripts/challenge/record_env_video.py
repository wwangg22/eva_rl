# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a video of any reBot env so the scene can be inspected visually.

Smoke tests confirm shapes and predicates but say nothing about whether the scene *looks*
right -- an object spawned inside a wall, a fixture at the wrong height, or a gripper
clipping through geometry all pass a shape check. This renders the workspace camera and
writes an mp4.

Modes:

* ``settle``  -- zero actions; shows the authored scene and where things come to rest.
* ``random``  -- small random arm actions; shows the workspace being swept, and whether
  the arm can collide with the fixture.
* ``replay``  -- replay a saved action sequence (``.npy``, shape (T, num_envs, act_dim)),
  which is how a scripted expert's rollout gets recorded.

.. code-block:: bash

    python scripts/challenge/record_env_video.py --task Rebot-PrecisionSlot-Play-v0
    python scripts/challenge/record_env_video.py --task Rebot-PrecisionSlot-Tight-v0 --mode random
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a video of a reBot env.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--mode", type=str, default="settle", choices=["settle", "random", "replay"])
parser.add_argument("--replay_file", type=str, default=None, help="npy of actions for --mode replay")
parser.add_argument("--out_dir", type=str, default="logs/videos")
parser.add_argument("--cam_width", type=int, default=640)
parser.add_argument("--cam_height", type=int, default=360)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--tile", action="store_true", default=True, help="tile up to 4 envs into one frame")
# The repo's WORKSPACE_CAM_CFG pose was framed for the lift/pick-place scene and puts the
# slot fixture out of shot, so the camera is re-aimed with a look-at instead.
parser.add_argument("--cam_eye", type=float, nargs=3, default=[0.62, -0.42, 0.34])
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.235, -0.045, 0.055])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# nothing renders without this
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG


def grab(cam, env_idx: int) -> np.ndarray:
    """One RGB frame for a given env. The ``.torch`` accessor may be attr or callable."""
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[env_idx, ..., :3].detach().cpu().numpy().astype("uint8")


def compose(cam, n_tile: int) -> np.ndarray:
    """Env 0 alone, or a 2x2 tile so per-env randomization is visible at a glance."""
    if n_tile <= 1:
        return grab(cam, 0)
    frames = [grab(cam, i) for i in range(n_tile)]
    while len(frames) < 4:
        frames.append(np.zeros_like(frames[0]))
    top = np.concatenate(frames[0:2], axis=1)
    bot = np.concatenate(frames[2:4], axis=1)
    return np.concatenate([top, bot], axis=0)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(
        width=args_cli.cam_width, height=args_cli.cam_height
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    cam = e.scene["workspace_cam"]
    act_dim = e.action_manager.total_action_dim
    n_tile = min(4, e.num_envs) if args_cli.tile else 1

    # aim the camera at the working volume, per env (the cfg pose is env-relative but the
    # look-at API takes world coordinates, so the env origin has to be added back in)
    eye = torch.tensor(args_cli.cam_eye, device=e.device).unsqueeze(0) + e.scene.env_origins
    tgt = torch.tensor(args_cli.cam_target, device=e.device).unsqueeze(0) + e.scene.env_origins
    cam.set_world_poses_from_view(eye, tgt)

    replay = None
    if args_cli.mode == "replay":
        if not args_cli.replay_file:
            raise SystemExit("--mode replay needs --replay_file")
        replay = torch.as_tensor(np.load(args_cli.replay_file), dtype=torch.float32, device=e.device)
        print(f"[rec] replaying {replay.shape[0]} steps from {args_cli.replay_file}")

    os.makedirs(args_cli.out_dir, exist_ok=True)
    path = os.path.join(args_cli.out_dir, f"{args_cli.task}_{args_cli.mode}.mp4")

    frames = []
    with torch.inference_mode():
        env.reset()
        n_steps = replay.shape[0] if replay is not None else args_cli.steps
        for t in range(n_steps):
            if replay is not None:
                a = replay[t]
                if a.shape[0] != e.num_envs:
                    a = a[:1].expand(e.num_envs, -1)
            elif args_cli.mode == "random":
                # small, slow actions: large ones just fling the arm and show nothing
                a = 0.25 * torch.randn((e.num_envs, act_dim), device=e.device)
                a[:, -1] = 1.0  # keep the gripper open
            else:
                a = torch.zeros((e.num_envs, act_dim), device=e.device)
            env.step(a)
            frames.append(compose(cam, n_tile))

    imageio.mimsave(path, frames, fps=args_cli.fps, macro_block_size=1)
    print(f"\n[rec] wrote {path}  ({len(frames)} frames, {len(frames) / args_cli.fps:.1f} s, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}, {n_tile} env(s) tiled)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
