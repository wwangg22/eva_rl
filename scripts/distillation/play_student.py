# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a distilled student policy in the vision env.

Rolls the student for a number of complete episodes and reports mean return and lift rate
(cube raised above 0.10 m, the bottom of the commanded goal-height range).

.. code-block:: bash

    python scripts/distillation/play_student.py --checkpoint logs/distillation/<run>/best.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a distilled student policy.")
parser.add_argument("--task", type=str, default="Rebot-Lift-Cube-Vision-Play-v0", help="Vision task (use ...-Vision-Rand-Play-v0 for the randomized scene).")
parser.add_argument("--checkpoint", type=str, required=True, help="Student .pt from train_student.py.")
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import load_student  # noqa: E402

# the vision env's policy group starts with exactly the student proprio: joint_pos(8),
# joint_vel(8), target_pose(7), last_action(7) -- image features follow after
PROPRIO_DIM = 30
LIFT_HEIGHT = 0.10


def grab_frames(cam) -> torch.Tensor:
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[..., :3].detach().to(torch.uint8)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    student = load_student(args_cli.checkpoint, device=device)

    scene = env.unwrapped.scene
    workspace_cam, wrist_cam = scene["workspace_cam"], scene["wrist_cam"]
    env_origin_z = scene.env_origins[:, 2]

    obs, _ = env.reset()
    num_envs = env.unwrapped.num_envs
    returns = torch.zeros(num_envs, device=device)
    max_height = torch.zeros(num_envs, device=device)
    finished_returns, finished_lifts = [], []

    frames_ws, frames_wr = grab_frames(workspace_cam), grab_frames(wrist_cam)
    while len(finished_returns) < args_cli.episodes:
        with torch.inference_mode():
            actions = student.act(frames_ws, frames_wr, obs["policy"][:, :PROPRIO_DIM])
            obs, rew, terminated, truncated, _ = env.step(actions)
        frames_ws, frames_wr = grab_frames(workspace_cam), grab_frames(wrist_cam)

        returns += rew
        cube_z = scene["object"].data.root_pos_w[:, 2] - env_origin_z
        max_height = torch.maximum(max_height, cube_z)
        dones = (terminated | truncated).nonzero(as_tuple=False).flatten()
        for i in dones.tolist():
            finished_returns.append(returns[i].item())
            finished_lifts.append(bool(max_height[i] > LIFT_HEIGHT))
            returns[i] = 0.0
            max_height[i] = 0.0

    n = len(finished_returns)
    mean_return = sum(finished_returns) / n
    lift_rate = sum(finished_lifts) / n
    print(f"\n[eval] {n} episodes: mean return {mean_return:.2f}, lift rate {lift_rate:.1%}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
