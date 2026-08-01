# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose where a trained pick-and-place policy actually moves the objects.

Runs a checkpoint and logs, per step: each object's xy-distance to the basket center,
its height, whether it is held, whether it is over the basket footprint, and the gripper
command. Prints a summary that answers: does the carried object ever cross the basket
footprint? at what height? does the gripper ever open there? where does the hover
position concentrate (azimuth)? Optionally records a workspace video.

.. code-block:: bash

    python scripts/probe_pick_place_policy.py --headless \\
        --checkpoint logs/rl_games/rebot_pick_place/<run>/nn/<ckpt>.pth

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe a pick-and-place checkpoint.")
parser.add_argument("--task", type=str, default="Rebot-PickPlace-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--video", action="store_true", help="Record a workspace-camera video of env 0.")
parser.add_argument("--out_dir", type=str, default="logs/camera_previews/pick_place_probe")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.pick_place import mdp


def main():
    resume_path = os.path.abspath(args_cli.checkpoint)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None
    if args_cli.video:
        from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG

        env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=640, height=360)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    uenv = env.unwrapped
    n = uenv.num_envs
    device = uenv.device
    center = torch.tensor(mdp.BASKET_CENTER, device=device)
    ee = uenv.scene["ee_frame"]

    writer = None
    if args_cli.video:
        import imageio.v2 as imageio

        os.makedirs(args_cli.out_dir, exist_ok=True)
        writer = imageio.get_writer(
            os.path.join(args_cli.out_dir, "probe.mp4"), fps=round(1.0 / uenv.step_dt), macro_block_size=1
        )

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    stats = {}
    for name in mdp.OBJECT_NAMES:
        stats[name] = {
            "min_d": torch.full((n,), 1e9, device=device),
            "steps_held": torch.zeros(n, device=device),
            "steps_held_lifted": torch.zeros(n, device=device),
            "steps_over_fp": torch.zeros(n, device=device),
            "steps_over_fp_held": torch.zeros(n, device=device),
            "open_over_fp": torch.zeros(n, device=device),
            "z_when_near": [],  # heights when d < 0.10
            "xy_rel_hover": [],  # xy offset from basket center while held+lifted
        }

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
        ee_w = ee.data.target_pos_w.torch[..., 0, :]
        gripper_open = actions[:, 6] > 0  # binary gripper: positive command = open
        placed = mdp.placed_mask(uenv)
        for i, name in enumerate(mdp.OBJECT_NAMES):
            s = stats[name]
            pos_w = uenv.scene[name].data.root_pos_w.torch
            pos = pos_w - uenv.scene.env_origins
            d = torch.linalg.norm(pos[:, :2] - center, dim=1)
            held = torch.linalg.norm(pos_w - ee_w, dim=1) < 0.07
            lifted = pos[:, 2] > 0.06
            over_fp = ((pos[:, 0] - center[0]).abs() < 0.05) & ((pos[:, 1] - center[1]).abs() < 0.05)
            s["min_d"] = torch.minimum(s["min_d"], torch.where(placed[:, i], s["min_d"], d))
            s["steps_held"] += held.float()
            s["steps_held_lifted"] += (held & lifted).float()
            s["steps_over_fp"] += (over_fp & ~placed[:, i]).float()
            s["steps_over_fp_held"] += (over_fp & held & ~placed[:, i]).float()
            s["open_over_fp"] += (over_fp & held & gripper_open & ~placed[:, i]).float()
            near = (d < 0.10) & held & ~placed[:, i]
            if near.any():
                s["z_when_near"].append(pos[near, 2].cpu())
            hl = held & lifted & ~placed[:, i]
            if hl.any():
                s["xy_rel_hover"].append((pos[hl, :2] - center).cpu())
        if writer is not None:
            out = uenv.scene["workspace_cam"].data.output["rgb"]
            out = getattr(out, "torch", out)
            if callable(out):
                out = out()
            writer.append_data(out[0, ..., :3].detach().cpu().numpy().astype("uint8"))

    placed = mdp.placed_mask(uenv)
    T = args_cli.steps
    print("\n================ probe summary ================")
    print(f"envs {n}, steps {T}, checkpoint {os.path.basename(resume_path)}")
    for i, name in enumerate(mdp.OBJECT_NAMES):
        s = stats[name]
        z_near = torch.cat(s["z_when_near"]) if s["z_when_near"] else torch.zeros(0)
        xy = torch.cat(s["xy_rel_hover"]) if s["xy_rel_hover"] else torch.zeros(0, 2)
        print(f"\n-- {name} --")
        print(f"  placed at end:            {int(placed[:, i].sum())}/{n} envs")
        print(f"  min xy-dist to basket:    mean {s['min_d'].mean():.3f}, min {s['min_d'].min():.3f} m")
        print(f"  held fraction:            {s['steps_held'].mean() / T:.2%} of steps")
        print(f"  held+lifted fraction:     {s['steps_held_lifted'].mean() / T:.2%}")
        print(f"  over footprint (unplaced): {s['steps_over_fp'].mean() / T:.2%}")
        print(f"  over footprint & held:    {s['steps_over_fp_held'].mean() / T:.2%}")
        print(f"  gripper OPEN over fp:     {s['open_over_fp'].mean() / T:.2%}")
        if len(z_near):
            print(f"  z while held near basket (d<0.10): mean {z_near.mean():.3f}, p10 {z_near.quantile(0.1):.3f},"
                  f" p90 {z_near.quantile(0.9):.3f}")
        if len(xy):
            print(f"  hover offset from basket center: mean x {xy[:, 0].mean():+.3f}, y {xy[:, 1].mean():+.3f}"
                  f" (basket at {mdp.BASKET_CENTER}, robot base at origin)")
    print("===============================================")
    if writer is not None:
        writer.close()
        print(f"video: {os.path.join(args_cli.out_dir, 'probe.mp4')}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
