# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-completion evaluation for the pick-and-place task -- the >90% gate metric.

Runs a trained rl_games checkpoint for full-length episodes and reports:

- **success rate**: both objects inside the basket at the end of the episode and no
  object ever dropped off the table (the headline metric),
- per-object placed rates, ever-both-placed rate, drop rate,
- mean time to first both-placed among successes.

The time-out termination is disabled so every episode runs exactly its nominal length
and the final state can be read directly from the scene (auto-reset would otherwise
destroy it). A drop termination still auto-resets that env; the whole round counts as a
failure for it.

.. code-block:: bash

    python scripts/evaluate_pick_place.py --headless \\
        --checkpoint logs/rl_games/rebot_pick_place/<run>/nn/<ckpt>.pth

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate task completion for the pick-and-place task.")
parser.add_argument("--task", type=str, default="Rebot-PickPlace-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rl_games .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512, help="Rounded up to a multiple of num_envs.")
parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of the deterministic mean.")
parser.add_argument("--no_perturb", action="store_true", help="Disable the mid-episode object-nudge events.")
parser.add_argument(
    "--no_prestage", action="store_true", help="Full task only: no episodes start with an object pre-placed."
)
parser.add_argument("--json_out", type=str, default=None, help="Optional path to also write the metrics as JSON.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
from datetime import datetime

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.pick_place import mdp


def main():
    resume_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # run fixed-length episodes: read the final state from the scene instead of racing
    # the auto-reset
    steps_per_episode = int(round(env_cfg.episode_length_s / (env_cfg.decimation * env_cfg.sim.dt)))
    env_cfg.terminations.time_out = None
    if args_cli.no_perturb:
        env_cfg.events.nudge_object_a = None
        env_cfg.events.nudge_object_b = None
    if args_cli.no_prestage:
        env_cfg.events.prestage_object_a.params["prob"] = 0.0
        env_cfg.events.prestage_object_b.params["prob"] = 0.0
        if hasattr(env_cfg.events, "lying_object_a"):
            env_cfg.events.lying_object_a.params["prob"] = 0.0

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

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    rounds = math.ceil(args_cli.episodes / num_envs)
    step_dt = env.unwrapped.step_dt

    totals = {
        "episodes": 0,
        "success": 0,
        "placed_a_final": 0,
        "placed_b_final": 0,
        "ever_both_placed": 0,
        "dropped": 0,
        "attempts": 0,
        "failed_grasps": 0,
        "clean_grasp_episodes": 0,
        "clean_grasp_successes": 0,
        "fail_a_elevated": 0,
        "fail_a_lying": 0,
        "fail_a_upright": 0,
        "fail_b_elevated": 0,
        "fail_b_lying": 0,
        "fail_b_upright": 0,
    }
    success_times = []

    for round_idx in range(rounds):
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ever_both = torch.zeros_like(dropped)
        first_both_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        # grasp-attempt bookkeeping: an attempt is a close edge (gripper action < 0); it
        # failed if the gripper reopens without any object having been lifted (z > 6 cm)
        # or newly placed while closed. Ideal episode: zero failed attempts.
        closed_prev = torch.zeros_like(dropped)
        productive = torch.zeros_like(dropped)
        placed_at_close = torch.zeros(num_envs, dtype=torch.long, device=device)
        attempts = torch.zeros(num_envs, dtype=torch.long, device=device)
        failed_grasps = torch.zeros_like(attempts)

        for step in range(steps_per_episode):
            with torch.inference_mode():
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=not args_cli.stochastic)
                obs, _, dones, _ = env.step(actions)
            dropped |= dones.to(device=device, dtype=torch.bool)
            placed = mdp.placed_mask(env.unwrapped)
            both = placed.all(dim=1)
            newly = both & (first_both_step < 0)
            first_both_step[newly] = step
            ever_both |= both

            closed = actions[:, 6].to(device=device) < 0
            placed_count = placed.sum(dim=1).long()
            new_close = closed & ~closed_prev
            attempts += new_close.long()
            productive &= ~new_close
            placed_at_close = torch.where(new_close, placed_count, placed_at_close)
            heights = torch.stack([mdp.object_pos_local(env.unwrapped, n)[:, 2] for n in mdp.OBJECT_NAMES], dim=1)
            productive |= closed & ((heights.max(dim=1).values > 0.06) | (placed_count > placed_at_close))
            failed_grasps += (closed_prev & ~closed & ~productive).long()
            closed_prev = closed
        failed_grasps += (closed_prev & ~productive).long()

        placed = mdp.placed_mask(env.unwrapped)
        success = placed.all(dim=1) & ~dropped
        # failure diagnosis: final state of each unplaced object in failed episodes.
        # Quats are XYZW (Isaac Lab 3.0); world-z of the body z-axis = 1 - 2(qx^2+qy^2).
        for i, name in enumerate(mdp.OBJECT_NAMES):
            unplaced = ~success & ~placed[:, i]
            pos = mdp.object_pos_local(env.unwrapped, name)
            quat = env.unwrapped.scene[name].data.root_quat_w.torch
            up_z = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)
            elevated = unplaced & (pos[:, 2] > 0.05)
            lying = unplaced & ~elevated & (up_z.abs() < 0.7)
            upright = unplaced & ~elevated & ~lying
            key = name[-1]
            totals[f"fail_{key}_elevated"] += int(elevated.sum())
            totals[f"fail_{key}_lying"] += int(lying.sum())
            totals[f"fail_{key}_upright"] += int(upright.sum())
        totals["episodes"] += num_envs
        totals["success"] += int(success.sum())
        totals["placed_a_final"] += int((placed[:, 0] & ~dropped).sum())
        totals["placed_b_final"] += int((placed[:, 1] & ~dropped).sum())
        totals["ever_both_placed"] += int((ever_both & ~dropped).sum())
        totals["dropped"] += int(dropped.sum())
        totals["attempts"] += int(attempts.sum())
        totals["failed_grasps"] += int(failed_grasps.sum())
        totals["clean_grasp_episodes"] += int((failed_grasps == 0).sum())
        totals["clean_grasp_successes"] += int((success & (failed_grasps == 0)).sum())
        success_times += (first_both_step[success].float() * step_dt).tolist()
        print(
            f"[round {round_idx + 1}/{rounds}] success {int(success.sum())}/{num_envs},"
            f" dropped {int(dropped.sum())}, failed grasps {int(failed_grasps.sum())}"
        )

    n = totals["episodes"]
    metrics = {
        "checkpoint": resume_path,
        "task": args_cli.task,
        "episodes": n,
        "deterministic": not args_cli.stochastic,
        "success_rate": totals["success"] / n,
        "placed_a_rate": totals["placed_a_final"] / n,
        "placed_b_rate": totals["placed_b_final"] / n,
        "ever_both_placed_rate": totals["ever_both_placed"] / n,
        "drop_rate": totals["dropped"] / n,
        "mean_time_to_completion_s": sum(success_times) / len(success_times) if success_times else None,
        "perturbation_enabled": not args_cli.no_perturb,
        "prestage_enabled": not args_cli.no_prestage,
        "grasp_attempts_mean": totals["attempts"] / n,
        "failed_grasps_mean": totals["failed_grasps"] / n,
        "clean_grasp_rate": totals["clean_grasp_episodes"] / n,
        "success_and_clean_rate": totals["clean_grasp_successes"] / n,
        "failure_states": {k: totals[k] for k in totals if k.startswith("fail_")},
    }
    print("\n========== pick-and-place task completion ==========")
    print(f"episodes:                {n}")
    print(f"SUCCESS RATE:            {metrics['success_rate']:.1%}   (gate: >90%)")
    print(f"object_a placed (final): {metrics['placed_a_rate']:.1%}")
    print(f"object_b placed (final): {metrics['placed_b_rate']:.1%}")
    print(f"ever both placed:        {metrics['ever_both_placed_rate']:.1%}")
    print(f"dropped off table:       {metrics['drop_rate']:.1%}")
    if metrics["mean_time_to_completion_s"] is not None:
        print(f"mean time to completion: {metrics['mean_time_to_completion_s']:.2f} s")
    print(f"perturbation:            {'ON' if metrics['perturbation_enabled'] else 'off'}")
    print(f"prestaging:              {'on' if metrics['prestage_enabled'] else 'OFF (full task)'}")
    print(f"grasp attempts / ep:     {metrics['grasp_attempts_mean']:.2f}")
    print(f"FAILED grasps / ep:      {metrics['failed_grasps_mean']:.2f}   (target: ~0)")
    print(f"clean-grasp episodes:    {metrics['clean_grasp_rate']:.1%}   (no failed grasp)")
    print(f"success AND clean:       {metrics['success_and_clean_rate']:.1%}")
    fs = metrics["failure_states"]
    print(
        f"failed-ep obj states:    A lying {fs['fail_a_lying']}, upright {fs['fail_a_upright']},"
        f" held/air {fs['fail_a_elevated']} | B lying {fs['fail_b_lying']},"
        f" upright {fs['fail_b_upright']}, held/air {fs['fail_b_elevated']}"
    )
    print("====================================================")

    json_path = args_cli.json_out or os.path.join(
        "logs", "evaluations", f"pick_place_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics written to {json_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
