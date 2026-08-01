# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect teacher-labeled episodes for vision distillation.

Runs the state-based rl_games teacher in the state lift env with the two rig cameras bolted
on (at student/training resolution) and, by default, the visual domain randomization from
``lift_vision.visual_randomization``. Saves complete episodes -- camera frames, privileged
state, and the teacher's actions -- as ``.pt`` shards plus a ``meta.json`` describing the
layout. ``scripts/distillation/train_student.py`` consumes this dataset.

.. code-block:: bash

    python scripts/distillation/collect_episodes.py \\
        --checkpoint logs/rl_games/rebot_lift/<run>/nn/rebot_lift.pth --episodes 500

DAgger: pass ``--student_checkpoint`` to execute the current student's actions while still
recording the teacher's actions as labels (corrects covariate shift; run BC first).

Data layout (see also meta.json):
    episode = {
        "workspace": (T, H, W, 3) uint8, "wrist": (T, H, W, 3) uint8,
        "state": (T, 33) f32   # joint_pos[0:8] joint_vel[8:16] object_pos[16:19] target[19:26] last_action[26:33]
        "action": (T, 7) f32,  # teacher action for the state at the same index
        "reward": (T,) f32,
    }
    student proprio = state[:, PROPRIO_INDICES]  (drops the privileged object position)
"""

import argparse
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect teacher episodes for distillation.")
parser.add_argument("--task", type=str, default="Rebot-Lift-Cube-Play-v0", help="State-based task the teacher was trained on.")
parser.add_argument("--checkpoint", type=str, required=True, help="rl_games .pth checkpoint of the state-based teacher.")
parser.add_argument("--episodes", type=int, default=500, help="Complete episodes to save.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--width", type=int, default=160)
parser.add_argument("--height", type=int, default=90)
parser.add_argument("--out_dir", type=str, default=None, help="Default: data/distillation/<timestamp>/")
parser.add_argument("--episodes_per_shard", type=int, default=50)
parser.add_argument("--no_randomization", action="store_true", help="Plain scene: no visual randomization at all.")
parser.add_argument("--no_distractors", action="store_true", help="Keep randomization but drop the occluding clutter.")
parser.add_argument("--keep_dex_cube", action="store_true", help="Keep the textured DexCube instead of the color-randomized cuboid.")
parser.add_argument("--stochastic", action="store_true", help="Sample teacher actions instead of taking the deterministic mean.")
parser.add_argument("--student_checkpoint", type=str, default=None, help="DAgger: student drives, teacher labels.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG, WRIST_CAM_CFG
from reBot_RL.tasks.manager_based.lift_vision.visual_randomization import apply_visual_randomization

# student proprio = full state minus the privileged object position (indices 16:19)
PROPRIO_INDICES = list(range(0, 16)) + list(range(19, 33))


def grab_frames(cam) -> torch.Tensor:
    """Current rgb frames of all envs as a (N, H, W, 3) uint8 cpu tensor."""
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[..., :3].detach().to(torch.uint8).cpu()


class ShardWriter:
    """Accumulates episodes and writes them to numbered .pt shards."""

    def __init__(self, out_dir: str, episodes_per_shard: int):
        self.out_dir = out_dir
        self.episodes_per_shard = episodes_per_shard
        self.buffer: list[dict] = []
        self.num_shards = 0
        self.num_episodes = 0

    def add(self, episode: dict):
        self.buffer.append(episode)
        self.num_episodes += 1
        if len(self.buffer) >= self.episodes_per_shard:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        path = os.path.join(self.out_dir, f"shard_{self.num_shards:05d}.pt")
        torch.save(self.buffer, path)
        print(f"[collect] wrote {len(self.buffer)} episodes -> {path}")
        self.buffer = []
        self.num_shards += 1


def load_teacher(env, task: str, checkpoint: str):
    """Create an rl_games player for the state-based teacher (mirrors scripts/record_cameras.py)."""
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    return env, agent


def main():
    resume_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # cameras at student resolution; same spacing as the vision env so views match training
    cam_kwargs = {
        "width": args_cli.width,
        "height": args_cli.height,
        "update_period": env_cfg.decimation * env_cfg.sim.dt,
    }
    env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(**cam_kwargs)
    env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(**cam_kwargs)
    env_cfg.scene.env_spacing = 6.0
    if not args_cli.no_randomization:
        apply_visual_randomization(
            env_cfg, swap_cube=not args_cli.keep_dex_cube, distractors=not args_cli.no_distractors
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env, teacher = load_teacher(env, args_cli.task, resume_path)

    student = None
    if args_cli.student_checkpoint is not None:
        from student_model import load_student  # local import: only needed for DAgger

        student = load_student(args_cli.student_checkpoint, device=env.unwrapped.device)

    out_dir = args_cli.out_dir or os.path.join("data", "distillation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    writer = ShardWriter(out_dir, args_cli.episodes_per_shard)

    num_envs = env.unwrapped.num_envs
    workspace_cam = env.unwrapped.scene["workspace_cam"]
    wrist_cam = env.unwrapped.scene["wrist_cam"]

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = teacher.get_batch_size(obs, 1)
    if teacher.is_rnn:
        teacher.init_rnn()

    # Per-env episode buffers. Each env's FIRST episode is discarded: right after the initial
    # reset the camera frames can lag the physics state by a frame, so recording starts only
    # from each env's first auto-reset onward, where frames grabbed post-step stay in sync.
    buffers = [{"workspace": [], "wrist": [], "state": [], "action": [], "reward": []} for _ in range(num_envs)]
    recording = [False] * num_envs
    frames_ws = grab_frames(workspace_cam)
    frames_wr = grab_frames(wrist_cam)

    print(f"[collect] target: {args_cli.episodes} episodes from {num_envs} envs -> {out_dir}")
    while writer.num_episodes < args_cli.episodes:
        with torch.inference_mode():
            teacher_actions = teacher.get_action(teacher.obs_to_torch(obs), is_deterministic=not args_cli.stochastic)
            if teacher_actions.dim() == 1:  # rl_games squeezes the batch dim for single-env
                teacher_actions = teacher_actions.unsqueeze(0)
            if student is not None:
                proprio = obs[:, PROPRIO_INDICES]
                exec_actions = student.act(
                    frames_ws.to(env.unwrapped.device), frames_wr.to(env.unwrapped.device), proprio
                )
            else:
                exec_actions = teacher_actions
            next_obs, rew, dones, _ = env.step(exec_actions)
        if isinstance(next_obs, dict):
            next_obs = next_obs["obs"]

        state_cpu = obs.detach().cpu()
        label_cpu = teacher_actions.detach().cpu()
        rew_cpu = rew.detach().cpu()
        for i in range(num_envs):
            if recording[i]:
                buffers[i]["workspace"].append(frames_ws[i])
                buffers[i]["wrist"].append(frames_wr[i])
                buffers[i]["state"].append(state_cpu[i])
                buffers[i]["action"].append(label_cpu[i])
                buffers[i]["reward"].append(rew_cpu[i])

        # frames rendered for the post-step state, i.e. the obs used at the next iteration
        frames_ws = grab_frames(workspace_cam)
        frames_wr = grab_frames(wrist_cam)

        done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
        for i in done_ids:
            if recording[i] and len(buffers[i]["state"]) > 0:
                episode = {
                    "workspace": torch.stack(buffers[i]["workspace"]),
                    "wrist": torch.stack(buffers[i]["wrist"]),
                    "state": torch.stack(buffers[i]["state"]).float(),
                    "action": torch.stack(buffers[i]["action"]).float(),
                    "reward": torch.stack(buffers[i]["reward"]).float(),
                }
                writer.add(episode)
                if writer.num_episodes % 10 == 0:
                    print(f"[collect] {writer.num_episodes}/{args_cli.episodes} episodes")
            buffers[i] = {"workspace": [], "wrist": [], "state": [], "action": [], "reward": []}
            recording[i] = True

        obs = next_obs

    writer.flush()
    meta = {
        "task": args_cli.task,
        "teacher_checkpoint": resume_path,
        "student_checkpoint": args_cli.student_checkpoint,
        "num_episodes": writer.num_episodes,
        "num_shards": writer.num_shards,
        "image_hw": [args_cli.height, args_cli.width],
        "state_dim": 33,
        "action_dim": int(env.unwrapped.single_action_space.shape[0]) if hasattr(env.unwrapped, "single_action_space") else 7,
        "proprio_indices": PROPRIO_INDICES,
        "state_layout": "joint_pos[0:8] joint_vel[8:16] object_pos[16:19] target_pose[19:26] last_action[26:33]",
        "randomization": not args_cli.no_randomization,
        "distractors": not (args_cli.no_randomization or args_cli.no_distractors),
        "stochastic_teacher": args_cli.stochastic,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[collect] done: {writer.num_episodes} episodes, meta -> {os.path.join(out_dir, 'meta.json')}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
