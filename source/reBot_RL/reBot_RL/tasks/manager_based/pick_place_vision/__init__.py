# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision (two-camera) variant of the pick-and-place v1 task, for champion->student
distillation (reBot_ACT/docs/experiments/EXP08_visual_policy.md)."""

import gymnasium as gym

from ..pick_place import agents

gym.register(
    id="Rebot-PickPlace-Vision-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_vision_env_cfg:RebotPickPlaceVisionEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
