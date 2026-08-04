# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Challenge tasks for the reBot arm.

Each task isolates a manipulation skill the pick-and-place task does not exercise. See
``docs/CHALLENGE_SUITE.md`` for the measured hardware envelope every design respects and
``docs/envs/`` for a reference document per environment.
"""

import gymnasium as gym

from . import agents

_AGENT = f"{agents.__name__}:rl_games_ppo_cfg.yaml"

_TASKS = [
    # precision: tight-tolerance horizontal insertion
    ("Rebot-PrecisionSlot-v0", "precision_slot_env_cfg", "RebotPrecisionSlotEnvCfg"),
    ("Rebot-PrecisionSlot-Play-v0", "precision_slot_env_cfg", "RebotPrecisionSlotEnvCfg_PLAY"),
    ("Rebot-PrecisionSlot-Loose-v0", "precision_slot_env_cfg", "RebotPrecisionSlotLooseEnvCfg"),
    ("Rebot-PrecisionSlot-Tight-v0", "precision_slot_env_cfg", "RebotPrecisionSlotTightEnvCfg"),
    # extrinsic dexterity: reconfigure an ungraspable object before grasping
    ("Rebot-PreGrasp-v0", "pregrasp_env_cfg", "RebotPreGraspEnvCfg"),
    ("Rebot-PreGrasp-Play-v0", "pregrasp_env_cfg", "RebotPreGraspEnvCfg_PLAY"),
    # clutter: constrained retrieval without disturbing neighbours
    ("Rebot-ClutterExtract-v0", "clutter_env_cfg", "RebotClutterExtractEnvCfg"),
    ("Rebot-ClutterExtract-Play-v0", "clutter_env_cfg", "RebotClutterExtractEnvCfg_PLAY"),
    ("Rebot-ClutterExtract-Tight-v0", "clutter_env_cfg", "RebotClutterExtractTightEnvCfg"),
    # diagnostic controls, not rungs of the task: the pre-2026-08-04 fixed row under the
    # strict rule, and the whole pre-2026-08-03 task (topple-only AND the fixed row)
    ("Rebot-ClutterExtract-Fixed-v0", "clutter_env_cfg", "RebotClutterExtractFixedEnvCfg"),
    ("Rebot-ClutterExtract-Lenient-v0", "clutter_env_cfg", "RebotClutterExtractLenientEnvCfg"),
    # articulated object + irreversible ordering
    ("Rebot-DrawerOrder-v0", "drawer_env_cfg", "RebotDrawerOrderEnvCfg"),
    ("Rebot-DrawerOrder-Play-v0", "drawer_env_cfg", "RebotDrawerOrderEnvCfg_PLAY"),
]

for _id, _mod, _cls in _TASKS:
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_mod}:{_cls}",
            "rl_games_cfg_entry_point": _AGENT,
        },
        disable_env_checker=True,
    )
