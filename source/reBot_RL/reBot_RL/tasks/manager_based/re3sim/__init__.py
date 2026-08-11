# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tasks built on a reconstructed real workstation (Re3Sim capture 2026-08-05).

Unlike the other task packages here, the geometry is *measured* rather than invented: object
dimensions and masses come from calipers and a kitchen scale, and the scene's appearance comes
from a 3D Gaussian-splat reconstruction of the user's actual desk.

See ``docs/envs/re3sim/`` for the capture, the measured constraints that forced the design,
and the progress log.
"""

import gymnasium as gym

from . import agents

_AGENT = f"{agents.__name__}:rl_games_ppo_cfg.yaml"

_TASKS = [
    ("Rebot-Workstation-PickPlace1-v0", "workstation_env_cfg",
     "RebotWorkstationPickPlace1EnvCfg"),
    ("Rebot-Workstation-PickPlace1-Play-v0", "workstation_env_cfg",
     "RebotWorkstationPickPlace1EnvCfg_PLAY"),
    # harder rung: disturbing the clutter ends the episode. Not for development.
    ("Rebot-Workstation-PickPlace1-Strict-v0", "workstation_env_cfg",
     "RebotWorkstationPickPlace1StrictEnvCfg"),
    # ⭐ Same task WITH the two cameras a real rig has, for policies that only see pixels.
    # Separate ids rather than a flag: a CameraCfg in the scene creates one render product per
    # env, which the state-based task at 1024 envs neither wants nor can afford. These also
    # clone the gaussian desk per env, which any multi-env RENDER needs and no state run does.
    ("Rebot-Workstation-PickPlace1-Vision-v0", "workstation_vision_env_cfg",
     "RebotWorkstationPickPlace1VisionEnvCfg"),
    ("Rebot-Workstation-PickPlace1-Vision-Play-v0", "workstation_vision_env_cfg",
     "RebotWorkstationPickPlace1VisionEnvCfg_PLAY"),
    # ⭐ The vision task under VISUAL domain randomisation (docs/envs/re3sim/05_VISUAL_DR.md):
    # per-env camera pose/roll/focal jitter, lighting, backdrop, distractors. Appearance
    # only — dynamics identical to -Vision-v0, so the expert's numbers carry over.
    ("Rebot-Workstation-PickPlace1-VisionDR-v0", "workstation_vision_dr_env_cfg",
     "RebotWorkstationPickPlace1VisionDREnvCfg"),
    ("Rebot-Workstation-PickPlace1-VisionDR-Play-v0", "workstation_vision_dr_env_cfg",
     "RebotWorkstationPickPlace1VisionDREnvCfg_PLAY"),
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
