# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-randomized variant of the vision lift task (see ``visual_randomization``)."""

from isaaclab.utils.configclass import configclass

from .rebot_lift_vision_env_cfg import RebotCubeLiftVisionEnvCfg, RebotCubeLiftVisionEnvCfg_PLAY
from .visual_randomization import apply_visual_randomization


@configclass
class RebotCubeLiftVisionRandEnvCfg(RebotCubeLiftVisionEnvCfg):
    """Vision lift with randomized appearance, lighting, occlusions and camera mounts."""

    def __post_init__(self):
        super().__post_init__()
        apply_visual_randomization(self)


@configclass
class RebotCubeLiftVisionRandEnvCfg_PLAY(RebotCubeLiftVisionEnvCfg_PLAY):
    """Small randomized variant for visual inspection."""

    def __post_init__(self):
        super().__post_init__()
        apply_visual_randomization(self)
