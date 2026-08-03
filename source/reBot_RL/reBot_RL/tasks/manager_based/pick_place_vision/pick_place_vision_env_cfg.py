# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision variant of pick-and-place v1 for champion->student distillation (EXP08).

Adds the two cameras of the real rig (D455 over the workspace, D405 on the gripper --
see ``..lift.camera_cfg``) to the v1 play env and splits observations into two groups:

* ``policy``  -- the unchanged privileged 41-D state the 91.4% champion was trained on.
  Consumed ONLY by the teacher at collection/DAgger time.
* ``student`` -- raw camera pixels + 23-D proprio (joint_pos_rel 8, joint_vel_rel 8,
  last_action 7). This is everything the deployed policy is allowed to see: NO object
  poses, NO placed flags, NO basket center (the no-privileged-info contract,
  reBot_ACT/docs/experiments/EXP08_visual_policy.md section 4).

Images are raw uint8 (``normalize=False``): the stock rgb normalization subtracts the
PER-BATCH image mean, which changes with num_envs -- normalization belongs in the
trainer, where it is fixed and reproducible.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ..lift.camera_cfg import WORKSPACE_CAM_CFG, WRIST_CAM_CFG
from ..pick_place import mdp
from ..pick_place.pick_place_v1_env_cfg import ObservationsV1Cfg, RebotPickPlaceV1EnvCfg_PLAY

# Training resolution: same 16:9 aspect (and thus FOV) as the real 1280x720 streams.
_CAM_WIDTH = 160
_CAM_HEIGHT = 90

# Observation funcs that must NEVER appear in the student group. The Gate A smoke test
# audits the cfg against this list (EXP08 section 4 enforcement).
PRIVILEGED_OBS_FUNC_NAMES = (
    "objects_canonical",
    "basket_center_xy",
    "object_pose_in_robot_root_frame",
    "objects_placed_flags",
)

# Student proprio layout (matches the corresponding slices of the 41-D privileged obs:
# [0:16] joint_pos+joint_vel, [34:41] last_action).
STUDENT_PROPRIO_DIM = 8 + 8 + 7


@configclass
class VisionObservationsCfg:
    """Two groups: privileged teacher state + deployable student obs."""

    # teacher side: byte-identical to the state env the champion was trained on
    policy: ObservationsV1Cfg.PolicyCfg = ObservationsV1Cfg.PolicyCfg()

    @configclass
    class StudentCfg(ObsGroup):
        """What the real robot can sense: encoders, its own last action, two cameras."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        workspace_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("workspace_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            # images and vectors cannot concatenate: the group is a dict of named terms
            self.concatenate_terms = False

    student: StudentCfg = StudentCfg()


@configclass
class RebotPickPlaceVisionEnvCfg_PLAY(RebotPickPlaceV1EnvCfg_PLAY):
    """v1 play env (task randomization on, dynamics diversity stripped) + cameras."""

    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # cameras at training resolution, one render per policy step (decimation * dt)
        cam_kwargs = {"width": _CAM_WIDTH, "height": _CAM_HEIGHT, "update_period": self.decimation * self.sim.dt}
        self.scene.wrist_cam = WRIST_CAM_CFG.replace(**cam_kwargs)
        self.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(**cam_kwargs)
        # the D455 sits 0.9 m back with a 90 deg HFOV: push neighbor envs out of view
        # (2.0 m spacing puts them squarely in frame; 6.0 proven in lift_vision)
        self.scene.env_spacing = 6.0
