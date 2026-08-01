# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision variant of the reBot lift task.

Replaces the privileged cube-position observation with the two cameras of the real rig
(D455 over the workspace, D405 on the gripper, see ``..lift.camera_cfg``). The policy sees
proprioception + frozen-ResNet18 features of both views (1000 each); the critic keeps the
full privileged state (asymmetric actor-critic -- the critic is discarded at deployment).
Rewards, terminations, actions and commands are inherited unchanged from the state env.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from ..lift.camera_cfg import WORKSPACE_CAM_CFG, WRIST_CAM_CFG
from ..lift.rebot_lift_env_cfg import RebotCubeLiftEnvCfg

# Training resolution: same 16:9 aspect (and thus FOV) as the real 1280x720 streams, 64x
# fewer pixels. ResNet18 is fully convolutional up to its global pool, so it accepts this
# size directly.
_CAM_WIDTH = 160
_CAM_HEIGHT = 90


@configclass
class VisionObservationsCfg:
    """Asymmetric observations: camera features for the actor, privileged state for the critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations -- no cube state, only what the real robot can sense."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        # Frozen pretrained ResNet18 -> 1000-d features per camera (Isaac Lab runs the full
        # torchvision model incl. classifier head). Weights download on first use (cached in
        # ~/.cache/torch afterwards).
        workspace_features = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("workspace_cam"), "data_type": "rgb", "model_name": "resnet18"},
        )
        wrist_features = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "model_name": "resnet18"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations -- the state env's full policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RebotCubeLiftVisionEnvCfg(RebotCubeLiftEnvCfg):
    """Lift a cube to a commanded pose from camera observations."""

    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- cameras at training resolution; update once per policy step (decimation * dt)
        cam_kwargs = {"width": _CAM_WIDTH, "height": _CAM_HEIGHT, "update_period": self.decimation * self.sim.dt}
        self.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(**cam_kwargs)
        self.scene.wrist_cam = WRIST_CAM_CFG.replace(**cam_kwargs)

        # -- rendering is the bottleneck: far fewer envs than the 4096 of the state env
        self.scene.num_envs = 256
        # -- push neighbor envs back so they stay small in the 90 deg workspace view
        self.scene.env_spacing = 6.0

        # -- soften the inherited action-penalty curriculum. It ramps at 10k policy steps
        # (iter ~417), where the state policy was already lifting -- but with 16x fewer envs
        # the vision policy is still exploring there, and the original 1000x penalty jump
        # collapsed run 1 to reward -65 and froze the policy. Ramp later and only 100x.
        self.curriculum.action_rate.params["weight"] = -1e-2
        self.curriculum.action_rate.params["num_steps"] = 48000
        self.curriculum.joint_vel.params["weight"] = -1e-2
        self.curriculum.joint_vel.params["num_steps"] = 48000


@configclass
class RebotCubeLiftVisionEnvCfg_PLAY(RebotCubeLiftVisionEnvCfg):
    """Small variant for visual inspection and the vision-env test script."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
