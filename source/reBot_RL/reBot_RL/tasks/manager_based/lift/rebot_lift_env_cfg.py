# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place (lift-to-pose) task for the RS-rebot-dev-arm.

Reuses Isaac Lab's ``manipulation.lift`` MDP -- reach the cube, grasp it, lift it above the
table, then carry it to a commanded pose -- with the scene, workspace and control rates
retuned for this arm. The arm is much smaller than the Franka the upstream task targets
(~0.5 m reach vs ~0.85 m), so command ranges, cube size and cube placement are scaled down.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from reBot_RL.assets import REBOT_ARM_CFG

# Prim path of the arm's last link inside the asset. The vendored USD nests every link
# (base_link/link1/.../gripper_end), so the frame transformer needs the full chain rather
# than a flat body name.
_BASE_LINK = "Geometry/base_link"
_GRIPPER_END = f"{_BASE_LINK}/link1/link2/link3/link4/link5/link6/gripper_end"

# Start pose: fingertips hovering at ~(0.35, 0, 0.17), fingers near-horizontal, every joint
# >=0.3 rad from its limits (found with scripts/find_start_pose.py). The asset's authored
# home pose is unusable here -- it hangs the gripper 0.135 m *below* the base, inside the
# table top (table surface is at env z=0). Near-table top-down grasps are kinematically
# impossible for this arm (the wrist pitch chain j2/j3/j4 cannot point the fingers down
# next to the table), so the policy has to grasp the cube side-on.
_START_POSE = {
    "joint1": 0.0,
    "joint2": -1.35,
    "joint3": -0.3,
    "joint4": -0.85,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint_left": 0.04,
    "joint_right": 0.04,
}

# Gripper stroke. joint_left travels 0..0.05 m and joint_right 0..0.0715 m, so the shared
# open command has to stay inside the tighter of the two.
_GRIPPER_OPEN = 0.045
_GRIPPER_CLOSE = 0.0


@configclass
class RebotCubeLiftEnvCfg(LiftEnvCfg):
    """Lift a cube to a commanded pose with the reBot arm."""

    def __post_init__(self):
        super().__post_init__()

        # -- robot, mounted on the table surface (env z=0) in an elbow-up start pose
        self.scene.robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(_START_POSE)

        # -- actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_left", "joint_right"],
            open_command_expr={"joint_left": _GRIPPER_OPEN, "joint_right": _GRIPPER_OPEN},
            close_command_expr={"joint_left": _GRIPPER_CLOSE, "joint_right": _GRIPPER_CLOSE},
        )

        # -- goal pose command, scaled to this arm's workspace
        self.commands.object_pose.body_name = "gripper_end"
        self.commands.object_pose.ranges.pos_x = (0.20, 0.32)
        self.commands.object_pose.ranges.pos_y = (-0.12, 0.12)
        self.commands.object_pose.ranges.pos_z = (0.10, 0.25)

        # -- object: a cube small enough for the 0.045 m finger stroke
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.26, 0.0, 0.03], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # -- end-effector frame at the grasp center between the fingertips. Measured from the
        # finger meshes (scripts/find_start_pose.py): the fingers extend 0.089 m along
        # gripper_end local -X, so the TCP sits ~85% of the way out.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{_BASE_LINK}",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{_GRIPPER_END}",
                    name="end_effector",
                    offset=OffsetCfg(pos=(-0.075, 0.0, 0.0)),
                ),
            ],
        )

        # -- the object starts smaller and closer, so shrink the reset scatter to match
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.10, 0.10),
            "z": (0.0, 0.0),
        }

        # -- lift threshold scaled to the smaller cube
        self.rewards.lifting_object.params["minimal_height"] = 0.03
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.03
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.03

        # -- PhysX GPU buffers: the parent cfg's 16k aggregate-pair capacity is tuned for the
        # Franka scene and overflows by a hair here (PhysX then *silently drops contacts*:
        # "needs to increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity").
        self.sim.physics.gpu_total_aggregate_pairs_capacity = 128 * 1024

        # -- control/sim rates: this asset's drives are stiff (joint2 ~86k N*m/rad after the
        # USD deg->rad conversion), so the upstream 100 Hz physics step is too coarse.
        # 400 Hz physics with decimation 8 keeps the policy at 50 Hz.
        self.decimation = 8
        self.episode_length_s = 5.0
        self.sim.dt = 1.0 / 400.0
        self.sim.render_interval = self.decimation


@configclass
class RebotCubeLiftEnvCfg_PLAY(RebotCubeLiftEnvCfg):
    """Small, deterministic variant for visual inspection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
