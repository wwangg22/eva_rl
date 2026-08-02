# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-grasp reconfiguration (extrinsic dexterity) for the reBot arm.

The block spawns **lying flat**, in a pose where no grasp exists:

* the gripper's clear opening is **89.1 mm**, measured -- not the 45 mm that
  ``_GRIPPER_OPEN`` suggests, because that is a per-finger joint value and both fingers
  move (``scripts/analysis/gripper_stroke.py``);
* lying, the block's horizontal shadow is **100 mm** across at its narrowest, so the
  fingers cannot span it from any heading. Tipped up on edge it presents **60 mm**;
* and the arm cannot come at it from above -- 0.00 % of table-height voxels admit a
  finger axis within 26 deg of vertical (docs/CHALLENGE_SUITE.md C1).

So the policy must first *create its own preconditions*: push the block against the back
wall so it pivots up onto its end, then grasp it. The reorientation earns no task reward of
its own, and the reach-and-close gradient that drives the pick-place task is flat until it
has already happened -- which is exactly the credit-assignment structure the existing task
never poses.

This is the setup from "Learning to Grasp the Ungraspable with Emergent Extrinsic
Dexterity" (CoRL 2022), verified there with a plain parallel-jaw gripper, no tactile and no
force sensing -- so it is a fair fit for this hardware.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from reBot_RL.assets import REBOT_ARM_CFG

from ..lift.rebot_lift_env_cfg import _BASE_LINK, _GRIPPER_CLOSE, _GRIPPER_END, _GRIPPER_OPEN, _START_POSE
from . import mdp

FINGER_STIFFNESS = 2000.0
FINGER_DAMPING = 40.0

_H = mdp.PG_BLOCK_HALF  # (0.050, 0.030, 0.050) -- authored standing on edge
#: lying flat: 90 deg about x, so the 100 mm long axis goes horizontal and the 60 mm
#: graspable thickness goes vertical. (x, y, z, w).
_LYING_ROT = [0.7071, 0.0, 0.0, 0.7071]
#: lying, the block's far edge sits just short of the wall so a short push jams it there
_SPAWN_X = 0.265


@configclass
class PreGraspSceneCfg(InteractiveSceneCfg):
    """Table, arm, one flat block, and the back wall it gets levered against."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        # lying down the vertical half-extent is _H[1] = 0.030, so it rests at z = 0.030
        init_state=RigidObjectCfg.InitialStateCfg(pos=[_SPAWN_X, 0.0, _H[1]], rot=list(_LYING_ROT)),
        spawn=sim_utils.CuboidCfg(
            size=(2 * _H[0], 2 * _H[1], 2 * _H[2]),  # 100 x 60 x 100 mm
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # moderate friction: too high and the block skids instead of pivoting,
                # too low and it slides out from under the finger before it can tip
                static_friction=0.8, dynamic_friction=0.6, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.10)),
        ),
    )

    # the extrinsic feature the whole task hinges on
    # tall enough that a 100 mm block rotating up its face stays supported all the way
    wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(mdp.WALL_X + 0.02, 0.0, 0.070)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.36, 0.14),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.32, 0.36)),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["joint[1-6]"], scale=0.5, use_default_offset=True
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left", "joint_right"],
        open_command_expr={"joint_left": _GRIPPER_OPEN, "joint_right": _GRIPPER_OPEN},
        close_command_expr={"joint_left": _GRIPPER_CLOSE, "joint_right": _GRIPPER_CLOSE},
    )


@configclass
class ObservationsCfg:
    """Privileged state, 32-D."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)                       # 8
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)                       # 8
        block_pose = ObsTerm(func=mdp.block_pose_in_root)                 # 7
        # the affordance the policy has to reason about: how wide the block currently
        # presents, and how far up it is. It starts at 100 mm -- above the measured 89.1 mm
        # opening -- and falls to 60 mm once the block is up on edge.
        block_extent = ObsTerm(func=mdp.obs_min_grasp_width)              # 1
        block_up = ObsTerm(func=mdp.obs_block_up_axis)                    # 1
        actions = ObsTerm(func=mdp.last_action)                           # 7

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reaching = RewTerm(
        func=mdp.reach_block, weight=1.5,
        params={"std": 0.12, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # the load-bearing shaping term: nothing else pays out until the block is up
    uprighting = RewTerm(func=mdp.uprighting_progress, weight=20.0)
    # standing on edge the block's centre is already at 0.050, so the lift threshold has
    # to sit above that or simply tipping the block up would collect the lift reward
    lifting = RewTerm(
        func=mdp.block_lifted, weight=10.0,
        params={"minimal_height": 0.075, "ee_max_dist": 0.08,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    success = RewTerm(
        func=mdp.pregrasp_success, weight=60.0,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    dropping_penalty = RewTerm(
        func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": "block_dropped"}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    block_dropped = DoneTerm(func=mdp.block_dropped, params={"minimum_height": -0.05})
    # NOTE: no `block_toppled` term here -- lying down is the *start* state, not a failure


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_block = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # yaw jitter is applied in the BODY frame, and the block is lying down, so a
            # yaw delta here rolls it about its own long axis rather than spinning it on
            # the table -- keep it small, and vary position instead
            # keep x tight so the block's far edge stays within a short push of the wall
            "pose_range": {"x": (-0.010, 0.010), "y": (-0.030, 0.030), "yaw": (-0.10, 0.10)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )
    block_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "static_friction_range": (0.7, 0.9),
            "dynamic_friction_range": (0.5, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


@configclass
class RebotPreGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Topple an ungraspable flat block against a wall, then pick it up."""

    scene: PreGraspSceneCfg = PreGraspSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.scene.robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(_START_POSE)
        # see docs C2: the USD's 100 N/m finger drive cannot hold this block
        self.scene.robot.actuators = {
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"], stiffness=None, damping=None),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["joint_left", "joint_right"],
                stiffness=FINGER_STIFFNESS, damping=FINGER_DAMPING,
            ),
        }
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{_BASE_LINK}",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{_GRIPPER_END}",
                    name="end_effector",
                    # measured TCP, not the -0.075 inherited from the lift task -- see
                    # mdp/common.TCP_OFFSET
                    offset=OffsetCfg(pos=mdp.TCP_OFFSET),
                )
            ],
        )
        self.decimation = 8
        self.episode_length_s = 14.0  # two-phase task, needs longer than a plain pick
        self.sim.dt = 1.0 / 400.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=128 * 1024,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_patch_count=2**19,
        )


@configclass
class RebotPreGraspEnvCfg_PLAY(RebotPreGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
