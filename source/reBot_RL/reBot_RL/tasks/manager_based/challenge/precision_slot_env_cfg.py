# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Precision slot insertion for the reBot arm.

Grasp an upright block from the table and slide it into a snug slot whose inner width
exceeds the block by a configurable per-side clearance (1.5 mm by default). The
pick-place task's basket accepts the object anywhere inside a +/-50 mm footprint; this
slot rejects anything more than about a millimetre off the centreline or a few degrees
off square, so the same "carry it over and let go" policy scores zero.

Two measured facts about this arm shape the whole design (see docs/CHALLENGE_SUITE.md):

* **There is no top-down grasp below z = 0.19 m** -- 0.00 % of table-height voxels admit a
  finger axis within 26 deg of vertical, and the steepest approach available anywhere in
  that band is 42.3 deg off vertical. So the insertion is *horizontal*: the block slides
  in along +x rather than being lowered into a cavity.
* **The slot walls are shorter than the block.** The gripper holds the block's upper half
  while its lower half is between the walls; if the walls were as tall as the block the
  fingers would collide with them, and the arm cannot release from above to avoid that.
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

#: per-side clearance between block and slot wall [m]. The difficulty axis.
#: IndustReal uses 0.5-0.6 mm on a research-grade Franka; this starts looser.
DEFAULT_CLEARANCE = 0.0015

#: The USD authors the finger drive at 100 N/m, which caps the squeeze at ~1.7 N and the
#: payload at ~0.05 kg -- two orders below the arm's 2.5 kg rating (docs C2). 2000 N/m
#: gives 43 N of squeeze and holds 0.25 kg reliably, without the instability seen at 3e4.
#: The revolute joints keep their validated USD gains: only the fingers are overridden.
FINGER_STIFFNESS = 2000.0
FINGER_DAMPING = 40.0

_C = mdp  # geometry constants live in mdp.common and are re-exported


def _static_box(name: str, size: tuple[float, float, float], pos: tuple[float, float, float],
                color: tuple[float, float, float] = (0.35, 0.35, 0.40)) -> AssetBaseCfg:
    """A collision-only cuboid. The fixture is static, so no rigid body is needed."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Slot" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def _fixture_parts(clearance: float) -> dict[str, AssetBaseCfg]:
    """Build the slot from four static boxes, sized from the mdp constants.

    Deriving the geometry from the same constants the success predicate reads is
    deliberate: the v0 basket keeps its dimensions in two places (the env cfg and a
    separately generated USD) and nothing stops them drifting apart.
    """
    cx, cy = mdp.SLOT_CENTER
    half = mdp.slot_clearance_to_halfwidth(clearance)
    t, h, d, fz = mdp.WALL_THICKNESS, mdp.WALL_HEIGHT, mdp.SLOT_DEPTH, mdp.SLOT_FLOOR_Z
    outer = 2 * (half + t)
    return {
        "floor": _static_box("Floor", (d, outer, fz), (cx, cy, fz / 2)),
        "wall_py": _static_box("WallPY", (d, t, h), (cx, cy + half + t / 2, fz + h / 2)),
        "wall_ny": _static_box("WallNY", (d, t, h), (cx, cy - half - t / 2, fz + h / 2)),
        "back": _static_box("Back", (t, outer, h), (cx + d / 2 + t / 2, cy, fz + h / 2),
                            (0.55, 0.25, 0.25)),
    }


@configclass
class PrecisionSlotSceneCfg(InteractiveSceneCfg):
    """Table, arm, one block, and the slot fixture."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        # spawns to the robot's right, r ~ 0.26, azimuth ~ -31 deg: inside the envelope
        # the lift teacher actually grasped in (r <= 0.32), and well clear of the slot
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.220, -0.130, 0.035], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=sim_utils.CuboidCfg(
            # 30 mm across the fingers sits in the measured 26-42 mm grasp sweet spot
            size=(2 * 0.0225, 2 * 0.0150, 2 * 0.0350),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.85, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.75)),
        ),
    )

    # fixture parts are filled in __post_init__ once the clearance is known
    slot_floor: AssetBaseCfg = MISSING
    slot_wall_py: AssetBaseCfg = MISSING
    slot_wall_ny: AssetBaseCfg = MISSING
    slot_back: AssetBaseCfg = MISSING

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
    """Same 7-D action space as every other reBot task: 6 joint positions + binary gripper."""

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
    """Privileged state, 34-D."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)                       # 8
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)                       # 8
        block_pose = ObsTerm(func=mdp.block_pose_in_root)                 # 7
        # the task-frame error, the way Factory feeds `fingertip_pos_rel_fixed`
        slot_error = ObsTerm(func=mdp.slot_frame)                         # 4
        actions = ObsTerm(func=mdp.last_action)                           # 7

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Factory-style multi-scale keypoint shaping plus staged bonuses."""

    reaching = RewTerm(
        func=mdp.reach_block, weight=2.0,
        params={"std": 0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    lifting = RewTerm(
        func=mdp.block_lifted, weight=8.0,
        params={"minimal_height": 0.045, "ee_max_dist": 0.08,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # the three length scales ARE the curriculum: coarse pulls from anywhere, fine only
    # pays inside the last millimetres. A single tanh cannot express both.
    kp_baseline = RewTerm(func=mdp.keypoint_baseline, weight=6.0)
    kp_coarse = RewTerm(func=mdp.keypoint_coarse, weight=12.0)
    kp_fine = RewTerm(func=mdp.keypoint_fine, weight=20.0)

    engaged = RewTerm(func=mdp.engaged, weight=15.0)
    inserted = RewTerm(func=mdp.inserted, weight=60.0)

    dropping_penalty = RewTerm(
        func=mdp.is_terminated_term, weight=-30.0, params={"term_keys": "block_dropped"}
    )
    toppling_penalty = RewTerm(
        func=mdp.is_terminated_term, weight=-10.0, params={"term_keys": "block_toppled"}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    block_dropped = DoneTerm(func=mdp.block_dropped, params={"minimum_height": -0.05})
    block_toppled = DoneTerm(func=mdp.block_toppled, params={"max_tilt": 0.6})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_block = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # no yaw randomization on the plain reset: the block is authored upright, so a
            # body-frame yaw delta is a genuine yaw here (unlike the Y-up YCB assets in
            # pick_place, where it would tip them over)
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.03, 0.03), "yaw": (-0.35, 0.35)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("block"),
        },
    )
    block_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("block"),
            "static_friction_range": (0.9, 1.1),
            "dynamic_friction_range": (0.75, 0.95),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


@configclass
class RebotPrecisionSlotEnvCfg(ManagerBasedRLEnvCfg):
    """Precision slot insertion, 1.5 mm per-side clearance."""

    scene: PrecisionSlotSceneCfg = PrecisionSlotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    #: per-side clearance [m]; read back by the success predicate
    clearance: float = DEFAULT_CLEARANCE
    #: derived, so mdp.is_inserted and the scene geometry cannot disagree
    slot_half_width: float = DEFAULT_CLEARANCE + 0.0150

    def __post_init__(self):
        self.scene.robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(_START_POSE)
        # Split the single ".*" actuator group in two. The revolute joints keep
        # stiffness/damping = None so Isaac Lab preserves the USD's validated gains
        # (hardcoding the raw USD numbers de-scales every revolute gain 57.3x, since
        # UsdPhysics authors angular gains per degree). Only the fingers are overridden,
        # because the USD's 100 N/m caps the squeeze at ~1.7 N -- see docs C2.
        self.scene.robot.actuators = {
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"], stiffness=None, damping=None),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["joint_left", "joint_right"],
                stiffness=FINGER_STIFFNESS,
                damping=FINGER_DAMPING,
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

        self.slot_half_width = self.clearance + mdp.BLOCK_HALF[1]
        for key, cfg in _fixture_parts(self.clearance).items():
            setattr(self.scene, f"slot_{key}", cfg)

        self.decimation = 8
        self.episode_length_s = 12.0
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
class RebotPrecisionSlotEnvCfg_PLAY(RebotPrecisionSlotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False


@configclass
class RebotPrecisionSlotLooseEnvCfg(RebotPrecisionSlotEnvCfg):
    """3 mm clearance -- the easy rung of the difficulty ladder."""

    clearance: float = 0.0030


@configclass
class RebotPrecisionSlotTightEnvCfg(RebotPrecisionSlotEnvCfg):
    """0.5 mm clearance, matching IndustReal's tightest shipped peg."""

    clearance: float = 0.0005
