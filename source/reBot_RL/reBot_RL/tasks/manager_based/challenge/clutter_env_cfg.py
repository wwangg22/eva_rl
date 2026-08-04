# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clutter extraction for the reBot arm.

A target block sits in the middle of a tight row of four distractors. Extract it and set it
down in the goal zone **without moving any neighbour** -- displacing one by more than
``mdp.DISTURB_TOL`` (2 mm), or toppling it, ends the episode.

⚠ **The constraint was tightened on 2026-08-03 and this changes every published number.**
It used to be toppling alone, which a neighbour dragged the length of the table and set down
upright passed cleanly; a scripted expert scoring 73.3 % under the old rule scores ~16 %
under this one. ``Rebot-ClutterExtract-Lenient-v0`` reproduces the old behaviour, and exists
only so the old baselines stay re-runnable. Rationale and the measurement that forced it:
``eva_bc/clutter/docs/14_FEEDBACK_AND_NEXT.md``.

The row pitch is 42 mm against a 30 mm block, so the free gap between neighbours is 12 mm.
The gripper's fingers have to come down that gap, or the policy has to push a neighbour
aside first. That makes the correct first action frequently *not* a grasp, which is the
credit-assignment problem plain pick-and-place never poses: in the existing basket task the
greedy reach-and-close action is always right.

Reference point: VPG (Zeng et al. 2018) measured 82.7 % completion for a policy that learns
pushing and grasping jointly against 40.6 % for grasping alone, on a UR5 with a parallel jaw
-- the same class of gripper as this arm. ManiSkill's `PickClutterYCB` scores 0.00-0.08 for
challenge entrants, so this axis has a lot of headroom.
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

_H = mdp.CL_BLOCK_HALF  # (0.018, 0.015, 0.035)
#: row centre and pitch. r ~ 0.25 keeps every block inside the measured graspable envelope.
ROW_X = 0.250
ROW_PITCH = 0.042  # 12 mm of free gap between 30 mm blocks
#: distractor y offsets: two either side of the target at y = 0
_OFFSETS = (-2 * ROW_PITCH, -ROW_PITCH, ROW_PITCH, 2 * ROW_PITCH)


def _block(name: str, y: float, color: tuple[float, float, float], mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=[ROW_X, y, _H[2]], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=sim_utils.CuboidCfg(
            size=(2 * _H[0], 2 * _H[1], 2 * _H[2]),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.75, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class ClutterSceneCfg(InteractiveSceneCfg):
    """Table, arm, one target and four distractors in a row."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    target = _block("Target", 0.0, (0.10, 0.55, 0.85), 0.04)
    # distractors are deliberately LIGHTER than the target, so they topple readily and the
    # no-topple constraint actually bites rather than being satisfied by accident
    distractor_0 = _block("Distractor0", _OFFSETS[0], (0.75, 0.72, 0.68), 0.025)
    distractor_1 = _block("Distractor1", _OFFSETS[1], (0.75, 0.72, 0.68), 0.025)
    distractor_2 = _block("Distractor2", _OFFSETS[2], (0.75, 0.72, 0.68), 0.025)
    distractor_3 = _block("Distractor3", _OFFSETS[3], (0.75, 0.72, 0.68), 0.025)

    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*mdp.GOAL_XY, 0.001)),
        # visual only: a collider here would fight the block being set down on it
        spawn=sim_utils.CylinderCfg(
            radius=mdp.GOAL_RADIUS, height=0.002,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.70, 0.25)),
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
    """Privileged state, 42-D."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)                                   # 8
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)                                   # 8
        target_pose = ObsTerm(func=mdp.block_pose_in_root, params={"name": "target"})  # 7
        # 4 distractors x (dx, dy relative to the target, up-axis)
        clutter = ObsTerm(func=mdp.clutter_obs)                                       # 12
        actions = ObsTerm(func=mdp.last_action)                                       # 7

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reaching = RewTerm(
        func=mdp.reach_target, weight=2.0,
        params={"std": 0.10, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    lifting = RewTerm(
        func=mdp.block_lifted, weight=8.0,
        params={"minimal_height": 0.055, "ee_max_dist": 0.08,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"), "name": "target"},
    )
    extracted = RewTerm(func=mdp.target_extracted, weight=15.0)
    carrying = RewTerm(func=mdp.target_to_goal, weight=12.0, params={"std": 0.12})
    success = RewTerm(func=mdp.clutter_success, weight=60.0)

    # the dense half of the no-disturbance constraint: pays continuously so there is a
    # gradient to follow long before the 2 mm cliff
    disturbance = RewTerm(func=mdp.distractors_disturbed, weight=-3.0)
    topple_penalty = RewTerm(
        func=mdp.is_terminated_term, weight=-40.0, params={"term_keys": "distractor_toppled"}
    )
    # Mirrors `topple_penalty`, and it is not optional. `distractor_disturbed` ends the
    # episode, and ending an episode early is worth a large *positive* amount to an agent
    # that is accumulating negative shaping terms -- without this, adding the termination
    # would have made shoving a neighbour the profitable move.
    disturb_penalty = RewTerm(
        func=mdp.is_terminated_term, weight=-40.0, params={"term_keys": "distractor_disturbed"}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    target_dropped = DoneTerm(func=mdp.block_dropped, params={"minimum_height": -0.05, "name": "target"})
    #: the constraint that makes this a clutter task rather than a pick task
    distractor_toppled = DoneTerm(func=mdp.any_distractor_toppled)
    #: ...and the one that makes it a *careful* clutter task. Added 2026-08-03: toppling
    #: alone let a neighbour be dragged 200 mm to the goal and still scored it a success.
    #: Fires first in almost every case, since a block must slide before it can tip.
    distractor_disturbed = DoneTerm(
        func=mdp.any_distractor_disturbed, params={"tol": mdp.DISTURB_TOL}
    )


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.012, 0.012), "yaw": (-0.20, 0.20)},
                "velocity_range": {}, "asset_cfg": SceneEntityCfg("target")},
    )
    # jitter each distractor independently: a uniformly-spaced row is a much easier
    # problem than one with an uneven, sometimes-tighter gap
    reset_d0 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                         params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.005, 0.005)},
                                 "velocity_range": {}, "asset_cfg": SceneEntityCfg("distractor_0")})
    reset_d1 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                         params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.005, 0.005)},
                                 "velocity_range": {}, "asset_cfg": SceneEntityCfg("distractor_1")})
    reset_d2 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                         params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.005, 0.005)},
                                 "velocity_range": {}, "asset_cfg": SceneEntityCfg("distractor_2")})
    reset_d3 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                         params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.005, 0.005)},
                                 "velocity_range": {}, "asset_cfg": SceneEntityCfg("distractor_3")})
    # must run last: records where the distractors ended up, for the disturbance reward
    record_spawn = EventTerm(func=mdp.record_spawn_xy, mode="reset")


@configclass
class RebotClutterExtractEnvCfg(ManagerBasedRLEnvCfg):
    """Extract a target from a tight row without toppling its neighbours."""

    scene: ClutterSceneCfg = ClutterSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.scene.robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(_START_POSE)
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
        self.episode_length_s = 14.0
        self.sim.dt = 1.0 / 400.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=128 * 1024,
            friction_correlation_distance=0.00625,
            # 6 rigid bodies per env in contact -- more patches than the slot task needs
            gpu_max_rigid_patch_count=2**20,
        )


@configclass
class RebotClutterExtractEnvCfg_PLAY(RebotClutterExtractEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0


@configclass
class RebotClutterExtractTightEnvCfg(RebotClutterExtractEnvCfg):
    """Harder rung: the row is packed tighter, leaving a 6 mm gap."""

    def __post_init__(self):
        super().__post_init__()
        pitch = 0.036
        for i, s in enumerate((-2, -1, 1, 2)):
            getattr(self.scene, f"distractor_{i}").init_state.pos = [ROW_X, s * pitch, _H[2]]


@configclass
class RebotClutterExtractLenientEnvCfg(RebotClutterExtractEnvCfg):
    """The task exactly as it stood before 2026-08-03: topple-only, no displacement limit.

    Kept as a named variant rather than deleted, for one reason: every success number in
    ``eva_bc/clutter/docs/07_``-``13_`` was measured under this predicate, and a baseline
    that cannot be re-run is a baseline that cannot be checked. Nothing new should be
    measured here.
    """

    def __post_init__(self):
        super().__post_init__()
        self.terminations.distractor_disturbed = None
        self.rewards.disturb_penalty = None
        self.rewards.success.params = {"tol": float("inf")}
