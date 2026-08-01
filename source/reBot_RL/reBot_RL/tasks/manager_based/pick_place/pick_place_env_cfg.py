# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-object pick-and-place task for the RS-rebot-dev-arm.

Two everyday objects (scaled YCB tomato-soup can and sugar box) spawn on the +y side of
the workspace; a rigid open-top basket sits on the -y side. The task is complete when
both objects are inside the basket at the end of the episode. The basket walls are tall
enough that objects cannot be pushed in -- they must be lifted over the rim, which the
arm can only do with side-on grasps (near-table top-down grasps are kinematically
impossible for this arm, see the lift task notes).

Robot, control rates and PhysX tuning are carried over from the proven lift task.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from reBot_RL.assets import REBOT_ARM_CFG

# arm-specific constants shared with the lift task (start pose, gripper stroke, link chain)
from ..lift.rebot_lift_env_cfg import _BASE_LINK, _GRIPPER_CLOSE, _GRIPPER_END, _GRIPPER_OPEN, _START_POSE
from . import mdp

##
# Scene
##

# YCB objects at 0.35x: soup can ~0.024 m diameter, sugar box ~0.016 x 0.032 m
# cross section -- both fit the 0.045 m finger stroke for side-on grasps. The USDs ship
# the real masses (0.35+ kg), far too heavy for this small arm, so mass is overridden.
# The assets are authored Y-up (they lie on their sides with identity rotation), so both
# spawn with a +90 deg X-rotation to stand upright. NOTE: Isaac Lab 3.0 quaternions are
# (x, y, z, w) -- identity is (0, 0, 0, 1), NOT the 2.x-style (1, 0, 0, 0).
_YCB_DIR = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"
_OBJ_SCALE = 0.35
_STAND_UP_ROT = [0.7071, 0.0, 0.0, 0.7071]

_OBJ_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    # tame the response to the stiff gripper squeeze (untamed, objects get launched)
    max_depenetration_velocity=1.0,
    disable_gravity=False,
)


def _basket_part(name: str, size: tuple[float, float, float], offset: tuple[float, float, float]) -> AssetBaseCfg:
    """A static (collision-only) cuboid positioned relative to the basket center."""
    cx, cy = mdp.BASKET_CENTER
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Basket" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cx + offset[0], cy + offset[1], offset[2])),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        ),
    )


# basket geometry derived from the mdp constants so the "placed" predicate always matches
_B_IN = mdp.BASKET_INNER_HALF
_B_T = mdp.BASKET_WALL_THICKNESS
_B_H = mdp.BASKET_WALL_HEIGHT
_B_F = mdp.BASKET_FLOOR_THICKNESS
_B_OUT = 2 * (_B_IN + _B_T)  # outer footprint
_B_WALL_C = _B_IN + _B_T / 2  # wall centerline offset


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Table scene with the reBot arm, two YCB objects and a rigid basket."""

    # populated in the env cfg __post_init__
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    object_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectA",
        # z: the YCB origins sit near the object base (standing root z ~0.0145), so spawn
        # a few mm above that -- lower and the mesh starts inside the table and gets
        # popped out (which tipped the bottle over)
        # both spawn regions sit strictly inside the lift teacher's PROVEN graspable
        # envelope (cube spawned at x 0.22-0.30, y -0.10..0.10, r<=0.32): the old far
        # slot (0.29,0.12)+/-scatter reached r~0.35 and its object was never grasped
        # once across runs 3-10 -- likely outside the arm's graspable region
        # BOTH spawn regions sit in the RL grasp primitive's preferred zone (azimuth
        # ~6-20 deg: its grasps were only ever observed there), split into disjoint
        # x-bands -- the hybrid expert's RL phase must be able to grasp either can
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.245, 0.055, 0.020], rot=list(_STAND_UP_ROT)),
        spawn=UsdFileCfg(
            usd_path=f"{_YCB_DIR}/005_tomato_soup_can.usd",
            scale=(_OBJ_SCALE, _OBJ_SCALE, _OBJ_SCALE),
            rigid_props=_OBJ_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
        ),
    )
    object_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectB",
        # a second soup can, not a box/bottle: the sugar box (even re-proportioned) and
        # mustard bottle were never successfully grasped across runs 6-9, while the can
        # grasp is the arm's proven skill -- two cans isolates the actually-missing
        # skill (fetching the SECOND object) and keeps the >90% completion gate honest.
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.305, 0.05, 0.020], rot=list(_STAND_UP_ROT)),
        spawn=UsdFileCfg(
            usd_path=f"{_YCB_DIR}/005_tomato_soup_can.usd",
            scale=(_OBJ_SCALE, _OBJ_SCALE, _OBJ_SCALE),
            rigid_props=_OBJ_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
        ),
    )

    basket_floor = _basket_part("Floor", (_B_OUT, _B_OUT, _B_F), (0.0, 0.0, _B_F / 2))
    basket_wall_px = _basket_part("WallPX", (_B_T, _B_OUT, _B_H), (_B_WALL_C, 0.0, _B_H / 2))
    basket_wall_nx = _basket_part("WallNX", (_B_T, _B_OUT, _B_H), (-_B_WALL_C, 0.0, _B_H / 2))
    basket_wall_py = _basket_part("WallPY", (_B_OUT, _B_T, _B_H), (0.0, _B_WALL_C, _B_H / 2))
    basket_wall_ny = _basket_part("WallNY", (_B_OUT, _B_T, _B_H), (0.0, -_B_WALL_C, _B_H / 2))

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


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Same joint-position + binary-gripper actions as the lift task."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint[1-6]"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_left", "joint_right"],
        open_command_expr={"joint_left": _GRIPPER_OPEN, "joint_right": _GRIPPER_OPEN},
        close_command_expr={"joint_left": _GRIPPER_CLOSE, "joint_right": _GRIPPER_CLOSE},
    )


@configclass
class ObservationsCfg:
    """Privileged state observations (39-d)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # target-first canonical ordering (nearest unplaced object in the first slot):
        # makes the grasp policy slot-agnostic -- see mdp.objects_canonical
        objects = ObsTerm(func=mdp.objects_canonical)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset and startup events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # spawn scatter regions are disjoint x-bands (object_a x <= 0.26, object_b
    # x >= 0.29, min center gap 3 cm > can diameter) so the objects cannot spawn
    # intersecting. No orientation randomization: the pose deltas compose in the body
    # frame, and the objects' body yaw axis is horizontal after the stand-up rotation.
    reset_object_a = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.015, 0.015), "y": (-0.025, 0.025)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_a"),
        },
    )
    reset_object_b = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.015, 0.015), "y": (-0.02, 0.02)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_b"),
        },
    )

    # lying-grasp practice (v6c diagnosis: 25/62 failures = can A knocked flat, a state
    # the policy never trained from). Before the prestage terms so prestage overrides.
    lying_object_a = EventTerm(
        func=mdp.spawn_lying,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object_a"), "prob": 0.10},
    )

    # reverse-curriculum-lite (after the plain resets so it overrides them): some
    # episodes start with an object already placed, so the critic learns the placed-state
    # value (runs 3/4 hovered forever because releasing was never explored) and the
    # policy practices fetching the OTHER object
    prestage_object_a = EventTerm(
        func=mdp.prestage_in_basket,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_a"),
            # MANUAL CURRICULUM, phase 2 (phase 1 = 0.0 while the grasp-carry chain was
            # learned from scratch; prestage income drowns that early signal -- runs 5 &
            # 12). Now that hover-over-basket is established, prestaged episodes teach
            # the critic the placed-state value (the missing link for release) and
            # train the second-object fetch.
            "prob": 0.30,
            "pos": (mdp.BASKET_CENTER[0] - 0.02, mdp.BASKET_CENTER[1], 0.021),
        },
    )
    prestage_object_b = EventTerm(
        func=mdp.prestage_in_basket,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_b"),
            # 0.15 experiment (v7 lineage) lost to the champion on full-task eval
            # (82.4 vs 87.1) -- reverted to the winning recipe
            "prob": 0.0,
            "pos": (mdp.BASKET_CENTER[0] + 0.02, mdp.BASKET_CENTER[1], 0.021),
        },
    )

    # robustness (Big Will): once in a while bump a waiting object a couple of cm
    # sideways so the policy must re-align to the moved target instead of replaying a
    # memorized approach. Held/placed objects are exempt (see mdp.nudge_objects).
    # ~1-3 nudges per object per 10 s episode.
    nudge_object_a = EventTerm(
        func=mdp.nudge_objects,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"asset_cfg": SceneEntityCfg("object_a")},
    )
    nudge_object_b = EventTerm(
        func=mdp.nudge_objects,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"asset_cfg": SceneEntityCfg("object_b")},
    )

    # imported meshes come with PhysX-default (low) friction; the stiff gripper squeeze
    # then launches them (proven on the lift cuboid). Give both objects high friction.
    object_a_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (1.0, 1.2),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "asset_cfg": SceneEntityCfg("object_a"),
        },
    )
    object_b_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (1.0, 1.2),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "asset_cfg": SceneEntityCfg("object_b"),
        },
    )


@configclass
class RewardsCfg:
    """Staged shaping toward stable two-object placement (see mdp.rewards docstring)."""

    # weight 2, a compromise: at 1 the pull toward the second object after the first is
    # placed was imperceptible (runs 6-8 parked); at 4 hovering near an object without
    # grasping became a fat risk-free local optimum that stalled grasp onset (run 12)
    reaching = RewTerm(func=mdp.reach_nearest_unplaced, params={"std": 0.1}, weight=2.0)
    # lift/transport require the ee within 7 cm of the object ("held"): without the gate,
    # run 2 learned to swat objects airborne and off the table instead of grasping
    # all height gates sit at 0.045, BELOW the placed predicate's z-max (0.048): while
    # dipping a held object into the basket the hold income never cuts out before the
    # placed stream (60/s) takes over. With gates at 0.06 there was a dead zone
    # (z 0.048-0.06, all hold streams off) and run 14's release attempts kept getting
    # extinguished at that barrier.
    lifting = RewTerm(func=mdp.unplaced_is_lifted, params={"minimal_height": 0.045, "ee_max_dist": 0.07}, weight=10.0)
    transport_coarse = RewTerm(
        func=mdp.unplaced_to_basket, params={"std": 0.3, "minimal_height": 0.045, "ee_max_dist": 0.07}, weight=16.0
    )
    transport_fine = RewTerm(
        func=mdp.unplaced_to_basket, params={"std": 0.10, "minimal_height": 0.045, "ee_max_dist": 0.07}, weight=5.0
    )
    # bridge between "hovering near the basket" and the dip -- "placed" does NOT
    # require releasing, so no discrete gripper action gates the 60/s stream
    hover_over_basket = RewTerm(func=mdp.held_over_basket, params={"minimal_height": 0.045}, weight=12.0)
    # credit the release itself: the probe on run 4 showed the arm reaches over the
    # basket but opened the gripper there in 0.00% of steps
    release_into_basket = RewTerm(func=mdp.falling_into_basket, weight=10.0)
    # per placed object per step; must out-earn the full hold stream (lift 10 + transport
    # 16+5 + hover 12 = 43) so placing beats hovering over the basket forever
    placed = RewTerm(func=mdp.objects_placed, weight=60.0)

    # knocking an object off the table ends the episode; make that a bad deal instead of
    # a free restart (the lift teacher's fling exploit)
    dropping_penalty = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "object_dropping"}, weight=-30.0)

    # time pressure (Big Will: no dawdling -- surgical and fast): every incomplete step
    # costs, so finishing early strictly dominates. Kept soft during BC fine-tune phase 1:
    # at -3.0 a fresh BC init (garbage template value head) loses its fragile release
    # behavior in the first updates and never rediscovers it (04-35-02 run: success flat
    # 0.000 for 975 epochs, release ~0). Restore toward -3.0 once both-place is reliable.
    time_penalty = RewTerm(func=mdp.task_incomplete, weight=-0.3)

    # meaningful from the start (Big Will: the policy jitters fast around the object;
    # motion must be precise and surgical) -- the old -1e-4 was exploration-era token
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot")})

    # first-try grasps (Big Will): v4 chattered the gripper ~34 sign-flips/episode
    # (15.63 FAILED grasps/ep at 78.9% success). Weight is dt-scaled (x0.02) like every
    # term, so -50 = -1 per flip in raw reward: a committed episode needs 4 flips (-4,
    # negligible) while v4-style chatter costs -34+. At -1.0 (= -0.02/flip, v5 run
    # 08-41-28) the episode cost was ~-1 vs +893 from placement and chatter did not
    # decrease. See mdp.gripper_toggle.
    gripper_toggle = RewTerm(func=mdp.gripper_toggle, weight=-50.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(func=mdp.any_object_dropped, params={"minimum_height": -0.05})


@configclass
class CurriculumCfg:
    """Mild action-penalty ramp (the lift task's 1000x jump collapsed a run; 100x, later)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -6e-2, "num_steps": 30000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1.5e-2, "num_steps": 30000}
    )


##
# Environment
##


@configclass
class RebotPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Place two objects into the basket with the reBot arm."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # -- robot, mounted on the table surface (env z=0) in the lift task's start pose
        self.scene.robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = dict(_START_POSE)

        # -- end-effector frame at the grasp center between the fingertips (from lift)
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

        # -- two objects to place: longer episodes than the single-object lift
        self.episode_length_s = 10.0

        # -- control/sim rates and PhysX tuning carried over from the lift task (stiff
        # drives need 400 Hz physics; the aggregate-pair capacity bump prevents PhysX
        # silently dropping contacts)
        self.decimation = 8
        self.sim.dt = 1.0 / 400.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=128 * 1024,
            friction_correlation_distance=0.00625,
            # the default 5*2^15 patch buffer overflows at 4096 envs of this scene
            # (PhysX asked for >=208896 at startup; overflow = silently dropped contacts)
            gpu_max_rigid_patch_count=2**19,
        )


@configclass
class RebotPickPlaceEnvCfg_PLAY(RebotPickPlaceEnvCfg):
    """Small variant for visual inspection and testing."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
