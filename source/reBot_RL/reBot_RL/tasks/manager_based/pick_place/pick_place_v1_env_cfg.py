# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place v1: randomized basket, wider spawns, diversity axes, 41-D obs.

Stage 0 of the planner->ACT->residual pipeline (reBot_ACT/PLAN.md). Differences to v0:

* the 5 static basket cuboids are replaced by ONE kinematic rigid body (compound
  collider authored in ``data/basket/basket.usda``) that a reset event moves per env per
  episode: r in [0.20, 0.27] m, azimuth in [-50, +50] deg, >= 0.14 m from every object;
* both cans sample an annulus sector r in [0.20, 0.32] (clamp 0.33), azimuth +/-45 deg,
  pairwise separation >= 0.06 m, lying-on-side probability 0.25 per can;
* diversity axes: per-ENV object scale in [0.8, 1.25]x (USD prestartup event -- true
  per-episode rescale is unsupported by PhysX), per-episode mass x[0.5, 2], +/-10%
  drive-gain jitter, +/-0.1 rad arm start-joint offsets;
* observation grows to 41-D: env-local basket center xy appended after
  ``objects_canonical`` (all other terms and their order unchanged);
* rewards are sparse only (placed stream + drop penalty): this env is driven by a
  motion-planner expert, then ACT/DAgger; dense RL shaping returns in a later stage.
  Prestage/curriculum events and the shaping terms are dropped; interval nudges stay
  available behind ``enable_nudges`` (off for nominal data collection).

v0 (``pick_place_env_cfg.py``) stays registered and untouched for comparison.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass

from . import mdp
from .pick_place_env_cfg import _OBJ_SCALE, PickPlaceSceneCfg, RebotPickPlaceEnvCfg, TerminationsCfg

# the movable basket USD authored by scripts/author_basket_usd.py (5-box compound
# collider, dims from mdp/common.py so the placed predicate always matches)
_BASKET_USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "basket", "basket.usda")
)

##
# Scene
##


@configclass
class PickPlaceV1SceneCfg(PickPlaceSceneCfg):
    """v0 scene with the static basket cuboids replaced by one movable kinematic basket."""

    basket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        # default pose = the v0 center; the reset event re-samples it every episode
        init_state=RigidObjectCfg.InitialStateCfg(pos=[mdp.BASKET_CENTER[0], mdp.BASKET_CENTER[1], 0.0]),
        spawn=UsdFileCfg(
            usd_path=_BASKET_USD_PATH,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # replaced by the rigid body above
    basket_floor = None
    basket_wall_px = None
    basket_wall_nx = None
    basket_wall_py = None
    basket_wall_ny = None


##
# MDP settings
##


@configclass
class ObservationsV1Cfg:
    """Privileged state observations (41-d: v0's 39-d + env-local basket center xy)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # target-first canonical ordering preserved from v0 (BC data depends on it)
        objects = ObsTerm(func=mdp.objects_canonical)
        # the basket moves per env per episode in v1, so the policy must be told where
        basket_center = ObsTerm(func=mdp.basket_center_xy)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventV1Cfg:
    """Reset, diversity-randomization and (optional) perturbation events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # basket FIRST (its small sector can be fully covered by the object-clearance
    # discs, so objects-first can be infeasible), then the cans avoid it
    reset_basket = EventTerm(
        func=mdp.reset_basket,
        mode="reset",
        params={
            "radius_range": (0.20, 0.27),
            "azimuth_range": (-math.radians(50.0), math.radians(50.0)),
        },
    )
    # wider spawns: both cans, replacing the v0 per-object box regions
    reset_objects = EventTerm(
        func=mdp.reset_objects_wide,
        mode="reset",
        params={
            "radius_range": (0.20, 0.32),
            "azimuth_range": (-math.pi / 4, math.pi / 4),
            "min_separation": 0.06,
            "min_basket_clearance": 0.14,
            "max_radius": 0.33,
            "lying_prob": 0.25,
        },
    )

    # planner-expert is start-agnostic; RL's fixed start pose was a crutch. Small
    # offsets around _START_POSE, arm joints only (gripper stays at its default).
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # per-episode dynamics diversity: mass x[0.5, 2] on the default 0.04 kg
    object_a_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_a"),
            "mass_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )
    object_b_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object_b"),
            "mass_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    # light drive-gain jitter (+/-10% around the actuator cfg gains)
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # per-ENV object scale in [0.8, 1.25] x the v0 0.35x -- USD-level, so it can only
    # run before the simulation starts (mode "prestartup", needs replicate_physics
    # False); the sampled values are ABSOLUTE xformOp:scale, hence the multiplied range.
    # True per-episode rescale is not supported by PhysX. Mass does NOT follow the
    # geometry (the USD masses are overridden to 0.04 kg anyway); the mass events above
    # cover that axis.
    object_a_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.8 * _OBJ_SCALE, 1.25 * _OBJ_SCALE), "asset_cfg": SceneEntityCfg("object_a")},
    )
    object_b_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.8 * _OBJ_SCALE, 1.25 * _OBJ_SCALE), "asset_cfg": SceneEntityCfg("object_b")},
    )

    # interval nudges, carried over from v0 for recovery-episode collection; disabled in
    # the nominal cfg via RebotPickPlaceV1EnvCfg.enable_nudges
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

    # high friction so the stiff gripper squeeze cannot launch the cans (proven in v0)
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
class RewardsV1Cfg:
    """Sparse rewards only -- this env is driven by a planner expert, not shaped RL.

    ``objects_placed`` doubles as the success metric logger (Metrics/success_rate);
    weights are dt-scaled (x0.02) like every manager term but only relative magnitude
    matters here.
    """

    placed = RewTerm(func=mdp.objects_placed, weight=60.0)
    dropping_penalty = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "object_dropping"}, weight=-30.0)


##
# Environment
##


@configclass
class RebotPickPlaceV1EnvCfg(RebotPickPlaceEnvCfg):
    """Place two cans into a per-episode-randomized basket with the reBot arm."""

    scene: PickPlaceV1SceneCfg = PickPlaceV1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsV1Cfg = ObservationsV1Cfg()
    rewards: RewardsV1Cfg = RewardsV1Cfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventV1Cfg = EventV1Cfg()
    # no prestage/shaping curriculum in v1
    curriculum: object | None = None

    # interval nudge perturbations: off for nominal expert data collection, on for
    # recovery-episode collection (flip the flag or use a variant cfg)
    enable_nudges: bool = False

    def __post_init__(self):
        # robot/ee_frame/rates/PhysX tuning (incl. gpu_max_rigid_patch_count) from v0
        super().__post_init__()
        # USD-level (prestartup) scale randomization needs per-env physics parsing
        self.scene.replicate_physics = False
        if not self.enable_nudges:
            self.events.nudge_object_a = None
            self.events.nudge_object_b = None


@configclass
class RebotPickPlaceV1EnvCfg_PLAY(RebotPickPlaceV1EnvCfg):
    """Small variant for visual inspection and testing: task randomization (basket,
    spawns, lying) stays on, dynamics/geometry diversity is stripped for repeatability."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.events.reset_robot_joints = None
        self.events.object_a_mass = None
        self.events.object_b_mass = None
        self.events.actuator_gains = None
        self.events.object_a_scale = None
        self.events.object_b_scale = None
