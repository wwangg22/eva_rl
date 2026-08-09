# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place on a reconstructed real workstation.

The desk, the objects and their dimensions all come from a real capture
(``data/captures/2026-08-05/``) rather than from procedural invention: a rubix cube, a roll of
tape, a tape measure and an open cardboard box, each measured with calipers.

**The task.** Grasp the rubix cube from a random reachable pose with random yaw and place it
inside the open box, which sits stationary at a random reachable spot. The roll of tape and
the tape measure are **clutter** -- real bodies occupying real space that the arm has to work
around. They are not grasp targets.

Why only the cube is a target (decided 2026-08-05, revising the original plan):

* the roll of tape is **91 mm** across, against a gripper that opens **89.1 mm** on command
  (C3). It can only be forced wider by contact, which is not a grasp strategy;
* the tape measure is 184 g, sitting right at the cell where C2 measured the finger drive
  failing (0/6 holds at 0.25 kg with the authored 100 N/m stiffness);
* and the cube's 56 mm height is comfortably inside the envelope, so the task is bounded by
  manipulation skill rather than by hardware limits nothing can fix.

**Colliders: what is provisional and what is permanent.**

* The **box is authored** from the measured dimensions, permanently. This is not a shortcut:
  ``Re3Sim/workstation/tools/extract_object.py:313`` adds a synthetic flat base and
  Poisson-closes the mesh, so a reconstructed box is a *solid*. It has no interior and
  nothing can be placed into it (blocker B2).
* The **cube, roll of tape and tape measure use their reconstructed meshes when those exist**,
  and fall back to analytic primitives when they do not -- see ``_reconstructed_or``. The
  Re3Sim path is ``extract_object.py --target-height`` -> ``mesh_to_rigid_usd.py``, giving a
  rigid body whose collider is the reconstructed mesh, scaled by the caliper measurement: an
  object capture has no marker in frame, so **the calipers supply scale and the mesh supplies
  the shape**. Which path each object took is printed at load time, so a run is never
  ambiguous about what it was measuring. Force the primitives with
  ``REBOT_WORKSTATION_PRIMITIVES=1``.

The desk visual is likewise a placeholder until the 3DGS splats land as scene content.
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

import os

from reBot_RL.assets import REBOT_ARM_CFG

from ..lift.rebot_lift_env_cfg import _BASE_LINK, _GRIPPER_CLOSE, _GRIPPER_END, _GRIPPER_OPEN, _START_POSE
from . import mdp

#: C2: the USD's authored 100 N/m finger drive produces ~1.7 N of squeeze and fails at
#: 0.25 kg. Every challenge env overrides it, and so must this one.
FINGER_STIFFNESS = 2000.0
FINGER_DAMPING = 40.0

# re3sim/ -> manager_based/ -> tasks/ -> reBot_RL/ -> the package root that holds data/
_DATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data", "workstation"))
_BOX_USD = os.path.join(_DATA, "box.usda")
#: Reconstructed rigid-body assets, written by
#: ``Re3Sim/workstation/tools/finish_object_asset.py``. Present => used; absent => the
#: analytic primitive below stands in.
_OBJ_DIR = os.path.join(_DATA, "objects")

#: Set ``REBOT_WORKSTATION_PRIMITIVES=1`` to force the primitives even when the reconstructed
#: assets exist. Useful for a like-for-like comparison, and for reproducing a measurement that
#: was taken before the swap.
_FORCE_PRIMITIVES = os.environ.get("REBOT_WORKSTATION_PRIMITIVES", "") == "1"

#: The reconstructed desk as a NuRec gaussian field, written by
#: ``Re3Sim/workstation/tools/finish_scene_splats.sh`` and symlinked into ``data/workstation/``.
#: ``RE3SIM_SPLATS`` points at an alternative field so a pruned or re-trained one can be
#: A/B-rendered against the shipped asset without swapping a symlink -- the splat yaw bug
#: (02_RECONSTRUCTION.md §5.5) was found by rendering, and comparing renders is how the next
#: one gets found too.
_SPLATS_USD = os.environ.get("RE3SIM_SPLATS") or os.path.join(_DATA, "splats.usd")
_HAS_SPLATS = os.path.exists(_SPLATS_USD) and not _FORCE_PRIMITIVES
#: Desk slab thickness. Only the top face matters -- it is at z = 0 -- but the slab has to be
#: thick enough that a fast-moving finger cannot tunnel through it in one physics step.
_TABLE_T = 0.05
if _HAS_SPLATS:
    print(f"[workstation] desk: reconstructed splats {_SPLATS_USD} "
          f"+ an invisible fitted collider slab")
else:
    print("[workstation] desk: PLACEHOLDER SeattleLabTable -- splats not built yet")


#: ⭐ The **workstation camera**: in FRONT of the arm, looking back at it. Directive of
#: 2026-08-09. It used to sit at ``(-0.62, -0.52, 0.46)`` -- over the arm's shoulder from
#: behind, which is the one place from which the gripper is occluded by the arm itself.
#:
#: Lives here, in the env cfg, because BOTH ``eva_rl/scripts/render_workstation.py`` and
#: ``eva_bc/re3sim/expert/collect_demos.py`` mount a camera at this pose, and the two used to
#: carry their own copies. A view that is inspected in one tool and filmed in another has to
#: be the same view or neither measurement means anything.
#:
#: Chosen by rendering candidates and looking. Three of the first four were wrong in ways the
#: arithmetic did not predict: too low and the gaussian desk's near edge cuts the cube in
#: half, too close and the arm leaves frame.
STATION_CAM_EYE = (0.95, 0.00, 0.42)
STATION_CAM_TARGET = (0.17, 0.00, 0.06)


#: The Rubix cube is **authored**, not reconstructed -- ``scripts/author_rubiks_cube_usd.py``.
#: Photogrammetry cannot see this object: a flat glossy sticker has no matchable texture (its
#: specular highlight moves between frames) while the black grooves match perfectly, so MVS
#: returned a cube-shaped LATTICE with the centre of every sticker punched out, and the
#: extents gate passed it at +4.8 % because a hollow shell has the bounding box of the solid
#: it should be (HANDOFF §3.2). 54 coloured squares on a black box is not a reconstruction
#: problem.
#:
#: ``RE3SIM_CUBE_PATTERN`` selects the sticker pattern: an integer picks ``cube_pNN.usda``
#: (``p00`` is the solved cube), anything else falls back to ``cube.usda``. The patterns are
#: real cube states -- produced by turning a real cube, not by shuffling colours -- so a
#: policy trained across them has seen the object, not one image of it.
_CUBE_PATTERN = os.environ.get("RE3SIM_CUBE_PATTERN", "")


def _cube_usd() -> str:
    if _CUBE_PATTERN.isdigit():
        p = os.path.join(_OBJ_DIR, f"cube_p{int(_CUBE_PATTERN):02d}.usda")
        if os.path.exists(p):
            return p
        print(f"[workstation] RE3SIM_CUBE_PATTERN={_CUBE_PATTERN} has no asset at {p}")
    return os.path.join(_OBJ_DIR, "cube.usda")


def _rigid_props() -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=1.0,
        disable_gravity=False,
    )


def _material(static: float = 0.9, dynamic: float = 0.75) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=static, dynamic_friction=dynamic, restitution=0.0
    )


def _authored_or_primitive(usd: str, mass: float, primitive):
    """Spawn ``usd`` as a rigid body, or the analytic primitive if it has not been built.

    Same body as :func:`_reconstructed_or` and the same origin contract -- it exists so the
    cube, which is **authored**, does not print "reconstructed mesh" at load time and mislead
    whoever reads the log while chasing the next asset bug.
    """
    if _FORCE_PRIMITIVES or not os.path.exists(usd):
        why = "forced by REBOT_WORKSTATION_PRIMITIVES" if _FORCE_PRIMITIVES else "not built yet"
        print(f"[workstation] cube: PRIMITIVE collider ({why}). Expected {usd}")
        return primitive
    print(f"[workstation] cube: authored Rubix cube {os.path.basename(usd)}")
    return UsdFileCfg(
        usd_path=usd,
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=_material(),
    )


#: The clutter reverts to **analytic primitives** by default. The directive of 2026-08-09 is
#: that gaussians render the *workstation*, not the props, and the two scanned clutter bodies
#: are the last of the reconstructed objects: they carry the open washed-out-colour bug
#: (HANDOFF §5.1, 2.6x saturation loss) and render as translucent white blobs, which is worse
#: for a vision student than a plain flat cylinder. They are **never grasped** -- they exist
#: to occupy space -- so a cylinder at the measured diameter and a box at the measured
#: extents are exactly as correct as the scans for everything this env measures, and their
#: colliders are exact rather than convex hulls of a noisy mesh.
#:
#: Set ``RE3SIM_SCANNED_CLUTTER=1`` to put the reconstructed meshes back for an A/B.
_SCANNED_CLUTTER = os.environ.get("RE3SIM_SCANNED_CLUTTER", "") == "1"


def _reconstructed_or(name: str, mass: float, primitive):
    """The reconstructed asset for ``name`` if it has been built, else the analytic primitive.

    Re3Sim's actual contract is that the **mesh is the geometry** -- ``extract_object.py``
    recovers shape and orientation from the capture and the calipers supply only the metric
    scale a marker-less object capture cannot. Analytic primitives are therefore a stand-in,
    not the design, and this env says so at load time rather than in a comment nobody reads.

    Swapping is a file drop precisely because nothing in the MDP inspects the collider type:
    every predicate reads root poses. The one thing that *does* matter is the origin
    convention, and ``finish_object_asset.py`` re-centres the mesh to its bounding-box centre
    for exactly that reason -- ``REST_Z`` and ``placed_mask`` are calibrated for centre-origin
    bodies and would be off by half an object against a base-origin one.

    Mass, friction and the solver settings are re-asserted here rather than inherited from the
    USD, so the reconstructed and primitive paths differ in *shape alone* and the domain
    randomisation keeps its single source of truth.
    """
    usd = os.path.join(_OBJ_DIR, f"{name}.usda")
    if _FORCE_PRIMITIVES or not _SCANNED_CLUTTER or not os.path.exists(usd):
        why = ("forced by REBOT_WORKSTATION_PRIMITIVES" if _FORCE_PRIMITIVES
               else "scans off; RE3SIM_SCANNED_CLUTTER=1 to restore" if not _SCANNED_CLUTTER
               else "not built yet")
        print(f"[workstation] '{name}': PRIMITIVE collider ({why}). "
              f"Expected reconstructed asset at {usd}")
        return primitive
    print(f"[workstation] '{name}': reconstructed mesh {usd}")
    return UsdFileCfg(
        usd_path=usd,
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=_material(),
    )


@configclass
class WorkstationSceneCfg(InteractiveSceneCfg):
    """Desk, arm, the cube, two clutter objects and the open box."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # ------------------------------------------------------------------ the grasp target
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.22, 0.0, mdp.CUBE_SIZE / 2], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=_authored_or_primitive(_cube_usd(), mdp.CUBE_MASS, sim_utils.CuboidCfg(
            size=(mdp.CUBE_SIZE, mdp.CUBE_SIZE, mdp.CUBE_SIZE),
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.20, 0.15)),
        )),
    )

    # ------------------------------------------------------------------------- clutter
    # A solid cylinder, not an annulus: it is an obstacle and is never grasped, so the
    # central hole changes nothing that this task measures.
    rolloftape = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RollOfTape",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.20, 0.16, mdp.TAPE_HEIGHT / 2], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=_reconstructed_or("rolloftape", mdp.TAPE_MASS, sim_utils.CylinderCfg(
            radius=mdp.TAPE_DIAMETER / 2, height=mdp.TAPE_HEIGHT, axis="Z",
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.TAPE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.25, 0.30)),
        )),
    )
    tapemeasure = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TapeMeasure",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.20, -0.16, mdp.TAPEMEASURE_HEIGHT / 2], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=_reconstructed_or("tapemeasure", mdp.TAPEMEASURE_MASS, sim_utils.CuboidCfg(
            size=(mdp.TAPEMEASURE_LONG, mdp.TAPEMEASURE_ACROSS, mdp.TAPEMEASURE_HEIGHT),
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mdp.TAPEMEASURE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.80, 0.10)),
        )),
    )

    # ----------------------------------------------------------------- the open receptacle
    # Kinematic: it must stay where it is put. Authored by scripts/author_workstation_box_usd.py
    # from the same constants the placed predicate reads.
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[*mdp.BOX_DEFAULT_CENTER, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(usd_path=_BOX_USD),
    )

    # The desk. A flat collider whose top face is at z = 0, which is where the marker frame
    # puts the real desk surface once the iPad thickness is folded in (mdp/common.py).
    #
    # An analytic slab, not the reconstructed mesh, and that is the *better* answer rather
    # than the cheaper one: this desk is flat and its true shape is exactly a rectangle, while
    # multi-view stereo on a large weakly-textured surface is close to its worst case. It is
    # also far cheaper for the solver. Sized from `fit_table.py`'s rotated-rectangle fit.
    #
    # Invisible when the splats are loaded, because the splats already show the desk and a
    # solid slab drawn through them looks like a rendering bug.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, -_TABLE_T / 2],
                                                rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(1.60, 1.00, _TABLE_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.28)),
            visible=not _HAS_SPLATS,
        ),
    ) if _HAS_SPLATS else AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # The reconstructed desk, as 356 k gaussians rendered natively by RTX.
    #
    # Placed ONCE in the world, not per env. It is appearance-only -- no collider, nothing in
    # the MDP reads it -- and replicating 356 k gaussians across 1024 envs would cost gigabytes
    # to render a backdrop that only env 0 is standing in anyway. Vision distillation, when it
    # comes, renders one env at a time and will see the real desk; state-based training does
    # not look at pixels at all.
    splats = AssetBaseCfg(
        prim_path="/World/Splats",
        spawn=UsdFileCfg(usd_path=_SPLATS_USD),
    ) if _HAS_SPLATS else None
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
    """7-D: six arm joints (absolute, default-offset, scale 0.5) + a binary gripper.

    Identical to every other reBot env, and that is load-bearing -- the scripted expert, the
    BC dataset and the RL wrappers in ``eva_bc`` all encode actions as
    ``a[:6] = (q - q_default) / 0.5``, ``a[6] = +1 open / -1 close``.
    """

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
    """Privileged state, 41-D: 8 + 8 + 7 + 4 + 6 + 1 + 7."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)          # 8
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)          # 8
        target_pose = ObsTerm(func=mdp.target_pose)          # 7  cube pose in robot root
        box_pose = ObsTerm(func=mdp.box_pose)                # 4  centre xy + (cos, sin) yaw
        clutter = ObsTerm(func=mdp.clutter_obs)              # 6  2 x (dx, dy vs cube, z)
        placed = ObsTerm(func=mdp.target_placed_flag)        # 1
        actions = ObsTerm(func=mdp.last_action)              # 7

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
        func=mdp.target_lifted, weight=8.0,
        params={"minimal_height": 0.075, "ee_max_dist": 0.08,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    carrying = RewTerm(
        func=mdp.target_to_box, weight=12.0,
        params={"std": 0.15, "minimal_height": 0.075, "ee_max_dist": 0.08,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    success = RewTerm(func=mdp.place_success, weight=60.0)
    # a nudge toward tidiness, NOT the constraint -- see mdp.rewards.clutter_disturbed
    tidiness = RewTerm(func=mdp.clutter_disturbed, weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5e-3,
                        params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    target_dropped = DoneTerm(func=mdp.target_dropped, params={"minimum_height": -0.05})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # Must follow `reset_all`, which puts the arm back at the single pose every episode has
    # ever started from. Off unless `RE3SIM_ARM_START_JITTER` is set.
    reset_arm_start = EventTerm(func=mdp.randomize_arm_start, mode="reset")
    # ORDER MATTERS. The box is placed first and the objects are then kept clear of it:
    # sampling objects first can leave a 218 mm box no legal spot, and an env that spawns
    # unwinnable episodes silently caps every success number measured in it.
    reset_box = EventTerm(func=mdp.reset_box, mode="reset")
    reset_objects = EventTerm(func=mdp.reset_objects, mode="reset")
    # must run last: records where the clutter ended up, for the disturbance metric
    record_clutter = EventTerm(func=mdp.record_clutter_spawn, mode="reset")

    # ---- domain randomisation ----------------------------------------------------------
    # Mass and friction randomise at runtime and are honest. SIZE does not: C8 measured that
    # a post-build `xformOp:scale` never reaches the PhysX collider (a block scaled z x 0.5
    # rests at 32 mm, not the 17.5 mm its visual implies). Size DR therefore lives in the
    # separately-gated `-SizeDR-v0` variant, not here.
    # Every env shows a DIFFERENT, and valid, Rubix cube pattern. Startup, like the mass and
    # friction randomisation: nothing about a reset needs the paint to change.
    cube_pattern = EventTerm(func=mdp.randomize_cube_pattern, mode="startup")
    cube_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("cube"), "mass_distribution_params": (0.055, 0.095),
                "operation": "abs"},
    )
    cube_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("cube"),
                "static_friction_range": (0.7, 1.1), "dynamic_friction_range": (0.55, 0.95),
                "restitution_range": (0.0, 0.0), "num_buckets": 16},
    )
    clutter_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("rolloftape"),
                "static_friction_range": (0.7, 1.1), "dynamic_friction_range": (0.55, 0.95),
                "restitution_range": (0.0, 0.0), "num_buckets": 16},
    )


@configclass
class RebotWorkstationPickPlace1EnvCfg(ManagerBasedRLEnvCfg):
    """Grasp the cube off a reconstructed desk and place it in the open box."""

    scene: WorkstationSceneCfg = WorkstationSceneCfg(num_envs=1024, env_spacing=2.5)
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
                    offset=OffsetCfg(pos=mdp.TCP_OFFSET),  # C10, never -0.075
                )
            ],
        )
        self.decimation = 8
        # a full fetch is reach + grasp + lift + carry ~250 mm + release; the two-phase
        # pregrasp and clutter tasks both need 14 s and this is no shorter
        # 26 s = 1300 env steps at decimation 8 / dt 1/400.
        #
        # MEASURED: once the expert drives the approach from the reset pose instead of being
        # teleported to the pre-grasp, its demonstrations run 1018-1118 env steps. The old
        # 16 s gave 800, so evaluation truncated every episode mid-transit -- the policy
        # scored 3.4 % with `no_lift` at 91.9 % and a clutter displacement of 0.04 mm, i.e. it
        # had barely moved anything before the episode was taken away from it. That reads
        # exactly like "BC did not learn the task" and is not.
        #
        # The horizon is a property of the MANOEUVRE, so it has to be re-derived whenever the
        # manoeuvre changes length. 1300 leaves ~16 % headroom over the longest demonstration.
        self.episode_length_s = 26.0
        self.sim.dt = 1.0 / 400.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=128 * 1024,
            friction_correlation_distance=0.00625,
            # arm + 3 objects + a 5-part box collider in contact
            gpu_max_rigid_patch_count=2**20,
        )


@configclass
class RebotWorkstationPickPlace1EnvCfg_PLAY(RebotWorkstationPickPlace1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0


@configclass
class RebotWorkstationPickPlace1StrictEnvCfg(RebotWorkstationPickPlace1EnvCfg):
    """Harder rung: disturbing the clutter ends the episode.

    Kept as a separate variant rather than folded into the default, because ``clutter``
    measured what this costs: a 73.3 % expert scored 16.4 % once displacement was actually
    tested. Nothing should be developed against this until the default task is solved.
    """

    def __post_init__(self):
        super().__post_init__()
        self.terminations.clutter_disturbed = DoneTerm(
            func=mdp.clutter_disturbed_term, params={"tol": mdp.DISTURB_TOL}
        )
        # Mirrors the termination, and it is not optional: ending an episode early is worth a
        # large POSITIVE amount to an agent accumulating negative shaping terms, so without
        # this, shoving the clutter would become the profitable move.
        self.rewards.disturb_penalty = RewTerm(
            func=mdp.is_terminated_term, weight=-40.0,
            params={"term_keys": "clutter_disturbed"},
        )
