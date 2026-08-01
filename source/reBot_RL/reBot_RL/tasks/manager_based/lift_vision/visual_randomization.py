# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual domain randomization for the reBot lift scenes.

:func:`apply_visual_randomization` decorates any ``RebotCubeLiftEnvCfg``-derived config
(state- or vision-based) that has the ``workspace_cam``/``wrist_cam`` sensors, adding:

- **Cube appearance**: the instanceable DexCube USD is swapped for a same-size solid-color
  cuboid (instanceable prims cannot be edited per env), whose color is randomized per env at
  prestartup -- the policy cannot key on one texture.
- **Occlusions / clutter**: three distractor rigid bodies (box, cylinder, sphere) with
  per-env random color and scale, scattered around the cube region every reset. They pass
  between the cameras and the cube, producing partial occlusions in both views.
- **Background**: a static table mat under the workspace so the manipulation area is not
  bare table texture everywhere.
- **Lighting**: dome-light intensity/color randomized on reset, plus a distant "sun" light
  whose direction is also randomized -- shadows move between episodes.
- **Camera mounts**: per-env static jitter of both camera offsets (position + rotation) at
  prestartup, modeling real-world mounting tolerance.

USD-level per-env randomization requires ``scene.replicate_physics = False``, which this
function sets (slower scene construction, unchanged runtime).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from . import mdp as rand_mdp

# same footprint as the scaled DexCube of the state env (0.065 m * 0.4)
_CUBE_SIZE = 0.026

_DISTRACTOR_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=1,
    max_depenetration_velocity=1.0,
)

# Generated shapes ship no physics material, and the PhysX default friction is far below the
# DexCube's -- the stiff gripper squeeze then launches the cube like a watermelon seed
# (observed: cube flung meters away ~2 s after grasp). High friction + a tame depenetration
# cap make the cuboid graspable like the original cube.
_GRASPABLE_MATERIAL = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0)

# full-channel ranges; per-env colors are sampled at prestartup
_RGB_RANGES = {"r": (0.05, 0.95), "g": (0.05, 0.95), "b": (0.05, 0.95)}


def _distractor(name: str, spawn_shape: sim_utils.SpawnerCfg, pos: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=spawn_shape,
    )


def apply_visual_randomization(
    env_cfg,
    *,
    swap_cube: bool = True,
    distractors: bool = True,
    lights: bool = True,
    camera_jitter: bool = True,
) -> None:
    """Add visual-randomization scene entities and events to ``env_cfg`` in place.

    The keyword flags allow partial application, e.g. the distillation collector can drop
    the distractors so the state-based teacher is not physically obstructed.
    """
    # per-env USD edits (colors, scales, camera mounts) need un-replicated scenes
    env_cfg.scene.replicate_physics = False

    # -- cube: solid-color cuboid replacing the (uneditable) instanceable DexCube. Physics
    # matches the state env's cube closely: same size, comparable mass, same solver budget.
    if swap_cube:
        env_cfg.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.26, 0.0, 0.03], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=(_CUBE_SIZE, _CUBE_SIZE, _CUBE_SIZE),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=_GRASPABLE_MATERIAL,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
            ),
        )
        env_cfg.events.randomize_cube_color = EventTerm(
            func=base_mdp.randomize_visual_color,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mesh_name": "geometry/mesh",
                "colors": dict(_RGB_RANGES),
                "event_name": "rand_cube_color",
            },
        )

    # -- table mat: breaks up the uniform tabletop behind the cube in both camera views
    env_cfg.scene.table_mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableMat",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.0, 0.0015)),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.7, 0.003),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)),
        ),
    )

    # -- distractors: three shapes scattered through the cube region each reset. They spawn
    # slightly beyond the cube (larger x), i.e. between the cube and both cameras, so they
    # regularly occlude it partially in either view.
    if distractors:
        shapes = {
            "distractor_1": (
                sim_utils.CuboidCfg(
                    size=(0.03, 0.03, 0.03),
                    rigid_props=_DISTRACTOR_RIGID_PROPS,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=_GRASPABLE_MATERIAL,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.55, 0.1)),
                ),
                (0.34, 0.10, 0.02),
            ),
            "distractor_2": (
                sim_utils.CylinderCfg(
                    radius=0.013,
                    height=0.05,
                    rigid_props=_DISTRACTOR_RIGID_PROPS,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=_GRASPABLE_MATERIAL,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.3, 0.85)),
                ),
                (0.34, -0.10, 0.03),
            ),
            "distractor_3": (
                sim_utils.SphereCfg(
                    radius=0.016,
                    rigid_props=_DISTRACTOR_RIGID_PROPS,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=_GRASPABLE_MATERIAL,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.3)),
                ),
                (0.38, 0.0, 0.02),
            ),
        }
        for name, (shape, pos) in shapes.items():
            setattr(env_cfg.scene, name, _distractor(name.title().replace("_", ""), shape, pos))
            setattr(
                env_cfg.events,
                f"reset_{name}",
                EventTerm(
                    func=base_mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {"x": (-0.05, 0.05), "y": (-0.10, 0.10), "yaw": (-3.14, 3.14)},
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg(name),
                    },
                ),
            )
            setattr(
                env_cfg.events,
                f"randomize_{name}_color",
                EventTerm(
                    func=base_mdp.randomize_visual_color,
                    mode="prestartup",
                    params={
                        "asset_cfg": SceneEntityCfg(name),
                        "mesh_name": "geometry/mesh",
                        "colors": dict(_RGB_RANGES),
                        "event_name": f"rand_{name}_color",
                    },
                ),
            )
            setattr(
                env_cfg.events,
                f"randomize_{name}_scale",
                EventTerm(
                    func=base_mdp.randomize_rigid_body_scale,
                    mode="prestartup",
                    params={"scale_range": (0.7, 1.4), "asset_cfg": SceneEntityCfg(name)},
                ),
            )

    # -- lighting: randomize the global dome light on reset, and add a distant "sun" whose
    # direction is randomized too, so shadow direction varies between episodes. Both lights
    # are global -- one sample applies to all envs at once; diversity comes over time.
    if lights:
        env_cfg.scene.sun = AssetBaseCfg(
            prim_path="/World/sunLight",
            spawn=sim_utils.DistantLightCfg(intensity=1500.0, angle=0.53, color=(1.0, 1.0, 0.95)),
        )
        env_cfg.events.randomize_dome_light = EventTerm(
            func=rand_mdp.randomize_light,
            mode="reset",
            params={
                "prim_path": "/World/light",
                "intensity_range": (800.0, 4000.0),
                "color_range": ((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
            },
        )
        env_cfg.events.randomize_sun = EventTerm(
            func=rand_mdp.randomize_light,
            mode="reset",
            params={
                "prim_path": "/World/sunLight",
                "intensity_range": (300.0, 3000.0),
                "color_range": ((0.85, 1.0), (0.8, 1.0), (0.7, 1.0)),
                "yaw_pitch_ranges": ((0.0, 360.0), (5.0, 50.0)),
            },
        )

    # -- camera mounting tolerance: static per-env pose jitter, applied once at prestartup
    if camera_jitter:
        env_cfg.events.randomize_camera_mounts = EventTerm(
            func=rand_mdp.randomize_camera_mount,
            mode="prestartup",
            params={"sensor_names": ("workspace_cam", "wrist_cam"), "pos_jitter": 0.006, "rot_jitter_deg": 2.0},
        )
