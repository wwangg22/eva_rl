# Copyright (c) 2026. Re3Sim workstation effort.
"""The workstation task with its cameras, for training policies that only see pixels.

Identical dynamics, rewards, terminations and observation manager to
``RebotWorkstationPickPlace1EnvCfg`` -- the only additions are the two sensors a real rig has
and the scene changes rendering needs. Kept as a separate task rather than a flag on the
default one because a ``CameraCfg`` in the scene creates one render product PER ENV, and the
state-based task runs at 1024 envs where that is neither affordable nor wanted.

What this variant guarantees that ad-hoc camera mounting did not:

* **the workstation camera's pose lives with the scene.** It used to be re-declared by every
  tool that wanted a picture -- the renderer, the demo recorder, the evaluator -- and the poses
  drifted apart. Now ``STATION_CAM_EYE``/``TARGET`` are applied by a reset event
  (``mdp.aim_station_camera``), so every consumer sees the same view by construction.
* **the gaussian desk is per env.** ``/World/Splats`` is a single prim: at ``num_envs > 1``
  exactly one env has a desk and the others render the arm floating on the ground plane.
  Vision runs must not have to remember an environment variable to avoid that, so this cfg
  re-points the prim itself.

The wrist camera is imported, never re-declared: it carries this D405's factory intrinsics and
a tilt calibrated against the live feed, and a second copy would decalibrate on the first edit.

Resolution defaults to 160x120 -- 4:3, because the wrist intrinsics are calibrated at 640x480
and rendering 16:9 changes the vertical FOV. Raise it with ``RE3SIM_CAM_W`` / ``RE3SIM_CAM_H``,
but note rendering is bound by TOTAL render-product pixels across all envs: 128 envs x 2
cameras x 160x120 is fine, 64 envs x 1 camera x 640x480 comes back blank.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ..lift.camera_cfg import WRIST_CAM_CFG
from . import mdp
from .workstation_env_cfg import (
    STATION_CAM_EYE,
    STATION_CAM_TARGET,
    RebotWorkstationPickPlace1EnvCfg,
)

#: Per-camera render size. 4:3 -- see the module docstring.
CAM_W = int(os.environ.get("RE3SIM_CAM_W", "160"))
CAM_H = int(os.environ.get("RE3SIM_CAM_H", "120"))

#: The workstation camera's optics. Not the wrist's -- that one is a measured device and its
#: intrinsics come from the device. This is a lens choice: 17 mm on a 20.955 mm aperture is
#: 63.3 deg horizontally, which at the 1.07 m from the mount to the cube spans 132 cm.
STATION_LENS = dict(focal_length=17.0, horizontal_aperture=20.955)


@configclass
class RebotWorkstationPickPlace1VisionEnvCfg(RebotWorkstationPickPlace1EnvCfg):
    """The default task plus a wrist camera and the workstation camera."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.wrist_cam = WRIST_CAM_CFG.replace(
            width=CAM_W, height=CAM_H, data_types=["rgb"], update_period=0.0)
        self.scene.station_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/StationCam",
            update_period=0.0,
            width=CAM_W,
            height=CAM_H,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.02, 20.0), **STATION_LENS),
        )
        # Re-aimed after every reset: `reset_scene_to_default` restores prim poses, and a
        # camera left at its spawn pose films the floor.
        self.events.aim_station_cam = EventTerm(
            func=mdp.aim_station_camera,
            mode="reset",
            params={"eye": STATION_CAM_EYE, "target": STATION_CAM_TARGET},
        )

        # ⭐ One gaussian desk PER ENV. Without this, `/World/Splats` sits at the world origin,
        # Isaac Lab's cloner centres the env grid on that origin, and every env but whichever
        # one happens to straddle it renders the arm and the cube on bare ground. Nothing
        # downstream can detect it -- the images still look like images -- so it is not left to
        # an environment variable here.
        if getattr(self.scene, "splats", None) is not None:
            self.scene.splats.prim_path = "{ENV_REGEX_NS}/Splats"


@configclass
class RebotWorkstationPickPlace1VisionEnvCfg_PLAY(RebotWorkstationPickPlace1VisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
