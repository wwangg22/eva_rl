# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera sensors mirroring the real capture rig: a RealSense D455 overlooking the
workspace and a D405 mounted on the gripper.

Both are pinhole models of the cameras' standard 1280x720 (16:9) color streams. Only the
horizontal FOV and aspect ratio are set -- Isaac Lab derives the vertical aperture from the
aspect (square pixels), which lands within the datasheet's +-3 deg lens tolerance of the
spec'd vertical FOVs. Horizontal FOVs from the D400-series datasheet (rev 017):

- D455 RGB: 90 deg H (65 deg V at native 16:10; the 16:9 stream crops V to ~60 deg)
- D405: 84 deg H x 58 deg V (same imager serves depth and color; product page quotes 87 deg)

Once the physical rig exists, replace these with measured device intrinsics
(``rs-enumerate-devices -c``) via ``PinholeCameraCfg.from_intrinsic_matrix``.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from .rebot_lift_env_cfg import _GRIPPER_END

# USD default 35 mm-film horizontal aperture; focal_length = aperture / (2 * tan(HFOV / 2)).
_APERTURE = 20.955

WORKSPACE_CAM_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/WorkspaceCam",
    # Placeholder rig pose: 0.85 m from the cube region -- inside the D455's 0.6-6 m ideal
    # range -- looking at (0.25, 0, 0.05). Quaternion (x, y, z, w) = yaw 180 deg then pitch
    # 40.2 deg down (world convention: forward +X, up +Z). Replace with the measured mount
    # pose once the real rig exists.
    offset=CameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.6), rot=(-0.344, 0.0, 0.939, 0.0), convention="world"),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=_APERTURE / 2.0,  # 2*tan(45 deg) = 2 -> 90 deg HFOV
        horizontal_aperture=_APERTURE,
        clipping_range=(0.2, 20.0),
    ),
    width=1280,
    height=720,
    data_types=["rgb"],
)

WRIST_CAM_CFG = CameraCfg(
    # Leaf name is required: pointing prim_path at the rigid-body prim itself makes Isaac Lab
    # silently re-root the sensor at <link>/camera.
    prim_path="{ENV_REGEX_NS}/Robot/" + _GRIPPER_END + "/WristCam",
    # Mount chosen by Big Will from rendered sweep grids (logs/camera_previews/wrist_grid_sweep*):
    # position q5 (19 cm toward the gripper, 1 cm up from the first-guess mount) with rotation
    # "c" (forward prim +X, up prim +Z). NOTE the offset frame: the gripper_end geometry prim is
    # NOT the physics body frame -- it spawns world-aligned -- so tune this empirically via the
    # sweep scripts rather than deriving quaternions from link axes.
    # 2026-08-03: tilted -30 deg about the camera-local X (pitch) axis so the view centers the
    # gripper instead of sitting perpendicular to the wrist (Big Will's pick, tilt_x_m30 from
    # scripts/wrist_cam_tilt_sweep.py; optical axis lands 1.8 deg off the TCP, camera->TCP
    # 0.171 m). rot = old (0.5, -0.5, -0.5, 0.5) composed with R_x(-30 deg), xyzw.
    offset=CameraCfg.OffsetCfg(
        pos=(-0.092, 0.0, 0.042),
        rot=(0.353553, -0.353553, -0.612372, 0.612372),
        convention="opengl",
    ),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=_APERTURE / 1.80083,  # 2*tan(42 deg) -> 84 deg HFOV
        horizontal_aperture=_APERTURE,
        clipping_range=(0.02, 5.0),
    ),
    width=1280,
    height=720,
    data_types=["rgb"],
)
