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
    # 2026-08-09: position set from the MEASURED physical mount, replacing the sweep-picked
    # placeholder. Measured relative to the physical fingertip: 77 mm behind the tip along the
    # gripper axis, 67 mm above it, 9.5 mm to the robot's left (looking wrist->fingertips).
    # Conventions verified in-sim, world-space only (scratch gripper_frame_probe, 2026-08-09):
    #   - the gripper_end BODY origin is AT the fingertip: the closed finger-body origins
    #     coincide (0.1 mm) at a point 41.9 mm behind it along wrist->tip -- exactly the
    #     measured TCP_OFFSET. (The fingers/motor geometry extends backward from the origin.)
    #   - pointing direction taken from link6->gripper_end body positions (not local axes).
    #   - CAUTION: the gripper_left/gripper_right body NAMES are mirrored -- gripper_left sits
    #     on the robot's RIGHT. Robot-left here = up x fwd, not the naming.
    # FRAME (verified 2026-08-09, nested-camera swing test): the renderer composes this
    # offset with the gripper_end BODY pose via fabric -- the camera genuinely rides the
    # link -- but the sensor's reported pos_w / the USD transform stay frozen at the spawn
    # value. NEVER solve or verify this offset against cam.data.pos_w; verify by rendering.
    # (The old "offset frame is world-aligned" comment was a misdiagnosis of that stale
    # readout.) So the offset is authored directly in the gripper_end body frame, whose
    # origin is the physical fingertip with -X backward along the gripper axis, +Z up,
    # +Y robot-left: pos is the tape-measured mount verbatim -- 77 mm behind the tip,
    # 9.5 mm to the robot's left, 67 mm above.
    # Rotation: (0.5, -0.5, -0.5, 0.5) bore-sights the camera along the gripper axis
    # (opengl, xyzw); composed with R_x(-18.8 deg). The tilt was CALIBRATED against the
    # live D405 feed (2026-08-09): sweep sim tilts, measure how many pixel rows of the
    # fingertip silhouette are visible from the frame bottom, match the real feed's
    # 81 px / 16.9% -> -18.8 deg below the gripper axis (sim shows 75 px at -18, 98 px
    # at -21). The user's protractor said ~12 deg -- the delta is case-vs-optical-axis
    # + principal-point offset; trust the optical match. Residual mount uncertainty is
    # covered by randomize_camera_mount's per-env jitter.
    offset=CameraCfg.OffsetCfg(
        pos=(-0.077, 0.0095, 0.067),
        rot=(0.41162, -0.41162, -0.57495, 0.57495),
        convention="opengl",
    ),
    # MEASURED device intrinsics (2026-08-09, rs API, D405 serial 260522275150, color
    # 640x480 -- the 4:3 stream the real rig records, NOT the 16:9 datasheet crop):
    # fx=392.18 fy=391.44 ppx=314.62 ppy=225.73 -> 78.4 x 63.1 deg FOV. The 4:3 vertical
    # FOV is what puts the fingertip at the bottom edge of the real feed; the old 16:9
    # model clipped it. Principal point is 14 px above center (~2 deg of effective tilt),
    # so keep the off-center cx/cy. Distortion (mild inverse Brown-Conrady) not modeled.
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[392.18, 0.0, 314.62, 0.0, 391.44, 225.73, 0.0, 0.0, 1.0],
        width=640,
        height=480,
        clipping_range=(0.02, 5.0),
    ),
    width=640,
    height=480,
    data_types=["rgb"],
)
