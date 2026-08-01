# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event terms for visual domain randomization.

These operate at the USD level, so the light terms take effect on the next rendered frame
and the camera-mount term must run in ``prestartup`` mode (before the render products are
created).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_light_attr(prim, name: str):
    """Return the light attribute, handling both modern (``inputs:``-prefixed) and legacy names."""
    attr = prim.GetAttribute(f"inputs:{name}")
    if not attr.IsValid():
        attr = prim.GetAttribute(name)
    if not attr.IsValid():
        raise RuntimeError(f"Light prim '{prim.GetPath()}' has no attribute '{name}'.")
    return attr


def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path: str,
    intensity_range: tuple[float, float],
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    yaw_pitch_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
    """Randomize intensity/color (and optionally orientation) of the lights matching ``prim_path``.

    Meant for *global* lights (e.g. the dome light at ``/World/light`` or a distant "sun"
    light): every matching prim is re-randomized whenever the term fires, regardless of
    which envs are resetting. ``yaw_pitch_ranges`` (degrees) re-aims the light -- yaw about
    world +Z, pitch tilting away from straight-down -- which moves shadows around; only
    meaningful for directional lights such as ``DistantLight``.
    """
    from pxr import Gf, UsdGeom

    stage = env.sim.stage
    paths = sim_utils.find_matching_prim_paths(prim_path)
    if not paths:
        raise RuntimeError(f"randomize_light: no prims match '{prim_path}'.")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        _get_light_attr(prim, "intensity").Set(random.uniform(*intensity_range))
        color = Gf.Vec3f(*(random.uniform(*chan) for chan in color_range))
        _get_light_attr(prim, "color").Set(color)
        if yaw_pitch_ranges is not None:
            yaw = random.uniform(*yaw_pitch_ranges[0])
            pitch = random.uniform(*yaw_pitch_ranges[1])
            # identity aims a DistantLight along -Z (straight down); tilt about X, then spin about Z
            rot = Gf.Rotation(Gf.Vec3d.ZAxis(), yaw) * Gf.Rotation(Gf.Vec3d.XAxis(), pitch)
            quat = rot.GetQuat()
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    cur = op.Get()
                    op.Set(Gf.Quatf(quat) if isinstance(cur, Gf.Quatf) else quat)
                    break


def randomize_camera_mount(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_names: tuple[str, ...],
    pos_jitter: float = 0.005,
    rot_jitter_deg: float = 2.0,
):
    """Apply a per-env random offset to camera mounts, simulating real mounting tolerance.

    Perturbs each camera prim's local transform (position by up to ``pos_jitter`` meters per
    axis, orientation by up to ``rot_jitter_deg`` degrees about a random axis). Must run in
    ``prestartup`` mode: the jitter is written once at the USD level, so each env keeps a
    static, slightly-off camera mount for the whole run -- and the wrist camera still follows
    the gripper, since only its local offset changes.
    """
    from pxr import Gf, UsdGeom

    stage = env.sim.stage
    for name in sensor_names:
        sensor_cfg = getattr(env.cfg.scene, name)
        pattern = sensor_cfg.prim_path.replace("{ENV_REGEX_NS}", env.scene.env_regex_ns)
        paths = sim_utils.find_matching_prim_paths(pattern)
        if not paths:
            raise RuntimeError(f"randomize_camera_mount: no prims match '{pattern}'.")
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    cur = op.Get()
                    op.Set(type(cur)(*(cur[i] + random.uniform(-pos_jitter, pos_jitter) for i in range(3))))
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    axis = Gf.Vec3d(*(random.gauss(0.0, 1.0) for _ in range(3)))
                    if axis.GetLength() < 1e-6:
                        axis = Gf.Vec3d.XAxis()
                    jitter = Gf.Rotation(axis.GetNormalized(), random.uniform(-rot_jitter_deg, rot_jitter_deg))
                    cur = op.Get()
                    quat = jitter.GetQuat()
                    op.Set(cur * (Gf.Quatf(quat) if isinstance(cur, Gf.Quatf) else quat))
