# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset and randomisation events for the reconstructed-workstation task."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from .common import (
    BOX_OUTER_X,
    BOX_OUTER_Y,
    CLUTTER_NAMES,
    CUBE_SIZE,
    OBJECT_NAMES,
    TAPE_HEIGHT,
    TAPEMEASURE_HEIGHT,
    TARGET_NAME,
    box_centers_local,
    box_yaws,
)

#: rest height of each object's root above the desk [m]
REST_Z = {
    TARGET_NAME: CUBE_SIZE / 2,
    "rolloftape": TAPE_HEIGHT / 2,
    "tapemeasure": TAPEMEASURE_HEIGHT / 2,
}

#: half-extent of each object's planar footprint, for the overlap rejection test [m]
FOOTPRINT_R = {
    TARGET_NAME: 0.040,       # 56 mm cube -> half-diagonal 39.6 mm
    "rolloftape": 0.0455,     # 91 mm disc
    "tapemeasure": 0.048,     # 71.5 x 64 mm -> half-diagonal 47.9 mm
}

#: Candidate lattice for the last-resort placement, as (radii, azimuths) swept into a grid.
#:
#: An earlier version used three FIXED slots, one per object, hand-verified pairwise clear.
#: That was wrong, and the diagnostic (``scripts/diag_workstation_spawn.py``) showed exactly
#: why: the slots are only mutually clear when **all three** objects use them. In practice the
#: cube and the tape roll place fine by rejection sampling and only the *tape measure* falls
#: back (20-28 % of envs), so its fixed slot landed wherever the other two had been placed at
#: random -- 25-44 overlaps per 256 envs, with centres as close as 0.3 mm.
#:
#: The fallback is therefore **conflict-aware**: score every lattice point by its clearance to
#: the box and to the already-placed objects, and take the best. That is the best available
#: position by construction rather than a position that is merely clear of a hypothetical.
FALLBACK_RADII = (0.16, 0.19, 0.22, 0.25, 0.28)
FALLBACK_N_AZ = 17
FALLBACK_AZ_LIMIT = 1.15


_LATTICE_CACHE: dict = {}


def _fallback_lattice(device) -> torch.Tensor:
    """Candidate xy positions for the conflict-aware fallback. Shape (K, 2). Cached."""
    key = str(device)
    if key not in _LATTICE_CACHE:
        az = torch.linspace(-FALLBACK_AZ_LIMIT, FALLBACK_AZ_LIMIT, FALLBACK_N_AZ, device=device)
        r = torch.tensor(FALLBACK_RADII, device=device)
        rr, aa = torch.meshgrid(r, az, indexing="ij")
        _LATTICE_CACHE[key] = torch.stack(
            [(rr * torch.cos(aa)).reshape(-1), (rr * torch.sin(aa)).reshape(-1)], dim=1)
    return _LATTICE_CACHE[key]


def reset_box(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    radius_range: tuple[float, float] = (0.26, 0.31),
    azimuth_abs_range: tuple[float, float] = (0.44, 0.96),
    yaw_jitter: float = 0.30,
):
    """Place the stationary open box at a random reachable spot, and record it.

    The box is a **kinematic** rigid body (see ``author_workstation_box_usd.py``). Kinematic
    is the honest way to express "this does not move": at 95 g a dynamic box would be shoved
    across the desk on first contact, and mass tricks leak -- a very heavy box still slides,
    just less. Kinematic bodies ignore contact forces entirely, which is what the user asked
    for.

    The box is 218 mm long inside an r <= 0.32 m reach envelope (C4), so it consumes a large
    fraction of the reachable workspace. Three things follow, and all are deliberate:

    * it is placed on the **far** arc (r 0.26-0.31) and yawed roughly **tangentially**, so
      its long axis sweeps across the workspace rather than radially out of it;
    * its azimuth is drawn **off-centre** (|az| in 25-55 deg, side chosen at random) rather
      than around straight-ahead. The first version sampled |az| <= 30 deg and the smoke test
      caught the consequence: with the keep-out region inflated by the tape roll's 45.5 mm
      radius the box covered most of the r in [0.16, 0.27] sector, rejection sampling ran out
      of tries, and objects spawned *inside* the box -- physics then ejected them to r = 0.51
      with 100 mm of vertical error. Off-centre placement always leaves the opposite side
      free;
    * it is placed FIRST, and ``reset_objects`` then keeps clear of it. Sampling objects
      first can leave the box no legal spot at all, and an env that spawns unwinnable
      episodes silently caps every success number measured in it.

    The env-local centre and yaw go into the buffers behind ``box_centers_local`` /
    ``box_yaws``, which the placed predicate, the rewards, the object sampler and the
    observations all read. They are **not** re-read from sim data: during a reset event
    ``data.root_pos_w`` still holds the pre-reset pose.
    """
    box = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device

    r = radius_range[0] + torch.rand(n, device=dev) * (radius_range[1] - radius_range[0])
    mag = azimuth_abs_range[0] + torch.rand(n, device=dev) * (
        azimuth_abs_range[1] - azimuth_abs_range[0])
    side = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    az = side * mag
    centers = torch.stack([r * torch.cos(az), r * torch.sin(az)], dim=1)
    # remembered so the object sampler has a guaranteed-free fallback region
    sides = getattr(env, "_box_side", None)
    if sides is None:
        sides = torch.ones(env.num_envs, device=dev)
        env._box_side = sides
    sides[env_ids] = side
    # tangential +/- jitter: the long axis lies across the reach annulus, not along it
    yaw = az + math.pi / 2 + (torch.rand(n, device=dev) * 2 - 1) * yaw_jitter

    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, :2] += centers
    pose[:, 5] = torch.sin(yaw / 2)   # (x, y, z, w): rotation about +z
    pose[:, 6] = torch.cos(yaw / 2)
    box.write_root_pose_to_sim(pose, env_ids=env_ids)
    box.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    box_centers_local(env)[env_ids] = centers
    box_yaws(env)[env_ids] = yaw


def reset_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float] = (0.15, 0.28),
    target_radius_range: tuple[float, float] = (0.225, 0.28),
    azimuth_range: tuple[float, float] = (-1.1345, 1.1345),
    min_separation: float = 0.015,
    box_clearance: float = 0.020,
    max_tries: int = 80,
):
    """Scatter the cube and the two clutter objects across the reachable desk.

    Rejection sampling in the env-local reach annulus (0 rad = +x, the robot's forward),
    sequential over ``OBJECT_NAMES`` so an accepted object is never thrown away. A candidate
    is rejected if it overlaps an already-placed object or intrudes into the box footprint.

    Overlap is tested between **inflated discs** -- each object's ``FOOTPRINT_R`` plus
    ``min_separation`` -- rather than as a fixed centre distance. The three objects here
    differ in size by a factor of two (a 91 mm tape disc against a 56 mm cube), so a single
    centre-distance threshold is either too loose for the disc or needlessly tight for the
    cube.

    Yaw is uniform on [0, 2pi); roll and pitch are fixed. That is the user's constraint and
    it also keeps each object's unobserved underside -- which the reconstruction fills with a
    synthetic flat base -- permanently hidden from the camera.

    ⭐ The GRASP TARGET gets a tighter annulus than the clutter (``target_radius_range``),
    and the reason is the planner, not the scene. The proven 12 953-entry grasp table
    (``data/pick_place_demos/grasp_table.pt``, shared with the pick-and-place expert) spans
    tool radii **0.221 .. 0.337 m**, and ``spike_plan_grasp.table_candidates`` matches by
    radius within 0.035 m and deliberately refuses to substitute distant candidates -- a
    rigid lateral shift of 2-3 cm leaves this wrist's feasibility manifold entirely.

    MEASURED by calling ``table_candidates`` on 60 real cube spawns: at the old
    ``radius_range`` minimum of 0.15 m only **34/60 (57 %)** returned a goalset at all, and
    every failure sat at radius 0.152-0.191 while every success sat at 0.194-0.279. With a
    goalset in hand, ``plan_grasp`` then succeeded **10/10**. So the binding constraint was
    purely that the cube could spawn nearer the base than the table reaches.

    0.225 m completes that logic (was 0.20). 0.20 was set from goalset EXISTENCE — the
    0.035 m matching tolerance still returns candidates down to r ≈ 0.19 — but candidate
    existence is not executability: the 2026-08-11 64-episode cuRobo-expert run measured
    the sub-floor band directly, and **every one of its 7 whole-episode grasp failures in
    the r < 0.225 band** was a shifted row air-closing or refusing to plan (the table's
    own "never translate laterally" rule, paid at execution time instead of planning
    time). 0.225 = the table's 0.221 floor plus margin. The clutter keeps the full
    ``radius_range`` -- it is never grasped, so the table does not constrain it, and
    shrinking it too would needlessly reduce scene diversity.

    NOTE this makes the task EASIER than every result recorded before 2026-08-10 (and
    2026-08-11's floor raise slightly again): the near band, where the arm is most
    folded, is no longer sampled for the target.

    Must run AFTER ``reset_box``: it reads the box centre from the buffer that event writes.
    """
    n = len(env_ids)
    dev = env.device
    centers = box_centers_local(env)[env_ids]
    yaws_box = box_yaws(env)[env_ids]
    # the box footprint as an oriented rectangle, inflated -- tested in the box's own frame
    half = torch.tensor([BOX_OUTER_X / 2, BOX_OUTER_Y / 2], device=dev)
    c_b, s_b = torch.cos(-yaws_box), torch.sin(-yaws_box)

    xy = torch.zeros(n, len(OBJECT_NAMES), 2, device=dev)
    for i, name in enumerate(OBJECT_NAMES):
        ri = FOOTPRINT_R[name]
        pending = torch.ones(n, dtype=torch.bool, device=dev)
        for _ in range(max_tries):
            if not pending.any():
                break
            idx = pending.nonzero(as_tuple=False).squeeze(1)
            m = len(idx)
            rr = target_radius_range if name == TARGET_NAME else radius_range
            r = rr[0] + torch.rand(m, device=dev) * (rr[1] - rr[0])
            az = azimuth_range[0] + torch.rand(m, device=dev) * (azimuth_range[1] - azimuth_range[0])
            cand = torch.stack([r * torch.cos(az), r * torch.sin(az)], dim=1)
            xy[idx, i] = cand

            d = cand - centers[idx]
            bx = c_b[idx] * d[:, 0] - s_b[idx] * d[:, 1]
            by = s_b[idx] * d[:, 0] + c_b[idx] * d[:, 1]
            outside_box = (bx.abs() > half[0] + ri + box_clearance) | (
                by.abs() > half[1] + ri + box_clearance
            )
            ok = outside_box
            for j, other in enumerate(OBJECT_NAMES[:i]):
                gap = ri + FOOTPRINT_R[other] + min_separation
                ok = ok & (torch.linalg.norm(cand - xy[idx, j], dim=1) >= gap)
            pending[idx] = ~ok

        # Conflict-aware fallback for draws that exhaust their tries. Score every lattice
        # point by its clearance to the box and to the objects already placed, and take the
        # argmax -- the best available position, not merely a position that is clear in
        # isolation. See FALLBACK_RADII for the two earlier versions this replaces and the
        # measurement that killed each.
        if pending.any():
            idx = pending.nonzero(as_tuple=False).squeeze(1)
            m = len(idx)
            lat = _fallback_lattice(dev)                              # (K, 2)
            cand = lat.unsqueeze(0).expand(m, lat.shape[0], 2)        # (m, K, 2)

            dd = cand - centers[idx].unsqueeze(1)
            lbx = c_b[idx].unsqueeze(1) * dd[..., 0] - s_b[idx].unsqueeze(1) * dd[..., 1]
            lby = s_b[idx].unsqueeze(1) * dd[..., 0] + c_b[idx].unsqueeze(1) * dd[..., 1]
            # signed clearance outside the box footprint, minus our own radius
            score = torch.maximum(lbx.abs() - half[0], lby.abs() - half[1]) - ri
            for j, other in enumerate(OBJECT_NAMES[:i]):
                dj = torch.linalg.norm(cand - xy[idx, j].unsqueeze(1), dim=2)
                score = torch.minimum(score, dj - (ri + FOOTPRINT_R[other]))
            xy[idx, i] = lat[score.argmax(dim=1)]

            cnt = getattr(env, "_spawn_fallbacks", None)
            if cnt is None:
                cnt = {}
                env._spawn_fallbacks = cnt
            cnt[name] = cnt.get(name, 0) + m

    for i, name in enumerate(OBJECT_NAMES):
        obj = env.scene[name]
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :3] = env.scene.env_origins[env_ids]
        pose[:, :2] += xy[:, i]
        pose[:, 2] += REST_Z[name]
        half_yaw = torch.rand(n, device=dev) * math.pi  # yaw ~ U[0, 2pi)
        pose[:, 5] = torch.sin(half_yaw)
        pose[:, 6] = torch.cos(half_yaw)
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def record_clutter_spawn(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Store where the clutter ended up, so displacement can be *measured* rather than assumed.

    LESSONS_INHERITED B3: a constraint that is never checked is not a constraint. This task
    does not currently forbid disturbing the clutter -- but it does report it, and the strict
    variant's termination reads exactly this buffer. Recording it unconditionally means the
    lenient and strict variants can never disagree about what "moved" means.
    """
    buf = getattr(env, "_clutter_spawn_xy", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, len(CLUTTER_NAMES), 2, device=env.device)
        env._clutter_spawn_xy = buf
    for i, name in enumerate(CLUTTER_NAMES):
        pos = env.scene[name].data.root_pos_w.torch[env_ids] - env.scene.env_origins[env_ids]
        buf[env_ids, i] = pos[:, :2]


def randomize_cube_pattern(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(TARGET_NAME),
):
    """Give every env its own Rubix cube pattern, so no two show the same object.

    A policy that has only ever seen one cube has seen one *image*, not the object. This
    rewrites the ``GeomSubset`` index arrays on each cloned ``Cube/Stickers`` mesh -- 54 quads
    redistributed over six colour subsets -- so each env displays a different, and **valid**,
    cube state. Geometry, mass, collider and origin are untouched: this is paint.

    Why it is done this way rather than by spawning different USDs
    --------------------------------------------------------------
    ``MultiUsdFileCfg.random_choice`` is a documented no-op in this build (it warns and
    ignores), and ``InteractiveSceneCfg.random_heterogeneous_cloning``, which replaced it,
    does not exist here at all. What *does* hold is that the cloned prims are real and
    editable -- MEASURED: at 4 envs every ``env_i/Cube/Stickers`` reports
    ``instanceable=False, instance_proxy=False`` and accepts a rebind -- so the paint can be
    changed after the fact.

    ``mode="startup"``, like the mass and friction randomisation, and for the same reason: a
    per-reset USD write would be paid every episode to change something no reset needs.

    Silently does nothing if the cube is not the authored asset (the analytic-primitive
    fallback has no sticker mesh), which is what makes ``REBOT_WORKSTATION_PRIMITIVES=1``
    still work.
    """
    if os.environ.get("RE3SIM_CUBE_PATTERN_DR", "1") != "1":
        return
    import omni.usd  # noqa: PLC0415
    from pxr import Vt  # noqa: PLC0415

    from .. import rubiks  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    # `cfg.prim_path` comes back ALREADY RESOLVED -- "/World/envs/env_.*/Cube", not
    # "{ENV_REGEX_NS}/Cube" -- so stripping the placeholder silently produced a doubled path
    # and the whole term no-op'd. Take the leaf and rebuild.
    leaf = env.scene[asset_cfg.name].cfg.prim_path.rsplit("/", 1)[-1]
    # The seed is the env's own index plus the run seed, so a pattern is reproducible from the
    # run's seed alone -- a demonstration set can be regenerated exactly.
    base = int(getattr(env.cfg, "seed", 0) or 0) * 100_003
    changed = 0
    # `mode="startup"` terms are called with `env_ids=None`, meaning "all of them" -- the
    # reset-mode convention of always receiving a tensor does not hold here.
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for i in ids:
        stickers = stage.GetPrimAtPath(f"/World/envs/env_{i}/{leaf}/Stickers")
        if not stickers.IsValid():
            # Loud, not silent. A no-op here means every env shows the same cube, which is
            # invisible in every automated check this env has -- nothing in the MDP reads the
            # paint. The first version returned quietly and did nothing at all for two runs.
            print(f"[workstation] cube pattern DR skipped: no prim at "
                  f"/World/envs/env_{i}/{leaf}/Stickers (analytic primitive?)")
            return
        faces = rubiks.pattern(base + i).face_indices()
        for color, idx in faces.items():
            sub = stage.GetPrimAtPath(f"/World/envs/env_{i}/{leaf}/Stickers/{color}")
            if sub.IsValid():
                sub.GetAttribute("indices").Set(Vt.IntArray(idx))
        changed += 1
    print(f"[workstation] randomised the Rubix cube pattern on {changed} envs "
          f"(RE3SIM_CUBE_PATTERN_DR=0 to disable)")


def randomize_arm_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    jitter: float | None = None,
):
    """Start the arm somewhere other than the single pose ``env.reset()`` always produces.

    Every episode this env has ever generated began at exactly ``_START_POSE``. That is a
    strong, invisible assumption: a policy trained on it has never seen the arm anywhere else
    at step 0, and the scripted expert's transit is solved *from* that pose, so neither has
    ever been asked whether it depends on it. This term asks.

    Uniform per-joint jitter on the six arm joints, clamped into the soft limits. The fingers
    are left alone -- the gripper is a binary command and starting it half-closed is not a
    different initial condition, it is an illegal one.

    **Off by default** (``jitter=0``), so every number measured before 2026-08-09 still
    describes the shipped env. ``RE3SIM_ARM_START_JITTER=<radians>`` turns it on without
    editing a config, which is how it gets swept.
    """
    j = float(os.environ.get("RE3SIM_ARM_START_JITTER", "0.0")) if jitter is None else jitter
    if j <= 0.0:
        return
    robot = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    arm, _ = robot.find_joints("joint[1-6]")
    arm = torch.tensor(arm, device=env.device)

    q = robot.data.default_joint_pos[ids].clone()
    v = torch.zeros_like(q)
    noise = (torch.rand(len(ids), len(arm), device=env.device) * 2.0 - 1.0) * j
    q[:, arm] += noise
    # Clamp into the joint limits. A start pose outside them is not a harder initial
    # condition, it is one PhysX will quietly project back and then the recorded "start" is
    # not the start that was drawn.
    lo = robot.data.soft_joint_pos_limits[ids, :, 0]
    hi = robot.data.soft_joint_pos_limits[ids, :, 1]
    q = q.clamp(lo, hi)
    robot.write_joint_state_to_sim(q, v, env_ids=ids)


def aim_station_camera(
    env,
    env_ids: torch.Tensor | None,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    sensor_name: str = "station_cam",
):
    """Point the workstation camera at its env, after every reset.

    The camera is a scene sensor rather than something each tool mounts for itself, so the
    pose has to live with the scene too. It cannot simply be baked into ``CameraCfg.OffsetCfg``
    at spawn: ``reset_scene_to_default`` restores prim poses, and a camera left at the default
    pose films the floor.

    ⭐ ONE POSE PER CAMERA. ``set_world_poses_from_view`` builds ``arange(count)`` for the
    indices but does NOT broadcast a single row to match, so handing it one pose at
    ``num_envs > 1`` leaves every camera but the first at its spawn pose -- and the render then
    shows the bare ground plane. That bug shipped twice before it was caught, once in the
    renderer and once in the demo recorder, because a single-env test cannot reach it.

    Uses the same call the demo recorder used, deliberately: the round-1 vision dataset was
    recorded through ``set_world_poses_from_view(origins + eye, origins + target)``, and a
    re-derived quaternion that is even slightly different would make every future render
    subtly inconsistent with the data already collected.
    """
    cam = env.scene[sensor_name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    org = env.scene.env_origins[ids]
    cam.set_world_poses_from_view(
        org + torch.tensor(eye, device=env.device),
        org + torch.tensor(target, device=env.device),
        env_ids=ids,
    )
