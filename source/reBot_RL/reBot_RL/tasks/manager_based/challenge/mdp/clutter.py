# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the clutter-extraction task.

A target block sits in a tight row of four distractors. **Moving** any distractor ends the
episode. The row spacing is narrower than the gripper's outer width when open, so the
policy either threads the fingers in precisely or first pushes a neighbour aside -- the
correct first action is often *not* a grasp, which is the credit-assignment problem plain
pick-and-place never poses.

Amended 2026-08-03 -- the constraint used to be *toppling*, and that was too weak
--------------------------------------------------------------------------------
``target_at_goal`` originally ended in ``& ~any_distractor_toppled(env)``, and toppling is
``up_z < 0.75``, i.e. about 41 degrees of tilt. A neighbour dragged the length of the table
and set down upright satisfied that predicate completely. Measured on 768 held-out episodes
of a scripted expert scoring 73.3 % under the old rule: the median "success" displaced a
neighbour by **13.7 mm** -- more than the row's own 12 mm free gap -- and in **22-25 % of
all episodes a neighbour was carried into the goal zone along with the target** (the p90 of
the displacement was ~206 mm, against 120-277 mm from the row to the goal).

The quantity needed to catch this was already computed every step: ``distractors_disturbed``
was wired to a shaping reward and to nothing else. It is now also a constraint, at
``DISTURB_TOL``. See ``eva_bc/clutter/docs/14_FEEDBACK_AND_NEXT.md``.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from . import common

#: names of the four distractor rigid objects, in row order
DISTRACTOR_NAMES = ("distractor_0", "distractor_1", "distractor_2", "distractor_3")
#: block half-extents (x, y, z) [m] -- 30 mm across the fingers, in the measured sweet spot
CL_BLOCK_HALF = (0.018, 0.015, 0.035)
#: a block counts as toppled once its own +z axis falls this far from world +z
TOPPLE_DOT = 0.75
#: a neighbour counts as *disturbed* once it slides this far in the plane from where it
#: spawned [m]. 2 mm, and the number is a judgement about the task rather than a
#: measurement -- but it is bracketed on both sides by measurements:
#:
#:   * the floor: between the reset write and the first control step the blocks settle by
#:     **< 0.1 mm** (measured -- it is why the spawn fingerprints in the demo collector
#:     round at 0.1 mm), so 2 mm is ~20x the solver's own noise and cannot fire by itself;
#:   * the ceiling: the free gap between neighbours is **12 mm**, so 2 mm is a sixth of the
#:     space a block has to move in, and reads as "it was not touched" rather than "it was
#:     nudged but stayed in its slot" (which is nearer 10 mm).
DISTURB_TOL = 0.002
#: goal: carry the target here (env-local xy) and set it down
GOAL_XY = (0.185, -0.185)
GOAL_RADIUS = 0.045
#: the target must clear the row before it counts as extracted
EXTRACT_Z = 0.090


def _up_z(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    q = common.object_quat(env, name)
    up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3))
    return up[:, 2]


def any_distractor_toppled(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination: the constraint that gives the task its teeth."""
    return torch.stack([_up_z(env, n) < TOPPLE_DOT for n in DISTRACTOR_NAMES], dim=1).any(dim=1)


def distractor_displacements(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-distractor planar displacement from its spawn pose [m]. Shape (N, 4).

    Zero before the first reset event has run, so every term built on this is safe to call
    during env construction.
    """
    if not hasattr(env, "_clutter_spawn_xy"):
        return torch.zeros(env.num_envs, len(DISTRACTOR_NAMES), device=env.device)
    cur = torch.stack([common.object_pos_local(env, n)[:, :2] for n in DISTRACTOR_NAMES], dim=1)
    return (cur - env._clutter_spawn_xy).norm(dim=-1)


def distractors_disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Total planar displacement of the distractors from their spawn poses [m].

    The dense shaping signal for "be gentle". It stays a *sum* over the four blocks, and it
    stays continuous, because that is what makes it useful as a gradient; the hard version
    of the same idea is ``any_distractor_disturbed``.
    """
    return distractor_displacements(env).sum(dim=1)


def max_distractor_displacement(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Worst single-distractor planar displacement [m]. Shape (N,).

    The *max*, not the sum, is the right statistic for the constraint: the task is "no
    neighbour was moved", so four blocks nudged 0.5 mm each must not read the same as one
    block shoved 2 mm.
    """
    return distractor_displacements(env).max(dim=1).values


def any_distractor_disturbed(
    env: ManagerBasedRLEnv, tol: float = DISTURB_TOL
) -> torch.Tensor:
    """Termination: a neighbour was moved. The constraint the task is actually about.

    Displacement from spawn is very nearly monotone -- a shoved block does not slide back --
    so reading it instantaneously and terminating on it is equivalent to latching it, and it
    ends the episode at the moment the run stops being a clean extraction rather than
    letting it play on to a success that no longer counts.
    """
    return max_distractor_displacement(env) > tol


def target_extracted(env: ManagerBasedRLEnv, name: str = "target") -> torch.Tensor:
    """Target lifted clear of the row, with every distractor still standing."""
    return (common.object_pos_local(env, name)[:, 2] > EXTRACT_Z) & ~any_distractor_toppled(env)


def target_at_goal(
    env: ManagerBasedRLEnv, name: str = "target", tol: float = DISTURB_TOL
) -> torch.Tensor:
    """Target set down inside the goal zone with the row left **undisturbed**.

    ``~any_distractor_disturbed`` subsumes ``~any_distractor_toppled`` in every case that
    matters -- a block cannot tip 41 degrees about a 36x30 mm footprint without its centre
    moving several millimetres -- but both are kept, because they are separate statements
    and the topple term is what ends the episode when a block is knocked over cleanly out of
    the plane.

    Pass ``tol=float("inf")`` to recover the pre-2026-08-03 predicate. That is the only way
    to reproduce any success number in ``docs/`` dated before that, and it is what the
    ``-Lenient-v0`` variant does.
    """
    p = common.object_pos_local(env, name)
    goal = torch.tensor(GOAL_XY, device=p.device)
    return (
        ((p[:, :2] - goal).norm(dim=1) < GOAL_RADIUS)
        & (p[:, 2] < 0.055)
        & ~any_distractor_toppled(env)
        & ~any_distractor_disturbed(env, tol)
    )


def reach_target(env: ManagerBasedRLEnv, std: float, ee_frame_cfg: SceneEntityCfg,
                 name: str = "target") -> torch.Tensor:
    d = (common.ee_pos_local(env, ee_frame_cfg) - common.object_pos_local(env, name)).norm(dim=1)
    return 1.0 - torch.tanh(d / std)


def target_to_goal(env: ManagerBasedRLEnv, std: float, name: str = "target") -> torch.Tensor:
    """Gated carry reward: only pays once the target is clear of the row."""
    p = common.object_pos_local(env, name)
    goal = torch.tensor(GOAL_XY, device=p.device)
    d = (p[:, :2] - goal).norm(dim=1)
    return (p[:, 2] > 0.070).float() * (1.0 - torch.tanh(d / std))


def clutter_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distractor xy offsets from the target, plus their up-axes. Shape (N, 12)."""
    tp = common.object_pos_local(env, "target")
    feats = []
    for n in DISTRACTOR_NAMES:
        feats.append(common.object_pos_local(env, n)[:, :2] - tp[:, :2])
        feats.append(_up_z(env, n).unsqueeze(-1))
    return torch.cat(feats, dim=-1)


class clutter_success(ManagerTermBase):
    """Sparse per-step success; logs the episode success rate on reset."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {})["Metrics/clutter_success_rate"] = (
            self._ever[env_ids].float().mean()
        )
        self._ever[env_ids] = False

    def __call__(
        self, env: ManagerBasedRLEnv, name: str = "target", tol: float = DISTURB_TOL
    ) -> torch.Tensor:
        ok = target_at_goal(env, name, tol)
        self._ever.copy_(self._ever | ok)
        return ok.float()


def record_spawn_xy(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset event: remember where the distractors started, for ``distractors_disturbed``."""
    cur = torch.stack([common.object_pos_local(env, n)[:, :2] for n in DISTRACTOR_NAMES], dim=1)
    if not hasattr(env, "_clutter_spawn_xy"):
        env._clutter_spawn_xy = cur.clone()
    else:
        env._clutter_spawn_xy[env_ids] = cur[env_ids]
