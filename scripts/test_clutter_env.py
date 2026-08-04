# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the clutter-extraction env.

Checks the row geometry is the pitch the config claims (measured off the settled blocks, not
read back out of the config), that everything starts upright and undisturbed, and that the
no-topple and no-disturbance constraints -- the things that make this a clutter task rather
than a pick task -- actually fire, and fire only when they should.

The V7 block and negative control (d) were added on 2026-08-03 with ``mdp.DISTURB_TOL``.
Control (d) is the one that matters: an upright neighbour dragged into the goal zone with the
target used to pass ``target_at_goal`` cleanly, and this file asserted the underlying
behaviour was correct ("shoved 30 mm but upright: not toppled"). It was correct about
``any_distractor_toppled`` and silent about whether toppling was the right constraint.

The V8 block was added on 2026-08-04 with ``mdp.reset_clutter_row``, which spawns the row at
a random heading and puts the target in a random slot. It checks the three separable claims
-- the placement is rigid, every slot is drawn, and nothing leaves the arm's envelope -- and
V1 now measures the pitch along the row's **own** axis, since a world-y projection would
report ``pitch * cos(yaw)`` and fail a correct row.

Every check appends to ``failures`` instead of asserting, so one run reports everything.

.. code-block:: bash

    python scripts/test_clutter_env.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the clutter-extraction env.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401  (registers the tasks)
from reBot_RL.tasks.manager_based.challenge import mdp

EXPECTED_DIM = 8 + 8 + 7 + 12 + 7  # 42
#: 90 deg about x -- lays a block on its side
LYING = [0.7071, 0.0, 0.0, 0.7071]
#: kinematic design bound on the row: every block inside r <= 0.32 m (docs C4)
R_MAX = 0.32


#: The two ways to measure the row's heading disagree, and both are needed. **Measured off
#: the settled blocks either way, never read out of the config or off ``_clutter_row_yaw``.**
#:
#: * from the blocks' **orientations** -- exact, because the distractors get no yaw jitter,
#:   so each one's yaw *is* the row's. This is what catches "the row was translated but not
#:   rotated", and it is the axis worth projecting onto.
#: * from the blocks' **centres** -- the dominant principal direction, and only good to
#:   ~43 mrad: the +/-10 mm fore-aft jitter tilts a 168 mm baseline by
#:   ``(0.010/sqrt(3)) / 0.133``. That is not a defect to fix, it is a property of the task
#:   (a policy reading only ``clutter_obs``, which carries no orientations, sees the heading
#:   through the same noise). It catches the converse: blocks rotated but laid out along a
#:   line that is not their own axis.
YAW_FROM_POS_SD = 0.043


def row_yaw_measured(e) -> torch.Tensor:
    """Row heading from the distractors' own orientations. Exact to the settle noise."""
    y = torch.stack([mdp.yaw_of(mdp.object_quat(e, d)) for d in mdp.DISTRACTOR_NAMES], dim=1)
    return y.median(dim=1).values


def row_yaw_from_centres(p: torch.Tensor) -> torch.Tensor:
    """Row heading from the five block centres alone. ``p`` is (n, 5, 2)."""
    q = p - p.mean(dim=1, keepdim=True)
    u = torch.linalg.eigh(q.transpose(1, 2) @ q)[1][..., -1]   # dominant, sign arbitrary
    u = u * torch.sign(u[:, 1:2])                              # point along the row's +y
    return torch.atan2(-u[:, 0], u[:, 1])                      # +y maps to (-sin, cos)


def along_row(p: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Each block's coordinate along the row axis implied by ``yaw``. -> (n, 5)."""
    u = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=1)
    return ((p - p.mean(dim=1, keepdim=True)) * u.unsqueeze(1)).sum(dim=-1)


def main() -> None:
    failures: list[str] = []
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    n = e.num_envs
    names = ("target",) + mdp.DISTRACTOR_NAMES

    def put(name, pos, quat=None):
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).repeat(n, 1)
        if quat is None:
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        e.scene[name].write_root_state_to_sim(
            torch.cat([pos + e.scene.env_origins, quat, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def zero_step(k):
        a = torch.zeros(n, 7, device=dev)
        for _ in range(k):
            env.step(a)

    with torch.inference_mode():
        obs, _ = env.reset()
        pol = obs["policy"]

        # ---- V6: observation plumbing ---------------------------------------
        if pol.shape[1] != EXPECTED_DIM:
            failures.append(f"obs dim {pol.shape[1]} != {EXPECTED_DIM}")
        if not torch.isfinite(pol).all():
            failures.append("observation contains non-finite values")
        if mdp.clutter_obs(e).shape != (n, 12):
            failures.append(f"clutter_obs shape {tuple(mdp.clutter_obs(e).shape)} != {(n, 12)}")

        # ---- V3 / V1: the row settles at the pitch the config claims ----------
        # Measured along the ROW's own axis, not along world y: since 2026-08-04 the row
        # spawns at a random heading, so a world-y projection would report the pitch times
        # cos(yaw) and call a correct row a failing one.
        zero_step(30)
        xy = torch.stack([mdp.object_pos_local(e, m)[:, :2] for m in names], dim=1)  # (n,5,2)
        zs = torch.stack([mdp.object_pos_local(e, m)[:, 2] for m in names], dim=1)
        yaw_meas = row_yaw_measured(e)
        s_sorted = along_row(xy, yaw_meas).sort(dim=1).values
        spacing = s_sorted[:, 1:] - s_sorted[:, :-1]                                 # (n,4)
        free_gap = spacing - 2 * mdp.CL_BLOCK_HALF[1]
        print(f"[V1] settled row pitch {[f'{float(s) * 1000:.1f}' for s in spacing[0]]} mm (env 0)")
        print(f"[V1] free gap between 30 mm blocks, over {n} envs: "
              f"{float(free_gap.min()) * 1000:.1f} - {float(free_gap.max()) * 1000:.1f} mm")
        if float(free_gap.min()) <= 0.0:
            failures.append("blocks overlap in the row -- the pitch is narrower than the blocks")
        if not ((zs > 0.020) & (zs < 0.050)).all():
            failures.append(f"blocks did not settle on the table: z in [{float(zs.min()):.4f}, "
                            f"{float(zs.max()):.4f}]")

        # ---- V8: the row randomisation (2026-08-04) ---------------------------
        # Three separable claims: the row is placed rigidly at a random heading, the target
        # can be any of the five blocks, and neither takes a block out of the arm's reach.
        r = xy.norm(dim=-1)
        print(f"[V8] block radius from the arm base: {float(r.min()):.3f} - {float(r.max()):.3f} m")
        if float(r.max()) > R_MAX:
            failures.append(f"a block spawned at r = {float(r.max()):.3f} m, outside the "
                            f"r <= {R_MAX} m design envelope (docs C4)")

        # the heading is real, it varies, and the event's own record agrees with the geometry
        d_yaw = float((yaw_meas - e._clutter_row_yaw).abs().max())
        d_ctr = float((row_yaw_from_centres(xy) - yaw_meas).abs().max())
        print(f"[V8] row heading: measured {float(yaw_meas.min()):+.3f} to "
              f"{float(yaw_meas.max()):+.3f} rad, sd {float(yaw_meas.std()):.3f}; "
              f"max |measured - recorded| {d_yaw * 1e3:.2f} mrad; "
              f"max |from centres - from orientations| {d_ctr * 1e3:.1f} mrad")
        if d_yaw > 0.005:
            failures.append(f"the blocks' own yaw disagrees with _clutter_row_yaw by "
                            f"{d_yaw:.3f} rad -- the row is translated but not rotated")
        # 4 sigma of the centre-fit noise; a genuinely misaligned layout misses by far more
        if d_ctr > 4 * YAW_FROM_POS_SD:
            failures.append(f"the row's centres lie along {d_ctr:.3f} rad away from the axis "
                            "the blocks are turned to -- the layout is not rigid")
        if float(yaw_meas.std()) < 0.05:
            failures.append(f"row heading sd {float(yaw_meas.std()):.3f} rad -- the row is not "
                            "actually being rotated")

        # A rigid transform leaves the pitch alone, so at EVERY heading the spacing must stay
        # inside the band the per-block jitter alone permits. A hard bound, not a statistical
        # one: two neighbours can differ by at most +/- 2 x DISTRACTOR_JITTER_Y.
        band = 2 * mdp.DISTRACTOR_JITTER_Y + 0.001
        if float(spacing.min()) < mdp.ROW_PITCH - band or float(spacing.max()) > mdp.ROW_PITCH + band:
            failures.append(f"row spacing spans {float(spacing.min()) * 1000:.1f}-"
                            f"{float(spacing.max()) * 1000:.1f} mm, outside the "
                            f"{(mdp.ROW_PITCH - band) * 1000:.0f}-{(mdp.ROW_PITCH + band) * 1000:.0f} mm "
                            "the jitter alone allows -- the row transform is not rigid")

        # the target's rank along the row IS its slot, and every slot must be reachable
        slots_seen = set()
        for _ in range(8):
            env.reset()
            zero_step(4)
            p = torch.stack([mdp.object_pos_local(e, m)[:, :2] for m in names], dim=1)
            s = along_row(p, row_yaw_measured(e))
            rank = s.argsort(dim=1).argsort(dim=1)[:, 0]        # `names[0]` is the target
            if not bool((rank == e._clutter_target_slot).all()):
                failures.append("the target's position along the row does not match "
                                "_clutter_target_slot -- the slot assignment is wrong")
            slots_seen |= {int(v) for v in rank.unique()}
        print(f"[V8] target slots drawn over {8 * n} spawns: {sorted(slots_seen)}")
        if len(slots_seen) != mdp.N_SLOTS:
            failures.append(f"only slots {sorted(slots_seen)} were ever the target, "
                            f"want all {mdp.N_SLOTS}")

        # and the frozen layout that -Fixed-v0 / -Lenient-v0 pin really is frozen
        ids = torch.arange(n, device=dev)
        mdp.reset_clutter_row(e, ids, random_slot=False, row_yaw=0.0, row_xy=0.0)
        e.sim.forward()
        e.scene.update(e.physics_dt)
        yaw_f = row_yaw_measured(e)
        if float(yaw_f.abs().max()) > 0.005 or not bool((e._clutter_target_slot == 2).all()):
            failures.append("the FIXED_ROW parameters did not pin the row square with the "
                            "target in the middle slot")
        print(f"[V8] FIXED_ROW: heading {float(yaw_f.abs().max()) * 1e3:.2f} mrad, "
              f"target slot {int(e._clutter_target_slot[0])} in every env")
        env.reset()
        zero_step(20)

        # ---- V5: the no-topple constraint, negative case first ----------------
        if mdp.any_distractor_toppled(e).any():
            failures.append("a distractor read as toppled at reset")
        dist0 = float(mdp.distractors_disturbed(e).max())
        print(f"[V5-] at reset: nothing toppled, disturbance {dist0 * 1000:.2f} mm")
        if dist0 > 0.005:
            failures.append(f"distractors_disturbed is {dist0 * 1000:.2f} mm at reset, want ~0 "
                            "-- record_spawn probably did not run last")

        # ---- V5: and the positive case -- it must actually be reachable -------
        d0 = mdp.DISTRACTOR_NAMES[0]
        p0 = mdp.object_pos_local(e, d0)[0].clone()
        put(d0, torch.tensor([float(p0[0]), float(p0[1]), 0.018], device=dev),
            torch.tensor(LYING, device=dev).repeat(n, 1))
        if not mdp.any_distractor_toppled(e).all():
            failures.append(f"any_distractor_toppled did not fire for a {d0} lying on its side "
                            f"-- TOPPLE_DOT = {mdp.TOPPLE_DOT} may be unreachable")
        print(f"[V5+] a distractor laid on its side: toppled "
              f"{int(mdp.any_distractor_toppled(e).sum())}/{n}")

        # a distractor merely SHOVED aside, still upright, must NOT count as toppled
        put(d0, torch.tensor([float(p0[0]), float(p0[1]) - 0.030, float(p0[2])], device=dev))
        if mdp.any_distractor_toppled(e).any():
            failures.append("negative control: a shoved-but-upright distractor read as toppled")
        shove = float(mdp.distractors_disturbed(e).max())
        print(f"[V5-] shoved 30 mm but upright: not toppled, disturbance {shove * 1000:.1f} mm")
        if shove < 0.020:
            failures.append("distractors_disturbed did not register a 30 mm shove")

        env.reset()
        zero_step(20)
        # re-read the distractor's pose: the reset re-jittered it, and `record_spawn`
        # re-recorded the spawn against the NEW pose. Putting it back at the pre-reset `p0`
        # would read as several mm of displacement, which since 2026-08-03 is itself a
        # failure of `target_at_goal` -- negative control (c) below would then pass for the
        # wrong reason and stop testing the topple constraint it is named for.
        #
        # Kept PER ENV, unlike the pre-reset `p0` above: `put` broadcasts a 1-D position to
        # every env, so restoring all 16 to env 0's pose is itself a multi-millimetre
        # displacement now that displacement is a constraint.
        p0 = mdp.object_pos_local(e, d0).clone()

        # ---- V7: the no-DISTURBANCE constraint, both directions ---------------
        # Added with `DISTURB_TOL`. Three controls, because a threshold needs to be shown
        # both to fire and not to fire: at rest it must be silent, below tol it must stay
        # silent, and above tol it must bind success.
        if mdp.any_distractor_disturbed(e).any():
            failures.append("any_distractor_disturbed fired on an undisturbed row at reset")
        print(f"[V7-] at reset: max displacement "
              f"{float(mdp.max_distractor_displacement(e).max()) * 1000:.3f} mm, not disturbed")

        def shift_d0(dy):
            off = torch.zeros_like(p0)
            off[:, 1] = dy
            put(d0, p0 + off)

        half_tol = mdp.DISTURB_TOL / 2.0
        shift_d0(half_tol)
        if mdp.any_distractor_disturbed(e).any():
            failures.append(f"any_distractor_disturbed fired for a {half_tol * 1000:.1f} mm nudge, "
                            f"below DISTURB_TOL = {mdp.DISTURB_TOL * 1000:.1f} mm")
        print(f"[V7-] nudged {half_tol * 1000:.1f} mm (below tol): correctly not disturbed")

        shift_d0(2 * mdp.DISTURB_TOL)
        if not mdp.any_distractor_disturbed(e).all():
            failures.append(f"any_distractor_disturbed did not fire for a "
                            f"{2 * mdp.DISTURB_TOL * 1000:.1f} mm shove")
        if mdp.any_distractor_toppled(e).any():
            failures.append("a shoved-but-upright distractor read as toppled -- the two "
                            "constraints are not independent")
        print(f"[V7+] shoved {2 * mdp.DISTURB_TOL * 1000:.1f} mm, still upright: disturbed "
              f"{int(mdp.any_distractor_disturbed(e).sum())}/{n}, toppled "
              f"{int(mdp.any_distractor_toppled(e).sum())}/{n}")
        put(d0, p0)

        # ---- V5: extraction and goal predicates -------------------------------
        tp = mdp.object_pos_local(e, "target")[0].clone()
        put("target", torch.tensor([float(tp[0]), float(tp[1]), 0.120], device=dev))
        if not mdp.target_extracted(e).all():
            failures.append("target_extracted did not fire for a target lifted clear of the row")

        gx, gy = mdp.GOAL_XY
        put("target", torch.tensor([gx, gy, 0.035], device=dev))
        if not mdp.target_at_goal(e).all():
            failures.append("target_at_goal did not fire for a target set down in the goal zone")
        print(f"[V5+] target in the goal zone: {int(mdp.target_at_goal(e).sum())}/{n}")

        # (a) at the goal in xy but still held high
        put("target", torch.tensor([gx, gy, 0.120], device=dev))
        if mdp.target_at_goal(e).any():
            failures.append("negative control (a): target_at_goal fired for a target held above the goal")

        # (b) set down, but well outside the goal radius
        put("target", torch.tensor([gx + 3 * mdp.GOAL_RADIUS, gy, 0.035], device=dev))
        if mdp.target_at_goal(e).any():
            failures.append("negative control (b): target_at_goal fired outside the goal radius")

        # (c) perfectly at the goal, but a distractor is down -- the constraint must dominate
        put("target", torch.tensor([gx, gy, 0.035], device=dev))
        lay = p0.clone()
        lay[:, 2] = 0.018   # per env: `p0` is (n, 3) since the V7 block above
        put(d0, lay, torch.tensor(LYING, device=dev).repeat(n, 1))
        if mdp.target_at_goal(e).any():
            failures.append("negative control (c): target_at_goal fired while a distractor was "
                            "toppled -- the no-topple constraint does not bind the success test")
        if mdp.target_extracted(e).any():
            failures.append("negative control (c): target_extracted ignores a toppled distractor")
        print("[V5-] target at the goal but a distractor down: correctly not a success")

        # (d) perfectly at the goal, distractor UPRIGHT but dragged to the goal with it.
        # This is the case the task was silently scoring as a full success until
        # 2026-08-03: it was worth 42 points of the measured expert's rate.
        put(d0, torch.tensor([gx, gy + 0.06, 0.035], device=dev))
        if mdp.any_distractor_toppled(e).any():
            failures.append("control (d) is not testing what it claims: the dragged distractor "
                            "toppled, so the topple constraint would have caught it anyway")
        if mdp.target_at_goal(e).any():
            failures.append("negative control (d): target_at_goal fired while an UPRIGHT "
                            "distractor had been dragged to the goal zone -- the "
                            "no-disturbance constraint does not bind the success test")
        print(f"[V5-] target at the goal but an upright distractor dragged "
              f"{float(mdp.max_distractor_displacement(e).max()) * 1000:.0f} mm with it: "
              "correctly not a success")
        put(d0, p0)

        # ---- V6: rewards are finite -------------------------------------------
        for fn, kw in ((mdp.reach_target, {"std": 0.10, "ee_frame_cfg": SceneEntityCfg("ee_frame")}),
                       (mdp.target_to_goal, {"std": 0.12}),
                       (mdp.distractors_disturbed, {}),
                       (mdp.clutter_obs, {})):
            v = fn(e, **kw)
            if not torch.isfinite(v).all():
                failures.append(f"{fn.__name__} produced non-finite values")

        # ---- V6: terminations --------------------------------------------------
        put("target", torch.tensor([0.250, 0.0, -0.50], device=dev))
        if not mdp.block_dropped(e, minimum_height=-0.05, name="target").all():
            failures.append("target_dropped did not fire for a target below the table")

        env.reset()
        _, _, term, _, _ = env.step(torch.zeros(n, 7, device=dev))
        if bool(term.any()):
            failures.append(f"{int(term.sum())}/{n} envs terminated on the first step after reset "
                            "-- the row probably spawns already toppled")

    print("\n" + "=" * 70)
    if failures:
        print("[result] FAIL")
        for f in failures:
            print("  - " + f)
    else:
        print("[result] PASS -- row geometry, the row randomisation, the topple AND")
        print("         no-disturbance constraints (each with a positive and a negative")
        print("         control), the goal predicate and four negative controls all check out.")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
