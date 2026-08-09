# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test + negative controls for ``Rebot-Workstation-PickPlace1-v0``.

Three things get checked, and the second two are the ones that matter:

1. **The env builds and randomises sanely** -- objects land at their rest heights, inside the
   reach annulus, not inside each other and not inside the box.
2. **The box does not move**, and the clutter does not drift, under a null action. This also
   *calibrates* ``DISTURB_TOL``: the tolerance has to be set against the solver's own noise
   floor, which is how ``clutter`` arrived at 2 mm (blocks drift 1 um under a null action).
3. **The success predicate means what it says.** ``placed_mask`` is exercised against states
   that must pass and states that must fail -- resting inside, perched on the rim, held above
   the box, clipped below the floor, in flight. This is LESSONS_INHERITED B1: the slot task's
   predicate bounded z only from below and scored 93.8 % on geometrically impossible states.

.. code-block:: bash

    python -u scripts/test_workstation_env.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test the reconstructed-workstation env.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--resets", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.re3sim import mdp

FAIL = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev, n = e.device, e.num_envs
    obs, _ = env.reset()

    print(f"\n=== observation space: {obs['policy'].shape}")
    check("obs width is the documented 41-D", obs["policy"].shape[1] == 41,
          f"got {obs['policy'].shape[1]}")

    zero = torch.zeros((n, 7), device=dev)
    zero[:, 6] = 1.0  # fingers open; a zero arm action holds the default pose

    # ------------------------------------------------------------------ 1. randomisation
    print("\n=== spawn sanity, over resets")
    rest_z = {mdp.TARGET_NAME: mdp.CUBE_SIZE / 2,
              "rolloftape": mdp.TAPE_HEIGHT / 2,
              "tapemeasure": mdp.TAPEMEASURE_HEIGHT / 2}
    worst_z, min_sep, min_r, max_r, min_box_gap = {}, 1e9, 1e9, 0.0, 1e9
    for _ in range(args_cli.resets):
        env.reset()
        for _ in range(20):           # let everything settle
            env.step(zero)
        pos = {k: mdp.object_pos_local(e, k) for k in mdp.OBJECT_NAMES}
        for k, p in pos.items():
            worst_z[k] = max(worst_z.get(k, 0.0), float((p[:, 2] - rest_z[k]).abs().max()))
            r = torch.linalg.norm(p[:, :2], dim=1)
            min_r, max_r = min(min_r, float(r.min())), max(max_r, float(r.max()))
        names = list(mdp.OBJECT_NAMES)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = torch.linalg.norm(pos[names[i]][:, :2] - pos[names[j]][:, :2], dim=1)
                min_sep = min(min_sep, float(d.min()))
            b = mdp.in_box_frame(e, pos[names[i]])
            # signed distance outside the box footprint (negative = intruding)
            gap = torch.maximum(b[:, 0].abs() - mdp.BOX_OUTER_X / 2,
                                b[:, 1].abs() - mdp.BOX_OUTER_Y / 2)
            min_box_gap = min(min_box_gap, float(gap.min()))

    for k, v in worst_z.items():
        check(f"{k} settles at its rest height", v < 0.004, f"max |dz| = {v * 1000:.2f} mm")
    check("objects do not overlap", min_sep > 0.05, f"closest pair {min_sep * 1000:.1f} mm")
    check("objects stay in the reach annulus", min_r > 0.13 and max_r < 0.33,
          f"r in [{min_r:.3f}, {max_r:.3f}] m")
    check("objects stay out of the box footprint", min_box_gap > 0.0,
          f"closest approach {min_box_gap * 1000:.1f} mm outside")

    # ------------------------------------------------- 2. null-action drift (calibration)
    print("\n=== null-action drift (this calibrates DISTURB_TOL)")
    env.reset()
    for _ in range(20):
        env.step(zero)
    box0 = mdp.object_pos_local(e, "box").clone()
    c0 = {k: mdp.object_pos_local(e, k).clone() for k in mdp.CLUTTER_NAMES}
    for _ in range(400):
        env.step(zero)
    box_d = float(torch.linalg.norm(mdp.object_pos_local(e, "box") - box0, dim=1).max())
    check("the box is immovable (kinematic)", box_d < 1e-6, f"max drift {box_d * 1e6:.2f} um")
    worst = 0.0
    for k, p0 in c0.items():
        d = float(torch.linalg.norm(mdp.object_pos_local(e, k)[:, :2] - p0[:, :2], dim=1).max())
        worst = max(worst, d)
        print(f"       {k}: {d * 1e6:.1f} um over 400 steps")
    check(f"clutter noise floor is well under DISTURB_TOL ({mdp.DISTURB_TOL * 1000:.0f} mm)",
          worst < mdp.DISTURB_TOL / 4, f"worst {worst * 1000:.3f} mm")

    # ------------------------------------------------------- 3. predicate negative controls
    print("\n=== success predicate: it must reject what it should reject")
    cube = e.scene["cube"]
    centers, yaws = mdp.box_centers_local(e), mdp.box_yaws(e)

    def put(dx_box, dy_box, z, vel=0.0):
        """Teleport the cube to an offset expressed in the BOX's own frame."""
        c, s = torch.cos(yaws), torch.sin(yaws)
        x = centers[:, 0] + c * dx_box - s * dy_box
        y = centers[:, 1] + s * dx_box + c * dy_box
        st = torch.zeros(n, 13, device=dev)
        st[:, :3] = e.scene.env_origins
        st[:, 0] += x
        st[:, 1] += y
        st[:, 2] += z
        st[:, 6] = 1.0
        st[:, 9] = vel  # vertical velocity
        cube.write_root_state_to_sim(st)
        e.sim.forward()
        e.scene.update(e.physics_dt)
        return mdp.placed_mask(e)

    z_rest = mdp.BOX_FLOOR_THICKNESS + mdp.CUBE_SIZE / 2  # 0.034
    zero_t = torch.zeros(n, device=dev)
    check("POSITIVE: resting on the interior floor, centred",
          bool(put(zero_t, zero_t, z_rest).all()))
    off = torch.full((n,), mdp.BOX_INNER_X / 2 - mdp.PLACED_MARGIN - 0.005, device=dev)
    check("POSITIVE: resting inside, near the long wall",
          bool(put(off, zero_t, z_rest).all()))
    check("NEGATIVE: perched on the rim", not bool(put(zero_t, zero_t,
          torch.full((n,), mdp.BOX_HEIGHT + mdp.CUBE_SIZE / 2, device=dev)).any()))
    check("NEGATIVE: held in the gripper above the box", not bool(put(zero_t, zero_t,
          torch.full((n,), 0.16, device=dev)).any()))
    check("NEGATIVE: clipped below the interior floor", not bool(put(zero_t, zero_t,
          torch.full((n,), -0.02, device=dev)).any()))
    check("NEGATIVE: in flight over the box (not settled)",
          not bool(put(zero_t, zero_t, z_rest, vel=-1.0).any()))
    out = torch.full((n,), mdp.BOX_OUTER_X / 2 + 0.05, device=dev)
    check("NEGATIVE: sitting on the desk beside the box",
          not bool(put(out, zero_t, mdp.CUBE_SIZE / 2).any()))
    # the yaw check: a point outside a ROTATED box that an axis-aligned test would accept
    check("NEGATIVE: outside the rotated box, where an axis-aligned test would say inside",
          not bool(put(out, out, z_rest).any()))

    print("\n" + "=" * 62)
    if FAIL:
        print(f"FAILED {len(FAIL)}: " + ", ".join(FAIL))
    else:
        print("ALL CHECKS PASSED")
    print("=" * 62)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
