# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Why are workstation objects spawning on top of each other?

``test_workstation_env.py`` reports the symptom *after* physics has run, which conflates two
different things: a bad spawn, and a good spawn that something later disturbed. This reads the
positions **the instant the reset event writes them**, before a single physics step, and
reports the rejection sampler's own internal state.

Three things it answers:

1. Are the objects overlapping *at spawn*, or only after settling?
2. How often does the guaranteed-free ``FALLBACK_SLOTS`` path fire? If it dominates, the
   annulus is too small and the fix is more room, not a better fallback.
3. Which pair is the offender, and by how much?

.. code-block:: bash

    python -u scripts/diag_workstation_spawn.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose the workstation spawn sampler.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--resets", type=int, default=6)
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
from reBot_RL.tasks.manager_based.re3sim.mdp.events import FOOTPRINT_R, REST_Z


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n = e.num_envs
    env.reset()

    names = list(mdp.OBJECT_NAMES)
    total_bad = 0
    for k in range(args_cli.resets):
        e._spawn_fallbacks = {}
        env.reset()
        # read positions BEFORE stepping physics
        pos = {nm: mdp.object_pos_local(e, nm).clone() for nm in names}

        print(f"\n--- reset {k}  (fallbacks fired: {e._spawn_fallbacks or 'none'})")
        for i in range(len(names)):
            p = pos[names[i]]
            r = torch.linalg.norm(p[:, :2], dim=1)
            dz = (p[:, 2] - REST_Z[names[i]]).abs()
            print(f"    {names[i]:12s} r in [{float(r.min()):.3f}, {float(r.max()):.3f}]  "
                  f"max|dz| {float(dz.max()) * 1000:5.1f} mm")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                need = FOOTPRINT_R[names[i]] + FOOTPRINT_R[names[j]]
                d = torch.linalg.norm(pos[names[i]][:, :2] - pos[names[j]][:, :2], dim=1)
                bad = int((d < need).sum())
                total_bad += bad
                flag = "  <== OVERLAP" if bad else ""
                print(f"    {names[i]:12s} vs {names[j]:12s}: min {float(d.min()) * 1000:6.1f} mm "
                      f"(need {need * 1000:.1f})  violations {bad}/{n}{flag}")

    print(f"\n[diag] total pairwise violations at spawn: {total_bad}")
    print("[diag] if this is ZERO, the sampler is fine and the smoke test's failures come")
    print("       from physics AFTER spawn -- look at the settle loop, not the sampler.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
