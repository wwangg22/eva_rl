#!/usr/bin/env python
"""Gate: the -VisionDR task's DYNAMICS are the -Vision task's, bit for bit.

The whole visual-DR design rests on one claim: the randomisation is appearance-only, so
the scripted expert's measured success carries over unchanged. This probe tests that claim
WITHOUT the expert (05_VISUAL_DR.md phase-2 gate): same seed, same scripted action
sequence, on each task in its own process — if any "visual-only" prim leaked a collider,
or a DR event touched sim state, the 41-D observation trajectories diverge.

Run once per task, then compare:

    for T in Vision VisionDR; do
      RE3SIM_SPLATS_PER_ENV=1 python -u scripts/probe_vdr_invariance.py --headless \
          --task Rebot-Workstation-PickPlace1-$T-v0 --out /tmp/traj_$T.pt
    done
    python scripts/compare_vdr_traj.py /tmp/traj_Vision.pt /tmp/traj_VisionDR.pt

Two processes, not one: tearing down a ManagerBasedRLEnv and building a second in the same
Kit app is exactly the kind of half-supported path that produces its own diffs.

Also records, on the DR task (harmless zeros on the stock task):
  * per-env station-cam poses and focal lengths after two consecutive resets — across-env
    and across-reset VARIETY, and
  * the same after re-seeding torch — DETERMINISM of the draws (collection replays them
    from the run seed alone).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a scripted-action trajectory.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402


def _cam_state(e):
    """Station-cam world poses + focal lengths, or zeros when the sensor is absent."""
    try:
        cam = e.scene["station_cam"]
    except KeyError:
        return torch.zeros(e.num_envs, 8)
    import warp as wp  # noqa: PLC0415

    pos, quat = cam._view.get_world_poses()
    to_t = lambda a: wp.to_torch(a) if isinstance(a, wp.array) else a  # noqa: E731
    import omni.usd  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    tmpl = cam.cfg.prim_path
    foc = torch.tensor([
        float(stage.GetPrimAtPath(tmpl.replace("env_.*", f"env_{i}"))
              .GetAttribute("focalLength").Get() or 0.0)
        for i in range(e.num_envs)
    ])
    # env-LOCAL positions: with world positions the across-env variety stat is dominated
    # by the metres-apart env origins and would pass even if every env drew ONE shared
    # pose (review 2026-08-11)
    pos_local = to_t(pos).cpu() - e.scene.env_origins.cpu()
    return torch.cat([pos_local, to_t(quat).cpu(), foc.unsqueeze(1)], dim=1)


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5   # the probe owns the horizon; no mid-run resets
    env_cfg.seed = 7
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n, dev = e.num_envs, e.device

    # --- DR draw determinism + variety (before the dynamics run, so the RNG state the
    # dynamics run starts from is exactly what a re-seeded collection would see) ---------
    torch.manual_seed(7)
    env.reset()
    cam_a1 = _cam_state(e)
    env.reset()
    cam_a2 = _cam_state(e)
    torch.manual_seed(7)
    env.reset()
    cam_b1 = _cam_state(e)

    # --- scripted dynamics run ----------------------------------------------------------
    torch.manual_seed(7)
    obs, _ = env.reset()
    obs = obs["policy"] if isinstance(obs, dict) else obs
    traj = [obs.cpu().clone()]
    for t in range(args_cli.steps):
        a = torch.zeros(n, 7, device=dev)
        # a deterministic sweep that moves every joint and works the gripper: enough
        # contact-free motion to expose any phantom collider in the arm's swept volume
        for j in range(6):
            a[:, j] = 0.35 * math.sin(0.02 * t + 0.9 * j)
        a[:, 6] = 1.0 if (t // 60) % 2 == 0 else -1.0
        obs, _, _, _, _ = env.step(a)
        obs = obs["policy"] if isinstance(obs, dict) else obs
        traj.append(obs.cpu().clone())

    torch.save({
        "task": args_cli.task,
        "traj": torch.stack(traj),          # (T+1, n, 41)
        "cam_reset1": cam_a1, "cam_reset2": cam_a2, "cam_reseeded": cam_b1,
    }, args_cli.out)
    print(f"[invariance] wrote {args_cli.out}: traj {tuple(torch.stack(traj).shape)}",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
