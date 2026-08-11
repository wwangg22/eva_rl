#!/usr/bin/env python
"""Gate for the visual-DR camera path + a stills grid to eyeball every DR axis.

Two jobs, in order, because the second is meaningless if the first fails:

1. **Bit-exactness probe.** `visual_dr.randomize_station_camera` builds its pose through
   `create_rotation_matrix_from_view` + a roll post-multiply + `set_world_poses(...,
   convention="opengl")`, while the stock aim goes through `set_world_poses_from_view`.
   Quaternion layout (wxyz vs xyzw) and convention conversions are exactly the kind of
   thing this repo has been bitten by, so equality at zero DR is ASSERTED, not assumed:
   the DR path with all magnitudes forced to zero must land every camera on the pose the
   stock path produces, to 1e-5.

2. **Stills grid.** Reset the -VisionDR task several times and save every env's station
   and wrist frames. The DR is appearance-only and nothing in the MDP reads appearance, so
   a wrong magnitude, a dead axis or a black frame is invisible to every automated check —
   the grid exists to be LOOKED AT (docs/envs/re3sim/05_VISUAL_DR.md logs the verdicts).

.. code-block:: bash

    RE3SIM_SPLATS_PER_ENV=1 python -u scripts/probe_vdr_camera.py --headless \
        --num_envs 8 --resets 4 --out /tmp/vdr_grid
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe + render the visual-DR vision task.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-VisionDR-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--resets", type=int, default=4, help="DR draws to render per env")
parser.add_argument("--settle", type=int, default=60, help="env steps before each render")
parser.add_argument("--out", type=str, default="/tmp/vdr_grid")
parser.add_argument("--skip-probe", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.re3sim import mdp  # noqa: E402
from reBot_RL.tasks.manager_based.re3sim.workstation_env_cfg import (  # noqa: E402
    STATION_CAM_EYE, STATION_CAM_TARGET,
)

# Higher than the 160x120 training default: these frames exist to be inspected by a human.
# 8 envs x 2 cams x 320x240 is well under the measured render-product ceiling.
os.environ.setdefault("RE3SIM_CAM_W", "320")
os.environ.setdefault("RE3SIM_CAM_H", "240")


def _cam_poses(cam):
    pos, quat = cam._view.get_world_poses()
    import warp as wp  # noqa: PLC0415

    to_t = lambda a: wp.to_torch(a) if isinstance(a, wp.array) else a  # noqa: E731
    return to_t(pos).clone(), to_t(quat).clone()


def main():
    torch.manual_seed(0)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env_cfg.seed = 0
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n = e.num_envs
    env.reset()

    cam = e.scene["station_cam"]

    if not args_cli.skip_probe:
        # ---- 1. zero-DR bit-exactness --------------------------------------------------
        # Stock path first.
        org = e.scene.env_origins
        eyes = org + torch.tensor(STATION_CAM_EYE, device=e.device)
        tgts = org + torch.tensor(STATION_CAM_TARGET, device=e.device)
        cam.set_world_poses_from_view(eyes, tgts)
        pos_ref, quat_ref = _cam_poses(cam)

        # DR path, magnitudes forced to zero but the QUATERNION MATH fully exercised.
        for k, v in (("RE3SIM_VDR", "1"), ("RE3SIM_VDR_CAM", "1"),
                     ("RE3SIM_VDR_CAM_POS", "0"), ("RE3SIM_VDR_CAM_AIM", "0"),
                     ("RE3SIM_VDR_CAM_ROLL", "0"), ("RE3SIM_VDR_BUMP_P", "0")):
            os.environ[k] = v
        mdp.randomize_station_camera(e, None, STATION_CAM_EYE, STATION_CAM_TARGET)
        pos_dr, quat_dr = _cam_poses(cam)

        dp = (pos_dr - pos_ref).abs().max().item()
        # quaternion double cover: q and -q are the same rotation
        dq = torch.minimum((quat_dr - quat_ref).abs().max(dim=-1).values,
                           (quat_dr + quat_ref).abs().max(dim=-1).values).max().item()
        print(f"[probe] zero-DR vs stock aim: max |dpos| = {dp:.2e} m, "
              f"max |dquat| = {dq:.2e}", flush=True)
        if dp > 1e-5 or dq > 1e-4:
            raise SystemExit("[probe] FAIL: the DR camera path does not reproduce the stock "
                             "aim at zero DR — quaternion layout or convention is wrong. Do "
                             "NOT collect with it.")
        print("[probe] PASS", flush=True)
        for k in ("RE3SIM_VDR_CAM_POS", "RE3SIM_VDR_CAM_AIM",
                  "RE3SIM_VDR_CAM_ROLL", "RE3SIM_VDR_BUMP_P"):
            del os.environ[k]

    # ---- 2. stills grid ----------------------------------------------------------------
    os.makedirs(args_cli.out, exist_ok=True)
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
    except ImportError:
        import imageio  # noqa: PLC0415

    zero = torch.zeros(n, 7, device=e.device)
    zero[:, 6] = 1.0  # gripper open
    for r in range(args_cli.resets):
        env.reset()
        for _ in range(args_cli.settle):  # ⭐ STEP, don't just render — HANDOFF §4.9
            env.step(zero)
        e.sim.render()
        rows = []
        for name in ("station_cam", "wrist_cam"):
            s = e.scene[name]
            s.update(dt=0.0)
            img = s.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
            rows.append(np.concatenate(list(img), axis=1))  # envs side by side
        grid = np.concatenate(rows, axis=0)                 # station over wrist
        path = os.path.join(args_cli.out, f"vdr_reset{r:02d}.png")
        imageio.imwrite(path, grid)
        print(f"[grid] wrote {path} ({grid.shape[1]}x{grid.shape[0]})", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
