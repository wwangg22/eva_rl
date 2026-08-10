#!/usr/bin/env python
"""Render the workstation camera at a fixed mount and a sweep of downward tilts.

Mount (Big Will, 2026-08-09): 66 cm in front of the robot base, 100 cm above the table top.
The desk's top face is z = 0 and the arm stands on it, so "above the table" is just z.

Tilt is measured DOWN FROM HORIZONTAL, so 0 deg looks straight at the arm and 90 deg looks
straight down at the desk. The optical axis hits the table at

    x_hit = EYE_X - EYE_Z / tan(tilt)

which is what actually decides whether the cube (x ~ 0.22) and the box are in frame -- at
75 deg the axis lands at x = 0.39, i.e. BEHIND the cube, so the workspace sits high in the
image. That is the thing to look at in these renders.

.. code-block:: bash

    python -u scripts/station_cam_tilt_sweep.py --enable_cameras --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep the workstation camera's downward tilt.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--out", type=str, default="docs/envs/re3sim/renders/station_tilt")
parser.add_argument("--settle", type=int, default=60)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--dist", type=float, default=0.66, help="metres in front of the base")
parser.add_argument("--height-above-table", type=float, default=1.00)
parser.add_argument("--tilts", type=str, default="45,55,60,65,70,75,80,85,90",
                    help="degrees below horizontal, comma separated")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # Same optics the demo recorder mounts for `station_cam`, so what is picked here is what a
    # policy would actually be trained on rather than a prettier preview lens.
    env_cfg.scene.render_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCam", update_period=0.0,
        width=args_cli.width, height=args_cli.height, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=17.0, horizontal_aperture=20.955,
                                         clipping_range=(0.02, 20.0)))
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    cam = e.scene["render_cam"]
    for _ in range(args_cli.settle):
        env.step(torch.zeros((e.num_envs, 7), device=e.device))

    out = os.path.abspath(args_cli.out)
    os.makedirs(out, exist_ok=True)
    origins = e.scene.env_origins.cpu().numpy()

    eye = np.array([args_cli.dist, 0.0, args_cli.height_above_table])
    for deg in [float(t) for t in args_cli.tilts.split(",")]:
        th = math.radians(deg)
        # Look back toward the arm (-x) and down by `th`; 1 m along the axis is an arbitrary
        # target distance -- only the direction matters to `set_world_poses_from_view`.
        target = eye + np.array([-math.cos(th), 0.0, -math.sin(th)])
        # One pose PER CAMERA -- `set_world_poses_from_view` does not broadcast (see
        # render_workstation.py). num_envs is 1 here, but keep the shape honest.
        cam.set_world_poses_from_view(
            torch.tensor(origins + eye, dtype=torch.float32, device=e.device),
            torch.tensor(origins + target, dtype=torch.float32, device=e.device),
        )
        for _ in range(12):
            e.sim.render()
            cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        import imageio.v2 as imageio
        path = os.path.join(out, f"tilt_{int(round(deg)):02d}.png")
        imageio.imwrite(path, rgb)
        hit = eye[0] - eye[2] / math.tan(th) if deg < 89.99 else eye[0]
        print(f"wrote {path}   tilt {deg:.0f} deg   axis hits table at x = {hit:+.3f} m"
              f"   mean pixel {rgb.mean():.1f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
