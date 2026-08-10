#!/usr/bin/env python
"""Render the reconstructed workstation so the gaussians can actually be looked at.

The splats are visual-only: nothing in the MDP reads them, so every automated check in this
project passes whether they are correctly placed, wildly offset, or not rendering at all. The
only way to know is to render a frame and look at it.

What to check in the output:

* **the desk surface meets the arm's base plate.** The physics desk is an analytic slab whose
  top face is at z = 0 and the arm stands on it, so if the gaussian desk sits above or below
  that line the marker-frame offset is wrong (02_RECONSTRUCTION.md 2.5).
* **the objects rest ON the gaussian desk**, not floating above or sunk into it.
* **the box sits flat** on the same surface.
* **no ghost objects** -- the scene capture was taken with the manipulated objects absent, so
  anything cube-shaped in the splats would be a double.

.. code-block:: bash

    python -u scripts/render_workstation.py --enable_cameras --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the reconstructed workstation.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--out", type=str, default="docs/envs/re3sim/renders")
parser.add_argument("--settle", type=int, default=60, help="physics steps before rendering")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--num_envs", type=int, default=1,
                    help="Clone this many envs. >1 exercises the multi-env splat alignment "
                         "that a single-env renderer structurally cannot test -- see the "
                         "`/World/Splats` note in `main`.")
parser.add_argument("--env", type=int, default=0, help="which env to point the camera at")
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

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402

# The workstation camera's pose is the ENV's, not this script's -- `collect_demos.py` mounts a
# camera at the same place, and a view that is inspected here and filmed there has to be one
# view. See `STATION_CAM_EYE` for why it is in front of the arm.
from reBot_RL.tasks.manager_based.re3sim.workstation_env_cfg import (  # noqa: E402
    STATION_CAM_EYE as FRONT_EYE,
    STATION_CAM_TARGET as FRONT_TARGET,
)

#: Viewpoints, chosen to answer different questions rather than to look nice.
#:   front         -- ⭐ the workstation camera; what a policy's third-person view sees
#:   over_shoulder -- the whole workstation, for "does it look like the photo"
#:   grazing       -- almost edge-on to the desk, which is where a z offset between the
#:                    gaussian desk and the physics desk shows up as a visible step
#:   top_down      -- object layout against the real desk
#:   close_cube    -- the grasp region at working distance
VIEWS = {
    "front":         (FRONT_EYE, FRONT_TARGET),
    "over_shoulder": ((-0.75, -0.65, 0.62), (0.30, 0.0, 0.03)),
    "grazing":       ((-0.55, 0.0, 0.045), (0.45, 0.0, 0.035)),
    "top_down":      ((0.30, 0.0, 1.05), (0.30, 0.0, 0.0)),
    "close_cube":    ((-0.10, -0.35, 0.30), (0.24, 0.0, 0.03)),
}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # The camera goes into the SCENE CONFIG, not built standalone. `InteractiveScene` is what
    # registers a sensor's initialisation callback on play; a `Camera` constructed afterwards
    # is never initialised and dies on its first read with a bare
    # `AttributeError: 'Camera' object has no attribute '_device'`.
    env_cfg.scene.render_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCam",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=1.0, horizontal_aperture=24.0,
            clipping_range=(0.01, 20.0)),
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    cam = e.scene["render_cam"]
    # Let contacts settle before rendering, or the objects are caught mid-drop and a reader
    # cannot tell a settling artefact from a misaligned desk.
    for _ in range(args_cli.settle):
        env.step(torch.zeros((e.num_envs, 7), device=e.device))

    out = os.path.abspath(args_cli.out)
    os.makedirs(out, exist_ok=True)
    origin = e.scene.env_origins[args_cli.env].cpu().numpy()

    # ⭐ The gaussian desk is spawned ONCE at `/World/Splats`, on the assumption that the env
    # being looked at stands at the world origin. That holds only at `num_envs == 1` -- Isaac
    # Lab's cloner CENTRES the env grid on the origin, so at 8 envs env 0 sits at
    # [2.5, -2.5, 0] and the desk is 3.5 m away, leaving the arm and objects floating on the
    # bare ground plane.
    #
    # This script used to hardcode `num_envs=1`, which is exactly why the bug survived: it was
    # the only tool that ever rendered anything, and it could not reach the configuration in
    # which the bug exists. A tool that only ever runs one configuration cannot find
    # configuration bugs. Hence `--num_envs`.
    if os.environ.get("RE3SIM_SPLATS_PER_ENV") == "1":
        # Cloned per env by the scene cfg -- every env has its own desk, so there is nothing
        # to move and moving `/World/Splats` would find no prim.
        print("[render] splats are PER-ENV; no re-placement needed", flush=True)
    elif float(np.abs(origin).max()) > 1e-6:
        import omni.usd  # noqa: PLC0415
        from pxr import Gf, UsdGeom  # noqa: PLC0415
        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Splats")
        if prim and prim.IsValid():
            # Re-use an existing translate op: `AddTranslateOp` on a prim that already has a
            # transform stack APPENDS a second op and the two then compose.
            xf = UsdGeom.Xformable(prim)
            op = next((o for o in xf.GetOrderedXformOps()
                       if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            (op or xf.AddTranslateOp()).Set(Gf.Vec3d(*[float(v) for v in origin]))
            print(f"[render] moved /World/Splats onto env {args_cli.env} at "
                  f"{np.round(origin, 3)}", flush=True)
        else:
            print("[render] WARNING: no /World/Splats -- rendering without the desk")
    origins = e.scene.env_origins.cpu().numpy()
    for name, (eye, target) in VIEWS.items():
        # One pose PER CAMERA. `set_world_poses_from_view(env_ids=None)` builds
        # `arange(num_envs)` for the indices but does NOT broadcast a single row to match, so
        # handing it one pose at num_envs > 1 leaves every camera but one at its spawn pose --
        # and the render then shows the bare ground plane. Another bug that could not exist
        # while this script was hardcoded to a single env.
        cam.set_world_poses_from_view(
            torch.tensor(origins + np.asarray(eye), dtype=torch.float32, device=e.device),
            torch.tensor(origins + np.asarray(target), dtype=torch.float32, device=e.device),
        )
        # Several renders: the RTX path is progressive, and the first frame after a camera
        # move is noisy or occasionally still shows the previous pose.
        for _ in range(12):
            e.sim.render()
            cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][args_cli.env, ..., :3].cpu().numpy().astype(np.uint8)
        import imageio.v2 as imageio
        path = os.path.join(out, f"{name}.png")
        imageio.imwrite(path, rgb)
        print(f"wrote {path}   mean pixel {rgb.mean():.1f}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
