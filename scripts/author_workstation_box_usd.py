# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the open cardboard box for the reconstructed-workstation task.

Writes ``source/reBot_RL/data/workstation/box.usda``: a single **kinematic** rigid body whose
collider is five thin cuboids (floor + four walls) at the box's measured dimensions.

Why this is authored rather than reconstructed (blocker B2, ``docs/envs/re3sim``):
``Re3Sim/workstation/tools/extract_object.py`` adds a synthetic flat base and Poisson-closes
the mesh. That is right for a cube and for anything resting on a table, and wrong for a
receptacle -- a closed 218 x 150 x 93 mm block has no interior, so nothing can be placed
*into* it. The reconstruction supplies the box's **visual**; this supplies its physics.

Dimensions are imported from ``re3sim/mdp/common.py`` so the geometry and the "placed"
predicate can never silently disagree -- the failure mode the v0/v1 basket has, where the
same numbers live in both an env cfg and a separately generated USD.

Runs with the plain ``pxr`` USD API (no Isaac Sim needed):

.. code-block:: bash

    python scripts/author_workstation_box_usd.py
"""

import importlib.util
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMMON = os.path.join(
    _REPO, "source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/mdp/common.py")
_OUT = os.path.join(_REPO, "source/reBot_RL/data/workstation/box.usda")

# load mdp/common.py straight off disk (importing the package would pull in isaaclab)
spec = importlib.util.spec_from_file_location("re3sim_common", _COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

OX, OY = common.BOX_OUTER_X, common.BOX_OUTER_Y
H, T, FT = common.BOX_HEIGHT, common.BOX_WALL, common.BOX_FLOOR_THICKNESS

# (name, size xyz, offset xyz). Walls are placed so their INNER faces sit at +/-(O/2 - T),
# which is exactly BOX_INNER_*/2 -- the number the placed predicate tests against.
PARTS = [
    ("Floor", (OX, OY, FT), (0.0, 0.0, FT / 2)),
    ("WallPX", (T, OY, H), ((OX - T) / 2, 0.0, H / 2)),
    ("WallNX", (T, OY, H), (-(OX - T) / 2, 0.0, H / 2)),
    ("WallPY", (OX, T, H), (0.0, (OY - T) / 2, H / 2)),
    ("WallNY", (OX, T, H), (0.0, -(OY - T) / 2, H / 2)),
]


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    stage = Usd.Stage.CreateNew(_OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    box = UsdGeom.Xform.Define(stage, "/Box")
    stage.SetDefaultPrim(box.GetPrim())
    # One kinematic rigid body; the child cubes below form its compound collider. Kinematic
    # because the box must not move: at 95 g a dynamic box is shoved on first contact, and
    # making it artificially heavy only slows that down instead of preventing it.
    rb = UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    rb.CreateKinematicEnabledAttr(True)

    for name, size, offset in PARTS:
        cube = UsdGeom.Cube.Define(stage, f"/Box/{name}")
        cube.CreateSizeAttr(1.0)  # unit cube; Cube.size is a single scalar, so scale it
        cube.AddTranslateOp().Set(Gf.Vec3d(*offset))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.62, 0.47, 0.31)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    stage.Save()
    print(f"[author_workstation_box_usd] wrote {_OUT}")
    print(f"  outer      : {OX * 1000:.0f} x {OY * 1000:.0f} x {H * 1000:.0f} mm")
    print(f"  inner      : {common.BOX_INNER_X * 1000:.0f} x {common.BOX_INNER_Y * 1000:.0f} mm")
    print(f"  wall/floor : {T * 1000:.0f} / {FT * 1000:.0f} mm")
    print(f"  a cube resting on the interior floor has its root at "
          f"{(FT + common.CUBE_SIZE / 2) * 1000:.0f} mm; the rim is at {H * 1000:.0f} mm")
    print(f"  ** BOX_OUTER_Y = {OY * 1000:.0f} mm is an ASSUMPTION -- not on the capture sheet.")
    for name, size, offset in PARTS:
        print(f"  {name}: size={tuple(round(v, 4) for v in size)} "
              f"offset={tuple(round(v, 4) for v in offset)}")


if __name__ == "__main__":
    main()
