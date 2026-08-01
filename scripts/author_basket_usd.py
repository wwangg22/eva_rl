# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-off authoring of the movable basket USD for the v1 pick-and-place env.

Writes ``source/reBot_RL/data/basket/basket.usda``: a single kinematic rigid body
(``RigidBodyAPI`` with ``kinematicEnabled``) whose collision is the same 5 cuboids
(floor + 4 walls) the v0 scene builds as static ``AssetBaseCfg`` parts. Dimensions are
imported from ``mdp/common.py`` so the geometry always matches the "placed" predicate.

Being one rigid body (instead of 5 static prims baked at config time) lets the v1 reset
event move the whole basket per env per episode with ``write_root_pose_to_sim``.

Runs with the plain ``pxr`` USD python API (no Isaac Sim needed):

.. code-block:: bash

    python scripts/author_basket_usd.py

"""

import importlib.util
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMMON = os.path.join(_REPO, "source/reBot_RL/reBot_RL/tasks/manager_based/pick_place/mdp/common.py")
_OUT = os.path.join(_REPO, "source/reBot_RL/data/basket/basket.usda")

# load mdp/common.py directly from file (importing the package would pull in isaaclab)
spec = importlib.util.spec_from_file_location("pick_place_common", _COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

_B_IN = common.BASKET_INNER_HALF
_B_T = common.BASKET_WALL_THICKNESS
_B_H = common.BASKET_WALL_HEIGHT
_B_F = common.BASKET_FLOOR_THICKNESS
_B_OUT = 2 * (_B_IN + _B_T)  # outer footprint
_B_WALL_C = _B_IN + _B_T / 2  # wall centerline offset

# (name, size xyz, offset xyz) -- identical to the v0 _basket_part layout
PARTS = [
    ("Floor", (_B_OUT, _B_OUT, _B_F), (0.0, 0.0, _B_F / 2)),
    ("WallPX", (_B_T, _B_OUT, _B_H), (_B_WALL_C, 0.0, _B_H / 2)),
    ("WallNX", (_B_T, _B_OUT, _B_H), (-_B_WALL_C, 0.0, _B_H / 2)),
    ("WallPY", (_B_OUT, _B_T, _B_H), (0.0, _B_WALL_C, _B_H / 2)),
    ("WallNY", (_B_OUT, _B_T, _B_H), (0.0, -_B_WALL_C, _B_H / 2)),
]


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    stage = Usd.Stage.CreateNew(_OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    basket = UsdGeom.Xform.Define(stage, "/Basket")
    stage.SetDefaultPrim(basket.GetPrim())
    # one kinematic rigid body; the child cubes below form its compound collider
    rb = UsdPhysics.RigidBodyAPI.Apply(basket.GetPrim())
    rb.CreateKinematicEnabledAttr(True)

    for name, size, offset in PARTS:
        cube = UsdGeom.Cube.Define(stage, f"/Basket/{name}")
        # unit cube scaled to the part dimensions (Cube.size is a single scalar)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*offset))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.30, 0.15)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    stage.Save()
    print(f"[author_basket_usd] wrote {_OUT}")
    for name, size, offset in PARTS:
        print(f"  {name}: size={tuple(round(v, 4) for v in size)} offset={tuple(round(v, 4) for v in offset)}")


if __name__ == "__main__":
    main()
