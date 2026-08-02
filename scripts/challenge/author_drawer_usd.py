# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the cabinet-with-a-drawer asset for ``Rebot-DrawerOrder-v0``.

Plain ``pxr`` -- no Isaac Sim needed, same approach as ``scripts/author_basket_usd.py``.

The asset is a one-DOF articulation: a fixed cabinet shell plus a drawer on a prismatic
joint sliding along the cabinet's local -x (toward the robot). The drawer carries a handle
bar sized and positioned to be **side-graspable**, because this arm has no top-down grasp
below z = 0.19 m (docs/CHALLENGE_SUITE.md C1) and could not reach over a lip.

.. code-block:: bash

    python scripts/challenge/author_drawer_usd.py
"""

import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OUT = os.path.join(_REPO, "source/reBot_RL/data/drawer/drawer.usda")

# --- geometry (metres). Kept in one place; drawer_env_cfg.py imports these. ---------
BODY_W = 0.130      # y, outer width
BODY_D = 0.090      # x, outer depth
BODY_H = 0.085      # z, outer height
WALL_T = 0.006
#: drawer travel along -x. Must exceed the block's depth so the block can go in.
TRAVEL = 0.070
#: interior cavity of the drawer
CAV_W = BODY_W - 2 * WALL_T
CAV_D = BODY_D - 2 * WALL_T
CAV_H = 0.040
#: handle bar: 20 mm across the fingers, inside the measured 26-42 mm sweet spot when
#: the fingers straddle it, and standing clear of the drawer front so they can get around
HANDLE_T = 0.020
HANDLE_STANDOFF = 0.022


#: cache of UsdPreviewSurface materials, one per colour
_MATS: dict = {}


def _material(stage, color):
    """A real UsdPreviewSurface for ``color``.

    ``displayColor`` alone is NOT enough: the RTX renderer ignores it, so the whole cabinet
    came out flat white and neither the drawer front nor the handle was legible in the
    recorded videos. A bound material is what actually shows up.
    """
    key = tuple(round(c, 4) for c in color)
    if key in _MATS:
        return _MATS[key]
    name = "Mat_%02x%02x%02x" % tuple(int(255 * c) for c in color)
    mat = UsdShade.Material.Define(stage, f"/Cabinet/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/Cabinet/Looks/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _MATS[key] = mat
    return mat


def _cube(stage, path, size, offset, color, collision=True):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)  # unit cube; Cube.size is a scalar, so shape comes from scale
    c.AddTranslateOp().Set(Gf.Vec3d(*offset))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(_material(stage, color))
    if collision:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c


def main():
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    stage = Usd.Stage.CreateNew(_OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Cabinet")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # ---- fixed shell -------------------------------------------------------
    # NOTE: the shell must be an ordinary rigid body anchored by a FixedJoint, NOT a
    # kinematic body. PhysX articulations cannot contain kinematic links -- doing so makes
    # the whole articulation fail to parse ("did not match any articulations").
    shell = UsdGeom.Xform.Define(stage, "/Cabinet/Shell")
    UsdPhysics.RigidBodyAPI.Apply(shell.GetPrim())
    UsdPhysics.MassAPI.Apply(shell.GetPrim()).CreateMassAttr(2.0)
    grey = (0.26, 0.28, 0.33)
    # open at -x (facing the robot); floor, roof, back and two sides
    _cube(stage, "/Cabinet/Shell/Floor", (BODY_D, BODY_W, WALL_T), (0, 0, WALL_T / 2), grey)
    _cube(stage, "/Cabinet/Shell/Roof", (BODY_D, BODY_W, WALL_T), (0, 0, BODY_H - WALL_T / 2), grey)
    _cube(stage, "/Cabinet/Shell/Back", (WALL_T, BODY_W, BODY_H),
          (BODY_D / 2 - WALL_T / 2, 0, BODY_H / 2), grey)
    _cube(stage, "/Cabinet/Shell/SideL", (BODY_D, WALL_T, BODY_H),
          (0, BODY_W / 2 - WALL_T / 2, BODY_H / 2), grey)
    _cube(stage, "/Cabinet/Shell/SideR", (BODY_D, WALL_T, BODY_H),
          (0, -BODY_W / 2 + WALL_T / 2, BODY_H / 2), grey)

    # ---- drawer ------------------------------------------------------------
    drawer = UsdGeom.Xform.Define(stage, "/Cabinet/Drawer")
    UsdPhysics.RigidBodyAPI.Apply(drawer.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(drawer.GetPrim())
    # light: this arm's payload is limited and the drawer is dragged, not carried
    mass.CreateMassAttr(0.12)
    tan = (0.78, 0.62, 0.30)
    dz = WALL_T + CAV_H / 2  # drawer sits on the shell floor
    _cube(stage, "/Cabinet/Drawer/Base", (CAV_D, CAV_W, WALL_T), (0, 0, WALL_T + WALL_T / 2), tan)
    _cube(stage, "/Cabinet/Drawer/SideL", (CAV_D, WALL_T, CAV_H), (0, CAV_W / 2 - WALL_T / 2, dz), tan)
    _cube(stage, "/Cabinet/Drawer/SideR", (CAV_D, WALL_T, CAV_H), (0, -CAV_W / 2 + WALL_T / 2, dz), tan)
    _cube(stage, "/Cabinet/Drawer/Back", (WALL_T, CAV_W, CAV_H), (CAV_D / 2 - WALL_T / 2, 0, dz), tan)
    _cube(stage, "/Cabinet/Drawer/Front", (WALL_T, CAV_W, CAV_H), (-CAV_D / 2 + WALL_T / 2, 0, dz), tan)
    # handle bar, standing off the front face so the fingers can straddle it side-on
    _cube(stage, "/Cabinet/Drawer/Handle", (HANDLE_T, 0.045, HANDLE_T),
          (-CAV_D / 2 - HANDLE_STANDOFF, 0, dz), (0.90, 0.20, 0.10))

    # ---- anchor the shell to the world (fixed-base articulation) ------------
    root_joint = UsdPhysics.FixedJoint.Define(stage, "/Cabinet/RootJoint")
    root_joint.CreateBody1Rel().SetTargets(["/Cabinet/Shell"])  # body0 empty == world

    # ---- prismatic joint: drawer slides along -x ----------------------------
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Cabinet/DrawerJoint")
    joint.CreateBody0Rel().SetTargets(["/Cabinet/Shell"])
    joint.CreateBody1Rel().SetTargets(["/Cabinet/Drawer"])
    joint.CreateAxisAttr("X")
    # negative = pulled out toward the robot
    joint.CreateLowerLimitAttr(-TRAVEL)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    # A drive with zero stiffness and a little damping makes the drawer behave like a real
    # one: it stays where it is put and resists being flung, but offers no spring force
    # pulling it back. Without any drive the joint is frictionless and the drawer coasts.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(8.0)
    drive.CreateMaxForceAttr(40.0)

    stage.Save()
    print(f"[author_drawer_usd] wrote {_OUT}")
    print(f"  shell   {BODY_D * 1000:.0f} x {BODY_W * 1000:.0f} x {BODY_H * 1000:.0f} mm, opening at -x")
    print(f"  cavity  {CAV_D * 1000:.0f} x {CAV_W * 1000:.0f} x {CAV_H * 1000:.0f} mm")
    print(f"  travel  {TRAVEL * 1000:.0f} mm along -x")
    print(f"  handle  {HANDLE_T * 1000:.0f} mm bar, {HANDLE_STANDOFF * 1000:.0f} mm standoff")


if __name__ == "__main__":
    main()
