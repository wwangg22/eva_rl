# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author a *proper* Rubik's cube for the reconstructed-workstation task.

Replaces the photogrammetry-reconstructed ``cube.usda``. The reconstruction was wrong in ways
no gate caught (``docs/envs/re3sim/HANDOFF.md`` §3.2): multi-view stereo cannot match a flat
glossy sticker -- the specular highlight moves between frames, so the same physical point
looks different in every one -- while the high-contrast black grooves match perfectly. The
mesh that came out was a cube-shaped **lattice** with the centre of every sticker punched out,
and it passed the extents gate at +4.8 % because *a hollow shell has the same bounding box as
the solid it should be*. This object is 54 flat coloured squares on a black box. There is
nothing to reconstruct: authoring it is both cheaper and strictly more accurate.

What this buys, beyond the shape being right:

* the collider is an **analytic box** of exactly ``CUBE_SIZE``, not a convex hull of a noisy
  mesh, so contact is exact and cheap;
* the colours are **authored constants on real materials**, sidestepping the open
  washed-out-vertex-colour bug (HANDOFF §5.1) that only affects the
  ``UsdPrimvarReader_float3 -> displayColor`` path;
* and the **sticker pattern is a parameter, not a baked-in property of the asset**.

Pattern robustness
------------------
A policy that has only ever seen a solved cube has seen one image of this object. ``--pattern``
takes ``solved`` or ``scrambled:<seed>``, and ``--variants N`` writes a numbered family whose
patterns differ. Every scramble is a **physically valid cube state**, because it is produced by
applying real face turns to a real cube rather than by shuffling colours: the model here is 27
cubies carrying an integer rotation matrix, and a face turn rotates the position *and* the
orientation of every cubie in that layer. A colour shuffle would happily put white next to
yellow on one cubie, which no real cube can do.

Geometry, in the body frame (origin at the cube's centre -- ``REST_Z`` and ``placed_mask``
are calibrated for centre-origin bodies, see ``mdp/common.py``):

* a black ``UsdGeom.Cube`` of side ``CUBE_SIZE``, which is also the collider;
* 54 quads, one per sticker, standing ``PROUD`` mm off the faces so they cannot z-fight,
  grouped into six ``GeomSubset``\\ s -- one per colour -- each bound to its own
  ``UsdPreviewSurface``.

No Isaac Sim needed; this is plain ``pxr``:

.. code-block:: bash

    python scripts/author_rubiks_cube_usd.py --variants 8
"""

import argparse
import importlib.util
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMMON = os.path.join(
    _REPO, "source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/mdp/common.py")
_OUT_DIR = os.path.join(_REPO, "source/reBot_RL/data/workstation/objects")

_RUBIKS = os.path.join(
    _REPO, "source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/rubiks.py")


def _load(name, path):
    """Load a module straight off disk -- importing the package would pull in isaaclab."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


common = _load("re3sim_common", _COMMON)
#: ⭐ The cube's STATE lives in the env package, not here, because
#: ``mdp/events.py:randomize_cube_pattern`` re-assigns the facelets of every cloned env at
#: startup and has to agree with this script about which quad is which facelet. Two copies of
#: that ordering would not fail -- they would render nonsense quietly.
rubiks = _load("re3sim_rubiks", _RUBIKS)
COLORS, BODY_COLOR = rubiks.COLORS, rubiks.BODY_COLOR

#: sticker edge as a fraction of the 1/3-face cell. A real 56 mm cube has ~16 mm stickers in
#: an 18.7 mm cell, i.e. 0.86.
STICKER_FRAC = 0.86
#: how far a sticker stands off the body face [m]. Large enough to beat depth precision at
#: this scale, small enough to be inside the caliper tolerance of the real object.
PROUD = 0.0002


def stickers(cube, size: float):
    """Yield ``(colour_name, 4 corner points)`` for all 54 stickers, wound CCW from outside.

    Walks :func:`rubiks.facelets` so the N-th quad emitted here is the N-th facelet there --
    the contract the runtime randomiser depends on.
    """
    cell = size / 3.0
    half = cell * STICKER_FRAC / 2.0
    for normal, iu, iv, grid in rubiks.facelets():
        n = np.asarray(normal)
        u, v = rubiks.face_axes(normal)
        c = n * (size / 2 + PROUD) + u * (iu * cell) + v * (iv * cell)
        yield cube.color_at(grid, normal), [
            c - u * half - v * half,
            c + u * half - v * half,
            c + u * half + v * half,
            c - u * half + v * half,
        ]


def write(path: str, cube, size: float, mass: float, name: str = "RubiksCube"):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)

    looks = UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    mats = {}
    for cname, rgb in list(COLORS.items()) + [("body", BODY_COLOR)]:
        mat = UsdShade.Material.Define(stage, f"{looks.GetPath()}/{cname}")
        sh = UsdShade.Shader.Define(stage, f"{looks.GetPath()}/{cname}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        # Stickers are vinyl and the body is moulded ABS: both are matte-ish with a low
        # specular, and a mirror-bright cube reads as plastic toy rather than as the object
        # in the capture.
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mats[cname] = mat

    # ---- the body: visual AND collider. An analytic box, so PhysX gets an exact box shape.
    # Authored at its true size with NO scale op: C8 measured that a post-build
    # ``xformOp:scale`` never reaches the PhysX collider (a block scaled z x 0.5 still rests
    # at its unscaled half-height), and the same trap is one keystroke away here.
    body = UsdGeom.Cube.Define(stage, f"/{name}/Body")
    body.CreateSizeAttr(float(size))
    h = float(size / 2)
    body.CreateExtentAttr([Gf.Vec3f(-h, -h, -h), Gf.Vec3f(h, h, h)])
    body.CreateDisplayColorAttr([Gf.Vec3f(*BODY_COLOR)])
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(mats["body"])

    # ---- the stickers: one mesh, six GeomSubsets, no collider.
    # ``modify_collision_properties`` only touches prims that already carry CollisionAPI, so
    # leaving it off here is what keeps the stickers out of the physics scene.
    pts, counts, idx, normals, by_color = [], [], [], [], {}
    for cname, quad in stickers(cube, size):
        by_color.setdefault(cname, []).append(len(counts))
        n = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        normals.append(Gf.Vec3f(*(n / np.linalg.norm(n)).astype(float)))
        idx.extend(range(len(pts), len(pts) + 4))
        pts.extend(Gf.Vec3f(*p.astype(float)) for p in quad)
        counts.append(4)

    mesh = UsdGeom.Mesh.Define(stage, f"/{name}/Stickers")
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    e = float(size / 2 + PROUD)
    mesh.CreateExtentAttr([Gf.Vec3f(-e, -e, -e), Gf.Vec3f(e, e, e)])
    # Bug #4 from the reconstruction pipeline: leaving this unauthored makes USD fall back to
    # ``catmullClark``, which rounds the corners off every sticker AND silently voids the
    # authored normals.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    binder = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    for cname, faces in by_color.items():
        sub = UsdGeom.Subset.CreateGeomSubset(
            mesh, cname, UsdGeom.Tokens.face, Vt.IntArray([int(f) for f in faces]),
            "materialBind")
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mats[cname])
    binder.Bind(mats["body"])

    stage.Save()
    return {c: len(f) for c, f in by_color.items()}


def build(pattern: str, size: float, mass: float, out: str):
    cube = rubiks.pattern(None if pattern == "solved" else int(pattern.split(":", 1)[1]))
    tally = write(out, cube, size, mass)
    bad = {c: n for c, n in tally.items() if n != 9}
    # Nine of every colour is the invariant a face turn cannot break; if it ever does, the
    # cubie model is wrong and every pattern shipped after it is wrong too.
    assert not bad and len(tally) == 6, f"{out}: invalid colour tally {tally}"
    print(f"[author_rubiks_cube_usd] {os.path.basename(out)}  pattern={pattern}  "
          f"{size * 1000:.1f} mm  {mass * 1000:.0f} g  54 stickers, 9 of each colour")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=common.CUBE_SIZE)
    ap.add_argument("--mass", type=float, default=common.CUBE_MASS)
    ap.add_argument("--out-dir", default=_OUT_DIR)
    ap.add_argument("--pattern", default="scrambled:0",
                    help="'solved' or 'scrambled:<seed>' -- the pattern for cube.usda itself")
    ap.add_argument("--variants", type=int, default=8,
                    help="also write cube_p00..cube_pNN, each a different valid scramble")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build(args.pattern, args.size, args.mass, os.path.join(args.out_dir, "cube.usda"))
    for i in range(args.variants):
        # variant 0 is the solved cube on purpose: it is the one a human can check by eye.
        build("solved" if i == 0 else f"scrambled:{100 + i}", args.size, args.mass,
              os.path.join(args.out_dir, f"cube_p{i:02d}.usda"))


if __name__ == "__main__":
    main()
