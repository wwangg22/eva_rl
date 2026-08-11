#!/usr/bin/env python
"""Regenerate the RS-rebot-dev-arm URDF from the shipped USD, for cuRobo.

``expert/rs_rebot.yml`` (the cuRobo robot config that the pick-and-place expert uses) points
its ``urdf_path`` at ``/home/william/.../00-arm-rs_asm-v3.urdf`` -- a path from another
machine. Only the USD is in this repo, and cuRobo builds its kinematic chain from URDF, so
without this the planner cannot be constructed at all.

The conversion keeps the LINK FRAMES identical to the USD's body frames, which is not
cosmetic: ``rs_rebot.yml`` carries per-link collision spheres expressed in those frames, and a
URDF with its own frame convention would put every sphere in the wrong place while still
loading cleanly.

Frame algebra, per USD physics joint:

    T0 = (localPos0, localRot0)   the joint frame in the PARENT body's frame
    T1 = (localPos1, localRot1)   the joint frame in the CHILD body's frame

    URDF <origin>  parent -> child at zero angle  =  T0 @ inv(T1)
    URDF <axis>    the USD axis (Z) is given in the JOINT frame, so in the child frame
                   it is R(localRot1) @ [0,0,1]

Meshes are deliberately NOT emitted: cuRobo collides via the sphere model in the yml, so the
URDF only has to supply the chain. Every link gets a tiny massless placeholder so URDF parsers
accept it.

⭐ Verify before trusting: ``--check`` compares this chain's forward kinematics against the
simulator's over random joint draws. A silently offset chain produces grasps that miss by a
constant, which looks like a controller problem for a long time.

.. code-block:: bash

    python scripts/usd_to_urdf_rebot.py --check
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_USD = os.path.join(_HERE, "..", "source", "reBot_RL", "data", "RS-rebot-dev-arm",
                    "00-arm-rs_asm-v3.usda")
_OUT = os.path.join(_HERE, "..", "source", "reBot_RL", "data", "RS-rebot-dev-arm",
                    "00-arm-rs_asm-v3.urdf")

#: (name, parent, child, kind). Order is the chain order; the fixed root joint is skipped
#: because URDF's root is `base_link` itself.
CHAIN = [
    ("joint1", "base_link", "link1", "revolute"),
    ("joint2", "link1", "link2", "revolute"),
    ("joint3", "link2", "link3", "revolute"),
    ("joint4", "link3", "link4", "revolute"),
    ("joint5", "link4", "link5", "revolute"),
    ("joint6", "link5", "link6", "revolute"),
    ("j_gripper_end", "link6", "gripper_end", "fixed"),
    ("joint_left", "gripper_end", "gripper_left", "prismatic"),
    ("joint_right", "gripper_end", "gripper_right", "prismatic"),
]


def _quat_to_R(q) -> np.ndarray:
    """USD quaternions are (w, x, y, z). `Gf.Quatf` is not iterable, hence the accessors."""
    if hasattr(q, "GetReal"):
        w = float(q.GetReal())
        im = q.GetImaginary()
        x, y, z = float(im[0]), float(im[1]), float(im[2])
    else:
        w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _T(p, q) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_to_R(q)
    m[:3, 3] = [float(v) for v in p]
    return m


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF uses fixed-axis roll-pitch-yaw (XYZ)."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:                      # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def build(usd_path: str) -> str:
    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path)
    prims = {p.GetName(): p for p in stage.Traverse() if "Joint" in str(p.GetTypeName())}

    links = ["base_link"] + [c for _, _, c, _ in CHAIN]
    out = ['<?xml version="1.0"?>', '<robot name="rs_rebot">']
    for ln in links:
        out += [f'  <link name="{ln}">',
                '    <inertial><mass value="0.01"/>'
                '<inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>',
                '  </link>']

    for name, parent, child, kind in CHAIN:
        p = prims[name]
        g = lambda a: p.GetAttribute(a).Get()  # noqa: E731
        T0 = _T(g("physics:localPos0"), g("physics:localRot0"))
        T1 = _T(g("physics:localPos1"), g("physics:localRot1"))
        M = T0 @ np.linalg.inv(T1)
        xyz = M[:3, 3]
        r, pi, y = _rpy(M[:3, :3])
        # the USD axis is Z in the JOINT frame; express it in the CHILD frame
        axis = _quat_to_R(g("physics:localRot1")) @ np.array([0.0, 0.0, 1.0])
        out.append(f'  <joint name="{name}" type="{kind}">')
        out.append(f'    <parent link="{parent}"/>')
        out.append(f'    <child link="{child}"/>')
        out.append(f'    <origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" '
                   f'rpy="{r:.9f} {pi:.9f} {y:.9f}"/>')
        if kind != "fixed":
            out.append(f'    <axis xyz="{axis[0]:.9f} {axis[1]:.9f} {axis[2]:.9f}"/>')
            lo, hi = g("physics:lowerLimit"), g("physics:upperLimit")
            if kind == "revolute":
                lo, hi = math.radians(float(lo)), math.radians(float(hi))
            else:
                lo, hi = float(lo), float(hi)
            out.append(f'    <limit lower="{lo:.9f}" upper="{hi:.9f}" '
                       f'effort="100.0" velocity="3.14"/>')
        out.append('  </joint>')
    out.append('</robot>')
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd", default=_USD)
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--check", action="store_true",
                    help="compare this chain's FK against the simulator's")
    args = ap.parse_args()

    urdf = build(os.path.abspath(args.usd))
    with open(os.path.abspath(args.out), "w") as f:
        f.write(urdf)
    print(f"wrote {os.path.abspath(args.out)}  ({len(urdf.splitlines())} lines)")
    if args.check:
        print("now run scripts/check_urdf_fk.py to compare against the simulator")


if __name__ == "__main__":
    main()
