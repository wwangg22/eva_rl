# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where can a *side-facing socket* go, and along which axis can it be entered?

A horizontal insertion needs more than a reachable TCP position: the TCP must reach the
socket mouth *and* the fully-inserted pose while holding a specific horizontal heading
(the socket axis). ``reachability_map.py`` records, per voxel, a bitmask of the 30-deg
horizontal headings it ever attained; this script reads that mask and reports candidate
socket sites.

.. code-block:: bash

    python scripts/analysis/query_socket_sites.py --depth 0.05
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Candidate side-facing socket sites.")
parser.add_argument("--npz", type=str, default="logs/analysis/reachability/reachability.npz")
parser.add_argument("--min_count", type=int, default=5)
parser.add_argument("--depth", type=float, default=0.05, help="insertion depth to clear [m]")
args = parser.parse_args()

d = np.load(args.npz)
c, cnt, mask = d["centers"], d["counts"], d["horiz_heading_mask"]
rel = cnt >= args.min_count
c, mask = c[rel], mask[rel]
r = np.linalg.norm(c[:, :2], axis=1)
z = c[:, 2]
cell = float(d["cell"])

print(f"reliable voxels: {len(c)}   heading bins: 12 x 30 deg\n")

# heading bin b covers [b*30 - 180, (b+1)*30 - 180) degrees
def bin_of(deg: float) -> int:
    return int(np.floor((((deg + 180.0) % 360.0)) / 30.0))


print("=" * 78)
print("Q. AT TABLE/SHELF HEIGHTS, HOW MANY HORIZONTAL HEADINGS ARE AVAILABLE?")
print("=" * 78)
for zlo, zhi in [(0.00, 0.06), (0.06, 0.12), (0.12, 0.20), (0.20, 0.30)]:
    m = (z >= zlo) & (z < zhi) & (r >= 0.15) & (r <= 0.40) & (np.abs(np.degrees(np.arctan2(c[:, 1], c[:, 0]))) <= 60)
    if m.sum() == 0:
        print(f"  z [{zlo:.2f},{zhi:.2f}): no voxels")
        continue
    nbits = np.array([bin(int(v)).count("1") for v in mask[m]])
    print(f"  z [{zlo:.2f},{zhi:.2f}): {m.sum():5d} voxels, headings available "
          f"median {np.median(nbits):.0f}/12, p90 {np.percentile(nbits, 90):.0f}/12, "
          f"none-available {100 * (nbits == 0).mean():.1f}%")

# ---------------------------------------------------------------------------
# A socket mounted at radius R facing the robot is entered along the -radial
# direction, i.e. heading pointing away from the base at azimuth(site).
# Check: is that heading attainable at BOTH the mouth and the inserted depth?
print("\n" + "=" * 78)
print(f"Q. CANDIDATE SOCKET SITES (radially-facing, {args.depth * 1000:.0f} mm insertion depth)")
print("=" * 78)
print("   This arm grasps side-on, so a held peg is gripped ACROSS its axis: the peg is")
print("   pushed in by translating the wrist, not by pointing the fingers down the bore.")
print("   The binding constraint is therefore that SOME gripper heading is attainable at")
print("   both the mouth and the fully-inserted pose, so the wrist can translate along the")
print("   socket axis without having to re-orient mid-insertion.")
print()
print(f"  {'r_mouth':>8} {'z':>6} {'az':>6} | {'#common':>8} | ok  common headings [deg]")
print("  " + "-" * 62)

# voxel index is floor(pos / cell) -- the same quantization reachability_map.py used
lut = {}
for i in range(len(c)):
    k = tuple(np.floor(c[i] / cell).astype(int))
    lut[k] = mask[i]


def mask_at(p: np.ndarray) -> int:
    return lut.get(tuple(np.floor(p / cell).astype(int)), 0)


STEPS = 5  # intermediate poses checked along the insertion stroke


def stroke_ok(site: np.ndarray, axis: np.ndarray) -> int:
    """Bitmask of headings attainable at EVERY pose along the insertion stroke."""
    m = 0xFFF
    for t in np.linspace(0.0, args.depth, STEPS):
        m &= mask_at(site + axis * t)
        if m == 0:
            break
    return m


print(f"  {'r':>5} {'z':>6} {'az':>5} {'axis':>6} | {'#hdg':>5} | ok  headings [deg]")
print("  " + "-" * 62)
best = []
n_ok = n_tot = 0
for zc in [0.05, 0.08, 0.11, 0.14, 0.18]:
    for rc in [0.18, 0.22, 0.26, 0.30]:
        for azc in [-30.0, 0.0, 30.0]:
            a = np.radians(azc)
            site = np.array([rc * np.cos(a), rc * np.sin(a), zc])
            radial = np.array([np.cos(a), np.sin(a), 0.0])
            tangent = np.array([-np.sin(a), np.cos(a), 0.0])
            for name, axis in [("radial", radial), ("tang+", tangent), ("tang-", -tangent)]:
                m = stroke_ok(site, axis)
                n = bin(int(m)).count("1")
                n_tot += 1
                n_ok += n > 0
                if n:
                    best.append((n, rc, zc, azc, name, m))
                avail = ",".join(f"{b * 30 - 180:+.0f}" for b in range(12) if m >> b & 1) or "none"
                print(f"  {rc:5.2f} {zc:6.2f} {azc:+5.0f} {name:>6} | {n:5d} | "
                      f"{'OK' if n else '--'}  {avail}")

print(f"\n  {n_ok} of {n_tot} (site, insertion-axis) pairs hold a constant heading over the")
print(f"  whole {args.depth * 1000:.0f} mm stroke ({STEPS} poses checked along it).")
print("\n  best candidates (most heading freedom):")
for n, rc, zc, azc, name, m in sorted(best, reverse=True)[:10]:
    hd = ",".join(f"{b * 30 - 180:+.0f}" for b in range(12) if m >> b & 1)
    print(f"    r={rc:.2f} z={zc:.2f} az={azc:+.0f} axis={name:<6} {n} headings: {hd}")
