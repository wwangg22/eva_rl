# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Query the reachability map produced by ``reachability_map.py``.

Answers the specific design questions a new task has to clear:
where can the fingers point *down*, how big is the side-grasp envelope at table
height, and how much wrist roll is left there. Pure numpy -- no Isaac Sim needed.

.. code-block:: bash

    python scripts/analysis/query_reachability.py
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Query the arm reachability map.")
parser.add_argument("--npz", type=str, default="logs/analysis/reachability/reachability.npz")
parser.add_argument("--min_count", type=int, default=5, help="samples per voxel to trust it")
args = argparse.Namespace(**vars(parser.parse_args()))

d = np.load(args.npz)
c = d["centers"]
cnt = d["counts"]
mn = d["min_approach_z"]  # most-downward approach axis seen in this voxel
mx = d["max_approach_z"]
nroll = d["n_roll_bins"]

rel = cnt >= args.min_count
c, mn, mx, nroll, cnt = c[rel], mn[rel], mx[rel], nroll[rel], cnt[rel]
r = np.linalg.norm(c[:, :2], axis=1)
z = c[:, 2]
az = np.degrees(np.arctan2(c[:, 1], c[:, 0]))

print(f"reliable voxels: {len(c)}  (>= {args.min_count} samples each)\n")

# ---------------------------------------------------------------- top-down
print("=" * 74)
print("Q1. WHERE CAN THE FINGERS POINT DOWN?  (approach_z <= -0.90, i.e. <26 deg off vertical)")
print("=" * 74)
td = mn <= -0.90
print(f"  voxels with a top-down-capable pose: {td.sum()} / {len(c)}  ({100 * td.sum() / len(c):.2f}%)")
if td.sum():
    print(f"  their z range   : {z[td].min():+.3f} .. {z[td].max():+.3f} m  (table top = 0.000)")
    print(f"  their r_xy range: {r[td].min():.3f} .. {r[td].max():.3f} m")
    print("\n  breakdown by height band:")
    for zlo, zhi in [(-0.30, -0.10), (-0.10, 0.0), (0.0, 0.10), (0.10, 0.25), (0.25, 0.45), (0.45, 0.90)]:
        m = (z >= zlo) & (z < zhi)
        if m.sum() == 0:
            continue
        f = td & m
        rr = f"r {r[f].min():.2f}-{r[f].max():.2f}" if f.sum() else "-"
        print(f"    z [{zlo:+.2f},{zhi:+.2f}): {f.sum():6d}/{m.sum():6d} voxels top-down ({100*f.sum()/m.sum():5.2f}%)  {rr}")

# ------------------------------------------------------- angled-down (45 deg)
print("\n" + "=" * 74)
print("Q2. HOW STEEP CAN THE APPROACH GET AT TABLE HEIGHT? (z in [0.00, 0.10))")
print("=" * 74)
band = (z >= 0.0) & (z < 0.10)
for lim, label in [(-0.9, "top-down (<26 deg off vert)"), (-0.7, "steep (<45 deg off vert)"),
                   (-0.5, "45 deg down"), (-0.34, "20 deg down"), (-0.17, "10 deg down")]:
    f = band & (mn <= lim)
    print(f"  approach_z <= {lim:+.2f}  [{label:28s}]: {f.sum():6d}/{band.sum():6d} voxels ({100*f.sum()/max(band.sum(),1):5.2f}%)")
if band.sum():
    print(f"\n  steepest approach anywhere in this band: approach_z = {mn[band].min():+.4f} "
          f"({np.degrees(np.arccos(-mn[band].min())):.1f} deg off vertical)")

# ------------------------------------------------- side-grasp working envelope
print("\n" + "=" * 74)
print("Q3. SIDE-GRASP ENVELOPE AT OBJECT HEIGHT (z in [0.01, 0.08), near-horizontal approach)")
print("=" * 74)
obj = (z >= 0.01) & (z < 0.08)
horiz = obj & (mn <= 0.10) & (mx >= -0.10)  # a near-horizontal approach exists here
print(f"  voxels with a near-horizontal approach: {horiz.sum()} / {obj.sum()}")
if horiz.sum():
    print(f"  radius   : {r[horiz].min():.3f} .. {r[horiz].max():.3f} m")
    print(f"  azimuth  : {az[horiz].min():+.1f} .. {az[horiz].max():+.1f} deg")
    print("\n  radial shells (front sector |azimuth| <= 60 deg):")
    front = horiz & (np.abs(az) <= 60)
    for rlo in np.arange(0.10, 0.55, 0.05):
        m = front & (r >= rlo) & (r < rlo + 0.05)
        if m.sum() == 0:
            print(f"    r [{rlo:.2f},{rlo+0.05:.2f}): --")
            continue
        print(f"    r [{rlo:.2f},{rlo+0.05:.2f}): {m.sum():5d} voxels  "
              f"az {az[m].min():+6.1f}..{az[m].max():+6.1f} deg  median roll bins {np.median(nroll[m]):.0f}/12")

# ------------------------------------------------------------- roll freedom
print("\n" + "=" * 74)
print("Q4. WRIST-ROLL FREEDOM (how many of 12 x 30-deg roll bins are attainable)")
print("=" * 74)
print("  -- matters for orientation-specific placement and for regrasp/reorient tasks")
for zlo, zhi, lab in [(0.01, 0.08, "object height"), (0.08, 0.16, "carry height"), (0.16, 0.30, "high")]:
    m = (z >= zlo) & (z < zhi) & (np.abs(az) <= 60) & (r >= 0.15) & (r <= 0.35)
    if m.sum() == 0:
        continue
    print(f"  {lab:14s} (r 0.15-0.35, |az|<=60): median {np.median(nroll[m]):.0f}/12, "
          f"p90 {np.percentile(nroll[m], 90):.0f}/12, max {nroll[m].max()}/12, n={m.sum()}")

# --------------------------------------------------- out-of-reach ring (tool use)
print("\n" + "=" * 74)
print("Q5. THE 'JUST OUT OF REACH' RING  (relevant to tool-use / non-prehensile tasks)")
print("=" * 74)
print("  TCP-reachable at object height, per radial shell, front sector:")
front_obj = obj & (np.abs(az) <= 45)
for rlo in np.arange(0.20, 0.60, 0.05):
    m = front_obj & (r >= rlo) & (r < rlo + 0.05)
    n_horiz = (m & horiz).sum()
    print(f"    r [{rlo:.2f},{rlo+0.05:.2f}): {m.sum():5d} TCP-reachable voxels, "
          f"{n_horiz:5d} of them side-approachable")
