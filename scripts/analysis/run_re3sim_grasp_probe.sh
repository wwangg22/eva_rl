#!/usr/bin/env bash
# B1 gate: grasp feasibility for every object in the 2026-08-05 workstation capture.
# One process per object -- a rigid body's collider cannot be resized after the scene is
# built (docs/CHALLENGE_SUITE.md C8), so each geometry needs its own stage.
set -u
cd "$(dirname "$0")/../.."
OUT=logs/analysis/re3sim_grasp
mkdir -p "$OUT"
for obj in control_56 control_24 rubixcube tapemeasure rolloftape rolloftape_onedge; do
  echo "=============================================================="
  echo "=== $obj"
  echo "=============================================================="
  python -u scripts/analysis/re3sim_grasp_probe.py --object "$obj" --headless \
    2>&1 | grep -v -e "^\[Warning\]" -e "omni\." -e "^\[[0-9.]*s\]" | tail -60
done
echo "=== B1 probe sweep finished"
