# 08 — The student 2×2: what visual DR bought (first evidence)

*2026-08-12. Both students trained on the four cuRobo datasets (§HANDOFF 1), evaluated
by `reBot_ACT/re3sim/act/run_eval_2x2.sh` → `eval_flow_vision.py`, 64 episodes per cell
(seeds 88000+88001 × 32 envs), `-Play` task spacing. Raw jsons + per-cell logs:
`reBot_ACT/re3sim/runs/eval_2x2/`.*

## The grid (success, pooled over 64 eps)

| student ↓ / eval env → | nominal `-Vision-Play` | DR `-VisionDR-Play` | degradation |
|---|---|---|---|
| **vbc_base** (nominal data, no augment) | **25.0 %** (seeds 16/34) | **14.1 %** (12/16) | −10.9 pts |
| **vbc_vdr** (nominal+3×DR data, augment 1.0) | **26.6 %** (34/19) | **20.3 %** (16/25) | −6.3 pts |

Reference gate: the cuRobo expert is 70.1 % on nominal.

## Full failure taxonomy (pooled)

| cell | success | lifted | over_box | dropped | near_miss | carried_astray | no_lift |
|---|---|---|---|---|---|---|---|
| vbc_base × nom | 25.0 % | 31.2 % | 26.6 % | 0.0 % | 1.6 % | 4.7 % | 68.8 % |
| vbc_base × DR | 14.1 % | 18.8 % | 18.8 % | 0.0 % | 4.7 % | 0.0 % | 81.2 % |
| vbc_vdr × nom | 26.6 % | 39.1 % | 37.5 % | 0.0 % | 10.9 % | 1.6 % | 60.9 % |
| vbc_vdr × DR | 20.3 % | 28.1 % | 25.0 % | 0.0 % | 4.7 % | 3.1 % | 71.9 % |

## What the grid says

1. **The bottleneck is the grasp, overwhelmingly.** `no_lift` is 61–81 % in every cell,
   while `dropped` is 0.0 % in every cell: once the cube is lifted, ~2/3–4/5 of episodes
   convert to success and nothing is ever dropped in transit. This is exactly the
   upstream prediction ("grasp precision is where students die") — every phase after
   the grasp is nearly lossless, so ALL improvement effort goes to the grasp.
2. **DR training cost nothing on nominal** (26.6 % vs 25.0 % — same within noise), and
   the vdr student lifts more often in every condition. The appearance-only DR data did
   not corrupt the nominal behaviour — consistent with the collection-side evidence
   (every DR collection ≈ the 70 % nominal gate).
3. **Directional robustness win, not yet a proven one.** Under DR the vdr student keeps
   20.3 % vs the base student's 14.1 % (+6.2 pts) and degrades less (−6.3 vs −10.9).
   At n=64/cell the standard error per cell is ~5 pts, so this gap is ~1σ —
   *directionally right, statistically unfinished.* The per-axis matrix (Step M, in
   flight on vbc_vdr) adds 640 more episodes and will sharpen this considerably.
4. **Per-seed spread is large** (e.g. vbc_base nominal: 15.6 % vs 34.4 % on the two
   seeds). Any future single-seed comparison at n=32 is noise — never quote one.
5. **Absolute level:** both students sit at ~⅓ of the expert's 70.1 %. A 25 %
   first-BC-round student with a clean post-grasp funnel is credible signal for
   DAgger (HANDOFF §7 Step D), whose whole point is to fix the state-distribution
   mismatch at the exact phase (approach/grasp) where the student diverges.

## Levers, ranked by this evidence

1. **DAgger under DR at the grasp** — the failure is concentrated where DAgger helps
   most. ⚠ The existing DAgger path is ArmKin-based; a cuRobo DAgger collector needs
   Big Will's go-ahead first (directive 1).
2. **Wrist-camera resolution / action representation** — the cube is ~7 px in the
   workspace cam; if the matrix's `wrist` and `cam` axes come back weak, this rises.
   Re-prices RAM/disk per HANDOFF §6c.
3. **More demos** — weakest lever: the funnel says distribution mismatch, not data
   volume.
