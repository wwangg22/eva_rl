# 09 — The robustness matrix: vbc_vdr per-axis, with held-out severity

*2026-08-12, `eval_vdr_matrix.sh re3sim/runs/vbc_vdr/ckpt_final.pt` — 10 rows × 64
episodes (seeds 88000+88001 × 32 envs) on `-VisionDR-Play-v0`, one DR group at a time
at pinned scale, `RE3SIM_SPLATS_PER_ENV=1`. Raw jsons: `reBot_ACT/re3sim/runs/vdr_matrix/`.*

| row | success | no_lift | near_miss | dropped |
|---|---|---|---|---|
| nominal (all off) | **31.2 %** | 60.9 % | 6.2 % | 0.0 % |
| cam_1x | 21.9 % | 65.6 % | 9.4 % | 0.0 % |
| **cam_1.5x** (held-out) | **18.8 %** | **81.2 %** | 0.0 % | 0.0 % |
| focal_1x | 23.4 % | 60.9 % | 9.4 % | 0.0 % |
| wrist_1x | 23.4 % | 71.9 % | 4.7 % | 0.0 % |
| light_1x | 21.9 % | 65.6 % | 10.9 % | 0.0 % |
| light_1.5x (held-out) | 23.4 % | 60.9 % | 9.4 % | 0.0 % |
| bg_1x | 21.9 % | 73.4 % | 3.1 % | 0.0 % |
| all_1x | 28.1 % | 65.6 % | 4.7 % | 0.0 % |
| all_1.5x (held-out) | 20.3 % | 68.8 % | 7.8 % | 0.0 % |

*(SE per row ≈ 5–6 pts at n=64 — differences under ~10 pts are ~1σ. Treat shapes, not
single cells.)*

## Reading

1. **No axis collapses.** Every single-axis row sits in a narrow 21.9–23.4 % band
   (~8–9 pts under the 31.2 % nominal row); even all-axes-at-1.5×-severity keeps
   20.3 %. The DR-trained student's robustness curve is *shallow and uniform* — there
   is no single broken axis for the next DR round to chase. That is the good version
   of this result: the training DR covered its axes.
2. **Camera pose is the leading (unproven) suspect.** Worst at both scales (21.9 →
   18.8 %) and the only row where `no_lift` spikes to 81 %, with `near_miss` going to
   zero — under camera-pose shift the student doesn't even *reach* wrong, it stops
   reaching. Geometrically sensible: cam pose corrupts the hand-eye mapping itself.
   A targeted n≥256 cam-only run would settle it if it ever matters; today it doesn't
   change the plan.
3. **Lighting held-out severity is free** (23.4 % at 1.5× vs 21.9 % at 1×, i.e. flat):
   appearance axes extrapolate; the geometric axis (cam) is the one that decays with
   severity.
4. **`dropped` = 0.0 % in all ten rows** (640 episodes) — the 2×2's conclusion at
   triple the sample: the post-grasp funnel is lossless everywhere; every point of
   failure is at or before the grasp.
5. **Protocol footnote:** all_1x here (28.1 %) ran with `RE3SIM_SPLATS_PER_ENV=1`
   while the 2×2's DR cell (20.3 %, same ckpt/seeds) used the task default — the two
   are ~1.2σ apart, so noise alone could explain it, but don't quote them
   interchangeably.

## What it means for the plan

The binding constraint is **not robustness** — it's the ~25–30 % base grasp rate.
The matrix says the DR training already bought a flat curve; buying more DR axes
would polish a number that is bottlenecked elsewhere. The lever ranked first by both
docs (08 §levers, this one) is **DAgger at the grasp with the cuRobo expert** —
approved by Big Will 2026-08-12 ("Use curobo for dagger"), collector built into
`run_expert_ws.py --dagger-ckpt` the same day.
