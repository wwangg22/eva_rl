# 10 — DAgger round 1 postmortem: the student collapsed; the investigation

*2026-08-12 evening. Status: OPEN — decisive ablation in flight. Every claim below is
tagged measured / tested / hypothesis.*

## What happened (measured)

Round 1 collection itself looked excellent: 384 episodes under DR, vbc_vdr driving
60–300 steps before cuRobo takeover, **62.0 % expert recovery success** → 238 shards
(12 GB), clean-grasp 84.6 %. Then vbc_vdr2 (base 4 + dagger, seed 2, same recipe)
trained with a healthy loss curve (final MSE 0.048) and **collapsed at eval**:

| student | nominal | DR |
|---|---|---|
| vbc_vdr (base 4, seed 1) | 26.6 % | 20.3 % |
| **vbc_vdr2 (base 4 + dagger, seed 2)** | **6.2 %** | **3.1 %** |

`lifted` fell 39 %→14 %, `no_lift` rose to ~85 % in both cells. The gap is ~3.3σ —
not eval noise.

## Suspects tested against data (all measured on the actual shards)

1. **Drop-the-cube opening labels — REAL DEFECT, too small to explain the collapse.**
   The port omitted `collect_demos.py`'s unrecorded post-takeover freeze, so shards
   open the instant the settle-hold opens the gripper: an episode whose student was
   holding the cube gets *"open the gripper"* as its first labels — supervision that
   contradicts the carry. Measured: cube already lifted at t0 in **11/238 shards
   (4.6 %)**. Poisonous but ~11 episodes of 1,336 ≈ far below collapse scale.
   **FIXED** the same evening: `dagger_drive` now ends with a 40-step unrecorded
   freeze (gripper open, world settles) before the shard opens.
2. **Idle/hold contamination from plan retries (`plan_fail: 356`) — REFUTED.**
   Idle-step fraction (all |joint_vel| < 0.01): dagger **0.8 %** vs base **1.3 %**.
   Planning happens between sim steps; retries record nothing.
3. **Broken frames / desk-splat loss — REFUTED.** Image means normal (~153–162), no
   black frames, filmstrips visually sane. The alarming "pink desk" episode is a
   valid heavy DR color draw: desk-patch RGB distributions are statistically
   identical (R−G spread p10/50/90 = 30/37/40 dagger vs 29/36/39 base).
4. **Contract violations — none found.** Shard keys/shapes byte-compatible, actions
   and joints in normal ranges, loads through `WorkstationVisionDataset` cleanly
   (validated at smoke time and again on the production dir).

## Open hypotheses (not yet separable from one another)

* **The data is causal at the distribution level**: 16 % of samples are expert
  transits *from arbitrary mid-workspace poses*; under a 50-chunk policy these may
  interfere with the nominal approach behaviour more than they help (classic DAgger
  needs the student's own states + on-policy correction, and this takeover variant
  labels only full recoveries, successes-only at that).
* **Seed variance**: vbc_vdr2 used seed 2; vbc_vdr used seed 1. A 4× collapse from
  seed alone would be extraordinary, but it is the cheapest hypothesis to kill.

## The decisive experiment (in flight, unit `ablate-dagger`)

Retrain **base 4 + dagger at seed 1** (vbc_vdr's own seed), then run both eval cells,
fully automated (`runs/vbc_vdr2_s1`, driver log `runs/ablate_dagger_driver.log`).

* If it collapses too → **the dagger data is causal**; next lever is data-side
  (e.g. drop the takeover-transit prefix and keep only the from-rest recovery, mix
  ratio caps, or DAgger only into states past the approach).
* If it lands ~26 % → seed-2 interaction/fluke; rerun vdr2 config across seeds.

## Rules this adds (candidate traps until the ablation rules)

* Take the *unrecorded settle* with you whenever porting takeover-DAgger: the shard
  must open on a world at rest.
* A healthy training loss says nothing about label consistency — the collapse was
  invisible until sim eval.
* Evaluate any new data source with a same-seed ablation before believing anything
  about the mix.
