# 10 — DAgger round 1 postmortem: the student collapsed; the investigation

*2026-08-12 evening, updated 08-13 early. Status: **DATA-CAUSAL, PROVEN** — the
seed ablation ruled variance out; the mix-ratio probe is in flight. Every claim below
is tagged measured / tested / hypothesis.*

## ⭐ Ablation verdict (2026-08-13 ~02:30)

**base 4 + dagger @ seed 1** (vbc_vdr's own seed): **3.1 % nominal / 4.7 % DR**,
no_lift 91–94 % — collapsed even harder than seed 2. The DAgger data is causal;
seed variance is dead. Grid so far:

| student | data | seed | nominal | DR |
|---|---|---|---|---|
| vbc_vdr | base 4 | 1 | 26.6 % | 20.3 % |
| vbc_vdr2 | base 4 + dagger (16 %) | 2 | 6.2 % | 3.1 % |
| vbc_vdr2_s1 | base 4 + dagger (16 %) | 1 | 3.1 % | 4.7 % |

Start-state measurements (60 eps/side): dagger t0 joint-distance p10/50/90 =
1.68/2.53/3.07 vs base 1.54/1.61/1.72 (drives genuinely drift — that IS DAgger);
~10 % start fingers-closed; cube stays in the graspable r-band (0.23–0.28) both
sides. Nothing mechanically wrong found beyond the (fixed, 4.6 %) drop-labels —
the harm looks distributional: 16 % of samples are expert transits/recoveries from
mid-workspace poses, and mixed at that ratio they erase the nominal approach
behaviour rather than add recovery competence.

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

## The next probe (in flight, unit `ablate-dagger-sub40`)

**Mix-ratio test**: base 4 + a 40-episode subsample of dagger (~3 % of samples,
every 6th shard), seed 1, both eval cells (`runs/vbc_vdr2_sub40`).

* If ~3 % still collapses → per-sample toxicity, a bug is still hiding in the shards.
* If it's neutral/positive → the 16 % ratio was an overdose; find the tolerable dose
  (classic DAgger β-mixing) or switch designs.

## Design fork for Big Will (decision pending)

1. **Dose the takeover data** (cheapest): cap dagger at the ratio the sub40 probe
   supports, possibly trimming each episode's long transit prefix.
2. **True relabel-DAgger with cuRobo**: label the student's own visited states with
   expert chunk plans (exp08's `label_chunks` flavor) — plans from each chunk
   boundary, ~2-3× collection cost, but supervision lands exactly on the student's
   state distribution instead of demos that merely start there.
3. **Drop DAgger for now**: the 2×2 + matrix say the base student's grasp is weak
   everywhere, not just off-distribution — more/better base demos or higher wrist
   resolution may be the truer lever.

## Rules this adds (candidate traps until the ablation rules)

* Take the *unrecorded settle* with you whenever porting takeover-DAgger: the shard
  must open on a world at rest.
* A healthy training loss says nothing about label consistency — the collapse was
  invisible until sim eval.
* Evaluate any new data source with a same-seed ablation before believing anything
  about the mix.
