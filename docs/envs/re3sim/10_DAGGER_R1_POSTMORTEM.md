# 10 — DAgger round 1 postmortem: the student collapsed; the investigation

*2026-08-12 evening, updated 08-13 late. Status: **REOPENED** — the relabel flavor
(true DAgger, completely different data structure) ALSO collapsed at a 1.35 % mix
(vbc_vdr3: 3.1 %/3.1 %), which breaks the "takeover data is toxic" story: at that
dose, no distributional mechanism is credible. The remaining confound is perfect:
every training that succeeded ran before 08-12 noon; every training since had
dagger data AND collapsed. **The mandatory control is in flight** (unit `repro-vdr`):
exact repro of vbc_vdr, base 4 only, seed 1, today's environment. Repro ≈26 % →
any dagger admixture is genuinely toxic (hunt the sharp mechanism: mixed-shard-type
training path, label-chunk stat skew — vdr3's action_mean shifted 0.26 via the
50-row-per-label stats mass — or contract subtlety). Repro collapses → the TRAINING
stack broke after 08-12 noon and every dagger conclusion below is an artifact.
Normalizer note: sub40 collapsed with ≤0.004 stat diffs, so stat skew alone is NOT
the common mechanism.*

## ⭐ The repro verdict (08-13 night): TRAINING-RUN VARIANCE IS THE STORY

The exact repro of vbc_vdr (base 4 only, seed 1, identical command) scored
**12.5 %** — neither the original 26.6 % nor the dagger-mix 3–6 %. Same-config
retrains are NOT reproducible: seed pins the data order at best; CUDA kernels,
DataLoader worker scheduling and augmentation draw order make every training a
fresh sample from a wide distribution. Consequences, in order of importance:

1. **Every single-run comparison this campaign has made carries ±huge error bars**:
   base-vs-vdr (25.0 vs 26.6), the DR-degradation deltas, and the magnitude (not
   necessarily the existence) of the dagger collapse. Eval-side noise for a FIXED
   ckpt is small (26.6 → 25.0 on re-eval); the variance lives in TRAINING.
2. What survives: four independent dagger-mix trainings all scored **3.1–6.2 %**,
   below the non-dagger band seen so far (12.5–26.6 %) — dagger harm remains
   likely, but its size is unknown until the band is mapped.
3. **repro2 = 15.6 %** (08-14 early). The base-config band is {12.5, 15.6, 26.6}:
   mean ≈ 18 %, and the original 26.6 % was the lucky draw. Against that band, the
   four dagger-mix runs (3.1–6.2 %) sit strictly below every non-dagger run —
   rank-test p ≈ 0.03 — so **dagger harm is real (≈ −14 pts vs the band mean),
   while every OTHER pairwise conclusion (base vs vdr, DR deltas) dissolves into
   training noise.**
4. Going forward, no two configs may be compared on single runs — minimum 2–3
   trainings per config, and eval with 4 seeds (128 eps) rather than 2. Recipe
   changes that stabilize training (EMA of weights, longer/cosine schedules,
   deeper eval) are candidates — Big Will's call, since the recipe's numbers were
   paid for upstream.
5. Overnight (unit `night-band`): repro3 + 2-extra-seed evals of all band ckpts
   to tighten both sides of the picture.

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

## Round 2 of evidence (2026-08-13 morning) — the full elimination

* **Sub40 (~3 % mix) ALSO collapsed**: 4.7 % / 4.7 %. Dose-independence.
* **Eval acquitted**: re-evaluating the known-good vbc_vdr ckpt in the same-day
  environment reproduced **25.0 %** (vs 26.6 % original) — the eval harness did not
  drift.
* **Normalizer acquitted**: stats tensors differ by ≤0.025 across good/collapsed
  ckpts. **No NaNs / outliers**: dagger proprio/action ranges ≈ base ranges across
  all 238 shards.
* **Open-loop probes (the decisive ones):** on base-dataset states, phase-binned
  chunk-prediction MSE is *equally good for all three policies* (0.005–0.036
  everywhere; sub40 ≈ vdr even at t=0). The collapsed policies imitate the expert
  perfectly on expert states — **the failure exists only in closed loop.**

## Conclusion (evidence-backed)

The takeover-DAgger data — at any tested dose — **destabilizes the closed loop
without touching open-loop competence.** Mechanism (hypothesis, but the only one
consistent with all of the above): takeover demos pair *mildly-to-strongly
off-nominal states* with *recovery-transit actions*; in closed loop the student's
own small deviations continuously visit mildly off-nominal states, so it falls into
recovery-mode behaviour it cannot execute to completion, drifts further off-nominal,
and re-triggers — an attractor. Even 40 episodes plant it. Note the takeover demos
supervise states the EXPERT visits after takeover, never the states the STUDENT
actually visits under its own policy — precisely the covariate-shift gap true DAgger
exists to close.

**Recommendation to Big Will**: reject the takeover flavor (evidence above); if
DAgger continues, build **true relabel-DAgger with cuRobo** — roll the student
closed-loop, at each chunk boundary plan the expert's chunk FROM the student's
actual state, store `label_chunks` (the loader already supports that shard type,
untouched all along). That supervises exactly the distribution where the disease
lives. Alternative: shelve DAgger and push base-data levers (demo count, wrist
resolution). Takeover data (`dagger_vdr_s41`) stays on disk for reference but must
NOT enter any training mix.

## Rules this adds (candidate traps until the ablation rules)

* Take the *unrecorded settle* with you whenever porting takeover-DAgger: the shard
  must open on a world at rest.
* A healthy training loss says nothing about label consistency — the collapse was
  invisible until sim eval.
* Evaluate any new data source with a same-seed ablation before believing anything
  about the mix.
