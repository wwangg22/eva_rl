# HANDOFF — re3sim workstation env

**Rewritten 2026-08-06, ~00:10.** Supersedes the 2026-08-05 planning-session handoff.
Read this first, then `01_STEP1_PLAN.md`, then `pickandplace1/LESSONS_INHERITED.md`.

Everything below is either **measured** on this box or **decided by the user**. Where
something is assumed, it says so.

---

## 0. TL;DR

**Updated 2026-08-06 ~04:15.** Items 1–3 are from the earlier session and still hold; 4–8 are
this session's.

1. **B1 is resolved and its premise was wrong.** The rubix cube grasps at **99 %** with the
   TCP at **25 mm** — *below* the "44 mm TCP floor" (C9) that the whole blocker was built on.
2. **The task was narrowed by the user**: only the cube is a grasp target; the roll of tape
   and the tape measure are **clutter**.
3. **The env exists** — `Rebot-Workstation-PickPlace1-v0`, registered, builds, 41-D obs.
   Its success predicate passes all 8 negative controls, and the spawn-sampler bug is fixed
   and re-verified (**ALL CHECKS PASSED**, §8.1).
4. ⭐ **Sequencing corrected by the user: the Re³Sim pipeline IS env setup**, not a follow-on
   track. See **[02_RECONSTRUCTION.md](02_RECONSTRUCTION.md)**.
5. **Scene alignment PASSED and beats the previous capture** — 262/300 frames, triangulation
   RMS **15.20 mm** (was 16.7), median orientation spread **0.66°** (was 0.91°). The fitted
   long table edge reproduces the previous capture's 1.642 m exactly, across two different
   marker sizes: a strong end-to-end check on metric scale. The desk is now IN the env as
   356,768 NuRec gaussians plus a fitted collider slab, and the smoke test passes with objects
   settling at exactly their rest height (0.00 mm, was 3.00 mm on the stock table).
6. **A POLICY EXISTS.** 896 demos (278 successful, 31.0 %) → 60 k steps of flow-matching BC →
   **21.9 % pooled** over three held-out spawn seeds on the demo-start protocol, which
   **retains 68 % of the 32.4 % expert** — a healthy BC port. From a cold `env.reset()` it is
   **0.0 %**, and that gap is the transit problem priced exactly. Both numbers are reported;
   see **[03_EXPERT_AND_BC.md](03_EXPERT_AND_BC.md) §4a**.
7. **The object captures do not contain their objects.** Cube and box both verified ABSENT
   from their sparse *and* dense reconstructions. Not a code problem — a re-shoot. A 30-second
   checker now exists (`check_object_present.py`). The three movable objects therefore keep
   analytic primitives, announced per-object at env load.
8. **The transit is the one open engineering problem**, root cause identified and confirmed,
   fix specified — [03 §3b](03_EXPERT_AND_BC.md).

---

## 1. What this project is

Turn a **real capture of the user's desk** into a photoreal, metrically-correct Isaac Lab
environment, then train a policy in it through the `eva_bc` ladder.

| repo | role |
|---|---|
| `~/Desktop/isaacLab/Re3Sim` | **real-to-sim** — video → COLMAP → 3DGS + OpenMVS → aligned USD |
| `~/Desktop/isaacLab/eva_rl` | **the env**, its docs, its probes |
| `~/Desktop/isaacLab/eva_bc` | **the expert + BC/RL ladder**, and the lab notebook of past lessons |

The split matters: the clutter task's env lives in `eva_rl`, its expert and BC code in
`eva_bc/clutter/`. This project mirrors that — env in `eva_rl/…/re3sim/`, expert in
`eva_bc/re3sim/`.

---

## 2. User decisions (binding)

Numbered as in the previous handoff, with the changes marked.

1. **reBot-B601-RS = RS-rebot-dev-arm.** `source/reBot_RL/data/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda`.
2. **The arm mounts where the marker was**, offset down by the iPad thickness. Marker frame
   origin is the marker's *top* surface ⇒ desk surface and base plate both at
   `z_marker = −0.0065`, which maps to **`z_env = 0`**.
3. ~~All three objects must be grasp targets.~~ **SUPERSEDED 2026-08-06:** *"honestly im fine
   with just grasping the rubix cube, and other stuff act as clutter, for pick and place 1
   env."* Only the cube is a target.
4. **The box is an open container**; objects go *into* it.
5. Objects spawn at random reachable positions with **random yaw only** — never flipped, so
   the synthetic underside is never exposed.
6. Domain randomisation: mass, size, friction.
7. Track everything in `eva_rl/docs/envs/re3sim/`.
8. **NEW — the user pushed back twice on shortcuts**, and was right both times:
   - *"i am a little worried you are not using re3sim pipeline to setup env, and trying to cut
     corners"*
   - *"i thought the caliper measurements were just to scale the mesh generated from re3sim?"*

   **They are correct.** In Re³Sim the *mesh* is the geometry (`canonicalise()` recovers shape
   and orientation) and the calipers supply only the **scale** a marker-less object capture
   cannot. Using analytic primitives as colliders is a simplification beyond what the pipeline
   does. See §6 for exactly what is provisional.

---

## 3. ⭐ THE BIG MEASURED RESULT — B1 is dead

### 3.1 What was believed

`docs/CHALLENGE_SUITE.md` **C9**: *"The TCP cannot go below ~44 mm above the table"*, called
"the single most restrictive constraint in the suite". Combined with **C1** (no top-down grasp
below z = 0.10 m), the previous handoff concluded the tape roll (24 mm) and tape measure
(36 mm) were entirely below the graspable floor and the cube (56 mm) was "marginal".

### 3.2 What is actually true

**C9 does not constrain grasping at all.** It was measured with the **fingers shut**
(`tcp_floor.py:125`, `a[:, 6] = -1.0`), which is the widest part of the gripper bottoming out
on the table. A *grasp* approaches with the fingers **open**, and the finger bodies reach down
to **z ≈ 5 mm**.

Measured by `eva_bc/re3sim/probes/p01_grasp_feasibility.py` — real geometry, full manoeuvre
(approach wide along a Cartesian chain → close → lift 90 mm), enclosure verified via the
C3 gap calibration, 128 envs each with its own friction bucket:

| grip z [mm] | 8 | 12 | 16 | 20 | **25** | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **rubixcube** (56×56, 73 g) | 0 % | 0 % | 16 % | 5 % | **99 %** | 77 % | 37 % | 18 % | 2 % | 79 % |
| control_56 (56 tall, 40 wide) | 0 % | 13 % | 25 % | 2 % | 66 % | 73 % | **100 %** | 95 % | 24 % | 72 % |
| control_24 (24 tall, 40 wide) | 0 % | 7 % | **87 %** | 48 % | — | — | — | — | — | — |

**Use grip z = 25 mm for the cube.** `gap` reads 56.1 mm there — exactly the cube's width, so
the fingers stalled *on the object*, not on air.

### 3.3 The two probe bugs, and why they matter

This took three runs. Both failures were in the **search**, not the hardware, and both are the
kind that produce confident wrong "this cannot be grasped" conclusions.

**Run 1 — inherited `grasp_geometry.py`'s defective cost.** `cost = |dp| + 0.25·(1 − |o·Y|)`,
no floor term, no restarts. With `|dp|` in metres, 1 mm of position error costs 0.001 against
orientation terms of order 0.25. Result: control_24 → 100 %, control_56 → **0 % at every
height**, `gap = −1.2 mm` (shut on air). A 56 mm block is *easier* than a 24 mm one, so a
taller control failing everywhere is a defect, not a fact. `_kin.py` names this exact failure:
it lets the CEM return poses with `gripper_end` **inside the table**.

**Run 2 — over-corrected.** Ported `_kin.ArmKin.cem` (hinged position cost `w_pos = 200`,
floor penalty, restarts) but copied the clutter expert's `floor_z = 0.012` and added a signed
`a_des = +x` approach term. The positive control **failed**: `posErr` 0.7–0.9 mm (fine) but
`o_align` collapsed to **0.40–0.69** against a 0.99 gate. Cause: the clutter task's blocks are
70 mm tall, but grasping a 24 mm object *requires* the fingers at z ≈ 5 mm — a 12 mm floor
penalises the exact configuration the grasp needs — and a strictly horizontal approach is
over-constrained for an arm C1 says approaches tilted.

**Run 3 — correct.** `floor_z = 0.002` (keep the arm out of the *table*, nothing more),
approach axis left **free**, standoff taken from the *achieved* `a_hat` projected horizontal.

**Transferable lesson:** *search hyper-parameters do not port across object scales.* The gate
thresholds and the floor height are properties of the task geometry, not of the arm.

**Also learned:** success tracks `o_align` tightly (0.90 at the 99 % cell, 0.65–0.78 at the
dead ones). **The expert must gate on alignment and retry**, not trust one CEM draw. The
`o_align ≥ 0.99` gate itself is *not achievable* here and should be relaxed to ~0.90 with
`held` as ground truth.

---

## 4. The task, as it now stands

**Grasp the rubix cube from a random reachable pose with random yaw; place it inside the open
box, which sits stationary at a random reachable spot.** The roll of tape and the tape measure
are clutter — real bodies occupying real space, never grasped.

Why the narrowing is well-founded independent of the user's preference:

- roll of tape is **91 mm** across vs. an **89.1 mm** commanded opening (C3) — it can only be
  forced wider by contact, which is not a grasp strategy;
- tape measure is **184 g**, right at the cell where C2 measured the finger drive failing
  (0/6 holds at 0.25 kg with the authored 100 N/m stiffness);
- the cube is comfortably inside the envelope, so the task is bounded by *skill*, not by
  hardware nothing can fix.

---

## 5. What was built this session

### 5.1 `eva_rl` — the environment

```
source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/
├── __init__.py                 # registers 3 gym IDs
├── workstation_env_cfg.py      # scene / actions / obs / rewards / terminations / events
├── agents/rl_games_ppo_cfg.yaml
└── mdp/
    ├── __init__.py
    ├── common.py               # ⭐ single source of truth: geometry + placed_mask
    ├── events.py               # reset_box, reset_objects, record_clutter_spawn, DR
    ├── observations.py
    ├── rewards.py
    └── terminations.py
scripts/author_workstation_box_usd.py   # the open box, 5 cuboids, kinematic
scripts/test_workstation_env.py         # smoke test + 8 predicate negative controls
scripts/analysis/re3sim_grasp_probe.py  # run-1 probe. SUPERSEDED — see §3.3. Kept for the record.
```

**Gym IDs:** `Rebot-Workstation-PickPlace1-v0`, `-Play-v0`, `-Strict-v0`.

**Observation: 41-D** = joint_pos 8 + joint_vel 8 + cube pose 7 + box pose 4 + clutter 6 +
placed 1 + last_action 7. (Coincidentally the same width as `eva_bc/act/dataset.py`'s
hardcoded split — **verify, do not assume, that the split matches.**)

**Action: 7-D**, identical to every other reBot env and load-bearing for the whole `eva_bc`
stack:
```python
a[:6] = (q - q_default) / 0.5     # absolute, default-offset, scale 0.5
a[6]  = +1 open / -1 close        # binary, nothing in between
```

**Geometry** (`mdp/common.py`, all measured 2026-08-05):

| | |
|---|---|
| cube | 56 mm, 73 g — the target |
| roll of tape | 91 mm dia × 24 mm, 42 g — clutter |
| tape measure | 71.5 × **64** × 36 mm, 184 g — clutter. **64 mm is an ESTIMATE**, not on the sheet |
| box outer | 218 × **150** × 93 mm. **150 mm is an ASSUMPTION**, not on the sheet |
| box inner | 212 × 144 mm, wall 3 mm, floor 6 mm |
| cube at rest in box | root z = 34 mm; rim at 93 mm |

**Success predicate** (`placed_mask`) — strengthened per LESSONS B1, bounded on **both** sides:
inside the interior footprint with a 12 mm margin, **above** the interior floor (z > 10 mm),
**below** a 62 mm ceiling well under the rim, and **settled** (speed < 0.05 m/s). Tested in
the box's own rotated frame, not axis-aligned.

**Box is kinematic** — the honest way to express "does not move". At 95 g a dynamic box is
shoved on first contact, and mass tricks only slow that down.

### 5.2 `eva_bc` — the expert and BC side

```
eva_bc/re3sim/
├── probes/
│   ├── _kin.py                     # ⭐ copied verbatim from clutter/probes/. Env-agnostic.
│   ├── p01_grasp_feasibility.py    # the B1 gate. PASSED.
│   ├── run_p01.sh
│   └── out/p01/*.json              # results
├── expert/
│   ├── workstation_expert.py       # RUN. Planner re-priced 2026-08-06 -- see 03_EXPERT_AND_BC.md
│   └── collect_demos.py            # RUN. Plan serially, execute in parallel
├── act/
│   ├── dataset.py                  # 41-D layout, 16/25/7 split
│   ├── train_flow.py               # rectified flow, chunk 50 / commit 15
│   ├── policy_runner.py            # ChunkController + load_checkpoint (asserts the task tag)
│   └── eval_flow.py                # batched sim eval + failure taxonomy
├── run_bc.sh                       # demos -> train -> eval, one command
└── docs/      (empty -- the docs live in eva_rl/docs/envs/re3sim/)
```

**`workstation_expert.py`** — solves the *entire* trajectory (approach → grasp → lift → carry
→ release) **before the fingers close**, because `cem`/`refine` call
`write_joint_state_to_sim`, which teleports the arm and re-opens the fingers; searching after
closing silently drops the cube and reads as a slip (A6, worth 0 %→100 % once already).
Recipe constants: `GRIP_Z = 0.025` (measured), `O_ALIGN_MIN = 0.90` (0.99 is unattainable at
this grip height), `LOW_Z_MIN = 0.002` (**not** the clutter expert's 0.012), `CARRY_Z = 0.175`
(rim 93 + cube half 28 + margin), dense Cartesian waypoints at ≤ 30 mm, clutter as
`cem(avoid=…)` keep-out boxes. The opening axis is snapped to the **nearest cube face normal**
— a parallel jaw closing on a cube *corner* rotates it instead of squeezing it, and spawn yaw
is uniform on [0, 2π).

**`collect_demos.py`** — the design worth keeping: **plan serially, execute in parallel.**
`ArmKin` uses the env count as its CEM population, so the naive approach plans for env 0 and
wastes the other 127. Instead it solves one plan per env, stacks the waypoints into `(n, 6)`
tensors (padding short chains by *holding* the final pose, not by fabricating motion), and
drives all envs at once — `num_envs` demos per batch. Recording is causal (obs captured
*before* the step). Emits a failure taxonomy per batch: plan-failed / never-got-there /
lifted-but-lost / over-box-not-inside.

`_kin.ArmKin` is the single most valuable thing to reuse: batched FK read back from the sim,
hinged-cost CEM with restarts + keep-out boxes, a DLS `refine()`, MDP-bypassing physics
execution (`hold_phys`/`run_phys`), and the C3 `gap()` read-back.

---

## 6. ⚠ What is provisional (the corner-cutting, named)

| piece | status |
|---|---|
| object dimensions, masses | **real** — user's calipers |
| box as an open container | **real and permanent** — must be authored (B2) |
| desk surface, marker→env frame | **real** — derived from the 165 mm marker + 6.5 mm iPad |
| **cube / tape / tapemeasure colliders** | **PROVISIONAL primitives.** Correct path: `extract_object.py --target-height` → `mesh_to_rigid_usd.py`. Swap `spawn=CuboidCfg(...)` → `spawn=UsdFileCfg(...)`; nothing in the MDP reads the collider type |
| **desk appearance** | **PLACEHOLDER** — stock Isaac `SeattleLabTable`, not the splats |
| **object appearance** | **PLACEHOLDER** — coloured primitives |
| `BOX_OUTER_Y = 150 mm` | **ASSUMPTION** — correct from the reconstruction or a caliper |
| `TAPEMEASURE_ACROSS = 64 mm` | **ASSUMPTION** — same |
| **size domain randomisation** | **NOT IMPLEMENTED.** See §8 |

---

## 7. Reconstruction status

`Re3Sim/data/` — **fresh dirs, the 2026-08-03/04 run is untouched**.

| capture | frames | note |
|---|---|---|
| `scene2/` | 300 | ⭐ **ArUco visible in 262/300 (87 %)** — marker gate passed comfortably |
| `object2_rubixcube/` | 300 | 57 frames flagged soft |
| `object2_rolloftape/` | 300 | |
| `object2_tapemeasure/` | 300 | |
| `object2_box/` | 300 | 29 frames flagged soft |

Frame selection is *windowed best-sharpness* (`video_to_frames.py`), not uniform fps.

**Running now:** `systemd --user` unit **`recon-scene`** → `reconstruct.py -i data/scene2`
(COLMAP sparse → 3DGS 30 k → OpenMVS dense). Long — hours. Log:
`Re3Sim/data/scene2/recon.log`. It is **resumable** via `data/scene2/progress.json`.

### ⚠ `reconstruct.py` needed FOUR fixes — the fork was written against an older pycolmap

All four are patched in place, each guarded so the script still runs on the old binding. They
cost ~40 minutes of restarts; anyone reconstructing the *object* captures will hit none of them
now, but should know they exist.

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `TypeError: extract_features(): incompatible function arguments` | SIFT settings moved into `FeatureExtractionOptions`, passed as `extraction_options=` not `sift_options=`; `camera_model` is no longer a top-level kwarg (it belongs to `ImageReaderOptions`) | branch on `hasattr(pycolmap, "FeatureExtractionOptions")` |
| 2 | `AttributeError: module 'pycolmap' has no attribute 'SequentialMatchingOptions'` | the *pairing* strategy split out of the matching options: `SequentialMatchingOptions` → `SequentialPairingOptions`, passed as `pairing_options=`; `matching_options=` now means the SIFT matcher settings | branch on `hasattr(pycolmap, "SequentialPairingOptions")` |
| 3 | **process SIGABRT**, `Failed to read faiss index` | `download_vocab_tree` fetches the COLMAP **3.11.1** tree, which is a legacy **flann** index. COLMAP switched to **faiss** in May 2025 and aborts rather than falling back | **switched to exhaustive matching** for ≤ 600 images — no index needed, and for a hand-held orbit that returns to its own start it cannot miss the loop closure. 300 images → 44,850 pairs |
| 4 | `Cannot use GPU feature matching without CUDA or OpenGL support` | **this pycolmap wheel is built without CUDA.** Forcing `device=Device.cuda` aborts the run | matcher runs on **CPU** with `num_threads = -1`. Acceptable: 24 cores. Note feature *extraction* did use the GPU — only the matcher is affected |

**Watch for:** the repo README records that COLMAP CPU SIFT *"silently reports '0 images' when
the OOM killer takes it"*. If the sparse model comes back empty, check `dmesg` before blaming
the capture.

---

## 8. Known-open issues

1. ~~**Smoke test not yet green.**~~ ✅ **RESOLVED — ALL CHECKS PASS** (2026-08-06 00:30).
   Final numbers: rest height max |dz| 3.0 mm, closest object pair **84.4 mm**, r ∈ [0.150,
   0.280], box clearance 50.5 mm, box drift **0.00 µm**, clutter drift **0.1 µm** over 400
   steps, and all 8 predicate controls pass. The 0.1 µm noise floor confirms `DISTURB_TOL`
   = 2 mm is calibrated with four orders of magnitude of headroom.

   It took **four** rounds, and the last two are worth reading because each fix was wrong in
   an instructive way:

   *Round 1 (8 fails).* The sampler put objects **inside the box**; physics ejected them to
   r = 0.51 m with 100 mm of vertical error and 62 mm of "drift". Cause: box azimuth drawn
   within ±30° of straight-ahead, and with the keep-out inflated by the tape roll's 45.5 mm
   radius the box covered most of the object sector — rejection sampling exhausted its tries
   and **silently kept the last rejected candidate**.

   *Round 2 (6 fails).* Box overlap **FIXED** — `objects stay out of the box footprint` now
   passes with 49.1 mm clearance, and all 8 predicate controls pass. But objects still overlap
   *each other* (closest pair 5.9 mm) and get ejected (r up to 0.405 m, tapemeasure drifting
   7.1 mm). Cause: the **fallback** path fanned objects by index without ever checking the
   spacing against their footprint radii.

   *Round 3 (fixed slots — WRONG).* Three fixed (radius, azimuth) slots, hand-verified
   pairwise clear. Still 5 fails. `scripts/diag_workstation_spawn.py` — written specifically
   to read positions **before physics runs**, so a bad spawn is not confused with a good spawn
   that something later disturbed — showed why in one pass:

   ```
   cube vs rolloftape  : min 100.6 mm (need 85.5)  violations   0/256
   cube vs tapemeasure : min   0.9 mm (need 88.0)  violations  44/256  <== OVERLAP
   fallbacks fired: {'tapemeasure': 68}
   ```

   **The fixed slots are only mutually clear if all three objects use them.** In practice the
   cube and tape roll place fine by rejection and only the *tape measure* falls back, so its
   fixed slot landed wherever the other two had been placed at random.

   *Round 4 (conflict-aware lattice — CORRECT).* The fallback now scores a 5 × 17 lattice of
   candidate positions by clearance to the box *and* to the already-placed objects, and takes
   the argmax — the best **available** position, not one clear of a hypothetical. Annulus also
   widened to r ∈ [0.15, 0.28]. Result: **0 violations at spawn**, fallback rate down from
   20–28 % to 9–13 %.

   **Generalisable lesson:** a fallback that guarantees a property *in isolation* guarantees
   nothing in a system where it is only sometimes used. And when a symptom is measured after
   physics, write the probe that reads the state *before* physics — theorising about which of
   the two it was cost more than the diagnostic did.
2. **Size DR is not implemented.** C8 (measured) says a post-build `xformOp:scale` never
   reaches the PhysX collider. The build-time route via `MultiAssetSpawnerCfg` is *also*
   uncertain here: `random_choice` is **deprecated** in this Isaac Lab build and heterogeneous
   assignment moved to `InteractiveSceneCfg.random_heterogeneous_cloning`. Deferred rather
   than faked. Mass and friction DR *are* wired and are honest.
3. **`o_align ≥ 0.99` gate is unachievable** at these grip heights — relax to ~0.90.
4. `clip_actions` in the copied rl_games cfg was **100.0** (the D1 bug). **Fixed to 1.0.**

---

## 9. What worked / what didn't

**Worked**
- Running the feasibility spike *before* writing the expert (LESSONS A10). It cost ~40 min and
  overturned the blocker the entire plan was organised around.
- **Built-in positive controls.** `control_24` failing is what exposed both bad searches. Two
  wrong "this cannot be grasped" conclusions were caught before they reached a doc.
- Writing the predicate's negative controls *first*. All 8 pass, including the rotated-box case
  an axis-aligned test would get wrong.
- Backgrounding everything as `systemd --user` units — frame extraction alone burned 58 min of
  CPU with zero GPU cost, fully overlapped with GPU work.
- The smoke test earning its keep immediately: the sampler bug would have silently capped every
  success number measured in the env.

**Didn't work**
- Copying search hyper-parameters between tasks (§3.3). Twice.
- `| grep | tail -N` in a systemd unit: output is buffered until EOF, so there is **no progress
  visibility**. Redirect to a log file instead.
- Substituting analytic colliders for reconstructed meshes without flagging it — the user
  caught it, correctly.
- Under-communicating the reconstruction-vs-training sequencing decision. It was stated once
  and not resurfaced when the user asked to start training.

**Corrected from the previous handoff**
- B1's premise (C9 blocks short-object grasps) is **wrong** — see §3.
- Decision 6.3 ("all three must be grasp targets") is **superseded**.

---

## 10. Plan from here — *subject to change*

> ### ⚠ SEQUENCING CORRECTED 2026-08-06 by the user
>
> *"training 3dgs and basically the re3sim pipeline is AFTER training the BC policy. THIS IS
> COMPLETELY WRONG!! we should be doing this as part of the setting up pickandplace env!"*
>
> The "reconstruction track (parallel)" below is **wrong** and is kept only so the change is
> visible. Reconstruction is **environment setup** and precedes demo generation, because a
> policy trained against primitive colliders is not a draft of the policy you want — it is a
> measurement of a different task, and every number has to be re-taken after the swap.
> The corrected pipeline, the measured alignment results, the object scale gate and the six
> `reconstruct.py` fixes are all in **[02_RECONSTRUCTION.md](02_RECONSTRUCTION.md)**.

### THE TWO THINGS TO DO NEXT (2026-08-06, ~03:00)

**1. Finish the transit — [03_EXPERT_AND_BC.md §3b](03_EXPERT_AND_BC.md).** This is the only
thing between here and a policy that runs from a cold `env.reset()`. Eight variants are
measured, the root cause is identified and confirmed (a position drive cannot hold a
torque-demanding pose without an error; `cem` evaluates candidates kinematically and cannot
see it), and the fix — a plan-time bias measured per pose with `hold_phys` — is implemented
behind `SETTLE_BIAS=1`. It makes real contact (`gap` −1.2 mm on air → 14.3 mm on the cube) but
does not yet raise success. Start there; do not re-derive it.

**2. Re-capture the four objects.** Not a code problem: the cube is absent from both its
sparse *and* its dense reconstruction ([02_RECONSTRUCTION.md §3.1c](02_RECONSTRUCTION.md)).
The scene capture is excellent, so this is a per-object re-shoot — matte surface, slower, less
background texture — not a redo of the session. Everything downstream of it is built and
gated.

### Immediate — the corrected order

Reconstruction first, because demonstrations generated against primitive colliders measure a
different task and would all have to be regenerated after the swap.

```bash
# 1. objects: dense clouds, then metric meshes + rigid USDs (both already running detached)
bash Re3Sim/workstation/tools/recon_all_objects.sh     # COLMAP + MVS, sequential
bash Re3Sim/workstation/tools/finish_all_objects.sh    # watches for each dense cloud

# 2. scene: crop the trained splats to the desk and emit the NuRec USD
bash Re3Sim/workstation/tools/finish_scene_splats.sh

# 3. re-run the env smoke test AGAINST THE RECONSTRUCTED COLLIDERS
cd eva_rl && python -u scripts/test_workstation_env.py --headless --num_envs 64

# 4. re-run the B1 grasp gate on the reconstructed cube -- the real acceptance test.
#    It should reproduce 99 % at grip z = 25 mm. If it does not, the mesh or the origin
#    convention is wrong and every demo built on it is worthless.
cd ../eva_bc && python -u re3sim/probes/p01_grasp_feasibility.py --object rubixcube --headless

# 5. expert smoke: read `planning: k/n`, then `SUCCESS k/n`, then the taxonomy
python -u re3sim/expert/collect_demos.py --headless --num_envs 32 --batches 1 --chatty

# 6. demos -> BC -> eval, one command
TAG=recon BATCHES=9 bash re3sim/run_bc.sh
```

**Gate before step 6: expert success ≳ 55 % overall.** If planning solves but execution fails,
the taxonomy says which leg:

- `no_lift` / `never-got-there` → the grasp pose or the approach;
- `lifted-but-lost` → `CARRY_Z` too low, or the finger drive (check `FINGER_STIFFNESS = 2000`
  is actually applied);
- `over-box-not-inside` → release height or the settle window.

Still outstanding: record the B1 result into `01_STEP1_PLAN.md` and `CHALLENGE_SUITE.md` C9 as
a **dated correction block**, not a silent edit (LESSONS E4).

### S1′ — expert (`eva_bc/re3sim/expert/`)
3. Port `clutter/expert/clutter_expert.py`'s structure onto `ArmKin`. Legs:
   **approach → grasp at z = 25 mm → lift → carry → release over the box**.
   - gate each solve on `pos_err ≤ 1.5 mm`, `o_align ≥ 0.90`, `low_z ≥ 2 mm`, retry `tries`×;
   - **finish every search before closing the gripper** (A6 — worth 0 %→100 % once already);
   - the release must clear the 93 mm rim: TCP above ~93 + cube/2 + margin ≈ 140 mm;
   - keep-out boxes around the clutter and the box walls via `cem(avoid=…)`.
4. **Measure expert success per spawn seed on a fixed suite**, with a failure anatomy. Do not
   proceed under ~70 %.

### S2′ — demos → BC
5. Batch demo generation to HDF5 with per-segment phase labels and a `train_mask` censoring
   the expert's own failed sub-attempts (port `clutter/act/collect_demos.py`).
6. Flow-matching chunk BC (`clutter/act/train_flow.py`): chunk 50, execute 15, temporal
   ensembling OFF, 10 Euler steps. **Do not shorten the execution horizon** (C2:
   59.4 → 32.8 → 3.1 → 0 % at 15/8/4/2).
7. Batched sim eval with per-env tensor action queues. **≥3 seeds**, champion on a *held-out*
   spawn seed, pooled ≥128 episodes. Single-run comparisons are void (C1).

### S3′ — RL and vision
8. Grasp-success bit probe (5 raw dims, 0 % FPR gate), then **x0-steering** on the frozen BC
   base — additive residual measured *exactly flat* elsewhere (D2); keep it only as an ablation.
9. Swap the reconstructed meshes and splats in (§6), then teacher→student vision distillation.

### Reconstruction track (parallel, GPU-serialised)
10. `recon-scene` → `align_to_marker.py --marker-size 0.165` → `fit_table.py` →
    `build_workstation.sh`. **Gate:** alignment RMS vs the 2.8 mm achieved last run; fitted
    surface z within ~1 mm of −0.0065.
11. Per-object: `reconstruct.py` → `extract_object.py --target-height` (from
    `measurements.txt`) → `mesh_to_rigid_usd.py`. **Gate:** reconstructed aspect ratio vs the
    measured one — the previous cube came out 50.4 mm against a true 57 mm because
    `--plane-thresh` ate its base.

---

## 11. Machine notes

- `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6`
- **Never invoke Isaac Sim through the env's python binary directly** — segfaults on a
  `CXXABI_1.3.15` mismatch ~1 s into Kit startup. Always go through activation.
- 10 GB RTX 3080, shared. `nvidia-smi` first. A 64-env PLAY smoke test is ~2 GB; the p01 probe
  ~1.5 GB; 3DGS will want 4–6 GB. Two small jobs coexist; 3DGS + training will not.
- **Long jobs: use the Bash tool's `run_in_background: true`, NOT `systemd-run --user`.**
  User instruction, 2026-08-06: *"harness background is the way to go … in the future"*.
  Harness jobs re-invoke the agent on completion; `systemd --user` units send **no
  notification** and have to be polled. The user works from **tmux**, so the Claude session
  survives their disconnects and systemd's survive-logout advantage is not needed.
  The units live at handoff time were started before this instruction and should be left
  running; new work goes to harness background jobs. (Hybrid, if a job ever needs both:
  run it under systemd and start a harness job that blocks on `systemctl --user is-active`
  purely to get the wake-up.)
- **Redirect to a log file.** A `| grep | tail -N` inside a background job buffers everything
  until EOF, so there is zero progress visibility — this cost real time twice.
- `python -u` always.
- OpenMVS v2.1.0: binaries exit 1 on `--help` by design; `ReconstructMesh -p` does not exist.

### Live units at handoff time

```bash
systemctl --user is-active recon-scene
tail -f /home/eva/Desktop/isaacLab/Re3Sim/data/scene2/recon.log
```

| unit | doing | notes |
|---|---|---|
| `recon-scene` | exhaustive CPU matching on `data/scene2`, then incremental mapping → 3DGS 30 k → OpenMVS | hours. Resumable via `data/scene2/progress.json` |
| `p01-b1` | finishing the remaining B1 rows (tapemeasure, rolloftape) | **informational only** — the cube is the only target now |

### Commands to resume

```bash
cd /home/eva/Desktop/isaacLab/eva_rl
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6

# 1. the immediate next action -- is the sampler fixed?
python -u scripts/test_workstation_env.py --headless --num_envs 64

# 2. re-run the B1 gate for any object (control_24 is the built-in positive control)
cd /home/eva/Desktop/isaacLab/eva_bc
python -u re3sim/probes/p01_grasp_feasibility.py --object rubixcube --headless

# 3. reconstruct an object capture once the scene finishes (GPU is serialised)
cd /home/eva/Desktop/isaacLab/Re3Sim
python -u re3sim/scripts/reconstruct.py -i data/object2_rubixcube
```
