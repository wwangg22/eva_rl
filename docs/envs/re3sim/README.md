# re3sim — photoreal reconstructed workstation

Environments built from a **real** capture of the user's desk rather than from procedural
primitives. The reconstruction half (photos → 3DGS + meshes → metrically-aligned USD) uses
the `Re3Sim` fork at `~/Desktop/isaacLab/Re3Sim`; the RL half lives here and follows the
same conventions as the rest of `docs/envs/`.

| doc | covers | status |
|---|---|---|
| [HANDOFF.md](HANDOFF.md) | **start here** — full state, decisions, what was learned, next actions | live |
| [02_RECONSTRUCTION.md](02_RECONSTRUCTION.md) | **the Re³Sim pipeline as env setup** — measured alignment, the object scale gate, the six `reconstruct.py` fixes | live |
| [03_EXPERT_AND_BC.md](03_EXPERT_AND_BC.md) | the scripted expert, its cost-function fix, and the flow-BC stack | live |
| [04_AUTHORED_CUBE_AND_RECOVERY_EXPERT.md](04_AUTHORED_CUBE_AND_RECOVERY_EXPERT.md) | ⭐ the authored Rubix cube (and its per-env patterns), the front camera, and the goalset/hold-check/regrasp expert | live |
| [05_VISUAL_DR.md](05_VISUAL_DR.md) | the -VisionDR tasks: every DR axis, every gate, every catch; machine limits; collection runbook | live |
| [06_VISION_POLICY.md](06_VISION_POLICY.md) | the pixels-only student rounds 1–2 — where it actually breaks (grasp precision, wrist blackout) | live |
| [07_CUROBO_EXPERT.md](07_CUROBO_EXPERT.md) | the pick-and-place cuRobo expert ported to the workstation: diagnosis ledger + measured state | live |
| [01_STEP1_PLAN.md](01_STEP1_PLAN.md) | Step 1 — a good randomised env to work with | **planning** |
| [pickandplace1/](pickandplace1/) | Step 2+ — expert → BC → eval → x0-steering → vision | **planning** |

**Gym IDs:** `Rebot-Workstation-PickPlace1-v0`, `-Play-v0`, `-Strict-v0`
**Env code:** `source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/`
**Expert + BC:** `~/Desktop/isaacLab/eva_bc/re3sim/`
**Reconstruction tools:** `~/Desktop/isaacLab/Re3Sim/workstation/tools/`
**Capture:** `~/Desktop/isaacLab/data/captures/2026-08-05/`
**Latest run:** `eva_bc/re3sim/runs/prim/` (demos, checkpoints, both eval JSONs)

## Progress log

Newest first. One entry per session; link out to a numbered doc when an entry needs more
than a paragraph.

### 2026-08-06 (evening) — a policy runs from a cold reset on the photoreal env

**The transit is solved and it was never the drive.** Nine variants and a day of gravity/drive
hypotheses; the answer was `reversed()` applied to both halves of the Cartesian chain, which
put the transit's endpoint in the *middle* of its own waypoint list. Caught by reporting the
achieved TCP against **two** references at once — the Cartesian target (101.6 mm) and the FK of
the pose actually commanded (**7.5 mm**). The arm was going exactly where it was sent.
End-of-transit error **101.6 → 1.0 mm**. Full account in
[03_EXPERT_AND_BC.md §4c](03_EXPERT_AND_BC.md).

**That exposed a real ~35 mm descent sag, and it is compensable.** Re-swept end-to-end,
`GRIP_Z = 56 mm` — *above* the cube's own 54.5 mm top, because the **settled** tool centre is
what lands on it. Both earlier values were honest measurements of the wrong thing: 25 mm on an
*analytic* cube, 32 mm under a protocol that skipped the descent.

**Results, all from the environment's own `reset()` — no teleport shortcut:**

| | value |
|---|---|
| expert | **30.6 – 34.2 %**, 84–98 % of layouts planned |
| BC (2048 demos / 100 k steps) | **20.1 %**, retains **65 %** of the expert |
| clutter displacement, median over successes | **0.04 mm** |

`success` tracks `lifted` to within ~2 points on every seed, so the carry and the place are
learned and the entire remaining gap is the grasp. Scaling is steep and not yet flat — 2.6× on
2× the data, with no architecture change.

**Two more measurement bugs, same family as the morning's:** the episode horizon did not move
when the manoeuvre doubled in length (demos 1018–1118 steps against an 800-step horizon —
3.4 % → 7.8 % once fixed), and both bias-compensation mechanisms made things *worse* and are
off by default.

**Video:** `re3sim/act/record_video.py` films the policy from the user's own validated wrist
mount (`WRIST_CAM_CFG`, D405, 84° HFOV, `tilt_x_m30`) and a workstation view, driven through
the same `ChunkController` the evaluation uses.

### 2026-08-06 (morning) — the photoreal env is real, and three of my own gates were lying

**The overnight conclusion that the objects were absent from their captures was WRONG.** It
was a sign error: the support plane's up-direction was decided from the *height distribution*
rather than from the *cameras*. With a room in frame the far field owns the larger tail, the
normal flips to point into the table, and everything above the desk reads as below it. All
four objects were there the whole time. Corrected in
[02_RECONSTRUCTION.md §3.1c](02_RECONSTRUCTION.md), which keeps the wrong reasoning alongside
the right one.

**The photoreal environment is now built and verified by eye, not just by gate:**

| piece | state |
|---|---|
| desk appearance | 356,768 NuRec gaussians, PSNR 28.13 |
| desk collider | analytic slab fitted to the reconstruction, top face at z = 0 |
| **cube** | **reconstructed mesh**, 53.3 × 58.0 × 54.5 mm vs 56³ measured (+4.8 %) |
| **tape measure** | **reconstructed mesh**, 73.1 × 69.6 × 33.4 mm vs 71.5 × ? × 36 (+7.1 %) |
| **roll of tape** | **reconstructed mesh**, 83.1 × 85.1 × 23.0 mm vs 91 × 91 × 24 (+8.7 %) |
| box | authored open container (permanent, B2). Its reconstruction is a metrology check |
| smoke test | **ALL CHECKS PASSED** with the reconstructed colliders loaded |

**A yaw bug that no automated check could ever have caught.** The marker→env transform applied
height but not rotation, so the reconstructed desk sat *beside* the workspace: measured, only
**14 %** of the object-spawn annulus had desk under it. Every object spawned over bare ground.
Because splats are visual-only, every predicate, smoke test and success rate was bit-identical
either way — it was found by rendering the scene and looking at it. Now **91 %** (§3.1f).
`scripts/render_workstation.py` exists so this is checkable in one command.

**Three separate bugs in gates I had written**, all of the same shape — a check that looks
rigorous, prints a confident number, and is not testing what it claims:

* §3.1a the held-back dimension was the *same axis* used to set the scale, so it passed a
  28.0 × 22.9 × 56.3 mm "56 mm cube";
* §3.1c-bis the gate reported *after* writing the USD, so a FAIL still shipped the asset;
* §3.1g when one axis is unknown it compared the wrong pair, failing a **good** tape-measure
  mesh at "+91 %" that is really +6.6 %.

**Acceptance test on the swap: the expert falls from ~31 % to 19.5 %.** That is the
reconstructed cube being genuinely harder to grasp than a perfect box — rounded corners, a
convex-hull collider. `lifted-but-lost` remains **0**, so the carry is unaffected; it is the
grasp. `GRIP_Z = 25 mm` was measured against the *analytic* cube and is therefore now an
assumption, so it is being re-swept (`re3sim/probes/sweep_grip_z.sh`).

**Open for the user:** the box's second outer dimension. The reconstruction stably reports
229 × 261 × 90 mm across ten parameter settings against a sheet value of 218 "longest", which
suggests 218 is the *width* and the real box is ≈ 248 × 218 × 93. The env currently authors
218 × 150, where 150 has always been flagged as a guess.

### 2026-08-06 (later) — sequencing corrected: reconstruction **is** env setup

The user's call, and it is right: *"we should be doing this as part of the setting up
pickandplace env! Setting up pickandplace env entails doing all of the re3sim pipeline NO
CUTTING CORNERS!"* A policy trained on primitive colliders measures a different task, so every
number would have to be re-taken after the swap — doing reconstruction first is cheaper, not
just more correct. Full write-up in **[02_RECONSTRUCTION.md](02_RECONSTRUCTION.md)**.

**Scene alignment PASSED, and beats the previous capture:** 262/300 frames see marker id 42,
triangulation RMS **15.20 mm** (was 16.7), median orientation spread **0.66°** (was 0.91°),
scale 0.297717 m/COLMAP-unit. The marker side is **measured** here (165 mm on a ruler) rather
than inferred from iPad screen DPI, which removes the previous scene's largest systematic
error. Independent check: the fitted long table edge comes back at **1.642 m**, matching the
previous capture to three decimals across two different marker sizes and two scale solves.

**The desk-height offset needed three independent estimates and none of them agreed** — 6.5 mm
from the calipers, 0.6 mm from the plane fit, −4.9 mm from an annulus that excludes the iPad.
The annulus reading came with a **26.5 mm interquartile range**, which is the answer: those
points are not one surface, they are the cables and tape and desk edge visible in the capture.
The sparse reconstruction cannot constrain this number, so the offset used is the caliper
measurement — a direct reading of the physical stack-up rather than an estimator over a
contaminated point set (§2.5).

**Expert ran for the first time** and the taxonomy is clean: 17/32 planned, of which 10
succeeded, with **zero** carry or release failures. The bottleneck is the planner's `o_align`
gate, and the cause is a cost imbalance — `_kin.cem`'s stock `w_o = 0.25` lets the search buy
a fifth of a millimetre with the whole opening axis. Fixed by re-pricing (`w_o` 0.25 → 4.0,
`w_pos` 200 → 600 to keep the hinge balanced), adding the DLS `refine` the first version never
called, and searching **both** valid cube opening axes and both grip heights.

**Two more `reconstruct.py` breaks found** (six total), both of which fire *after* COLMAP,
3DGS and MVS have all finished — the most expensive possible place: the `colmap` CLI and
OpenMVS binaries live in a different conda env from `pycolmap`, and `ReconstructMesh -p` does
not exist in v2.1.0 and was unchecked, so a missing mesh flowed on and the stage was marked
complete anyway.

**The desk is now reconstructed and in the env.** 3DGS finished at **PSNR 28.13 / L1 0.020**;
356,768 gaussians cropped to the tabletop and emitted as a NuRec `splats.usd` placed in the
env frame, plus an invisible analytic collider slab sized from `fit_table.py`. Env smoke test
**ALL CHECKS PASSED** — and the slab is a genuine improvement on the stock table it replaced:
objects settle at *exactly* their rest height (max |dz| **0.00 mm**, was 3.00 mm).

**The movable objects are blocked on capture quality, not on code.** The cube's dense cloud
reconstructs 48.5 M points but **the cube is not in it** — nothing compact rises clear of the
support plane, the tallest coherent cluster being a flat 3.9 × 3.0 × 0.47 patch of the
newspaper bed (§3.1c). Two real pipeline bugs were found and fixed on the way there, and both
are worth knowing:

* **the scale gate could not fire.** It held back the *longest axis* — which for a cube is the
  same axis used to set the scale, so it is correct by construction. It duly passed a mesh
  measuring **28.0 × 22.9 × 56.3 mm** as a 56 mm cube. A held-back check is only a check if it
  is independent of the thing being fitted.
* **no ROI.** The seed height was a fraction of the tallest thing in a whole-room cloud, which
  put the cut *above* the cube entirely. Now located from the camera geometry — an orbit's
  cameras ring the object, so their centroid projected onto the support plane finds it with no
  appearance assumption, which matters for a deliberately multi-coloured cube.

**A policy exists and the BC stack is healthy.** 896 demos (278 successful, 31.0 %) -> 60 k
steps of flow-matching BC -> evaluated on three held-out spawn seeds, on both protocols:

| | matched protocol (starts where the demos start) | cold `env.reset()` |
|---|---|---|
| pooled success | **21.9 %** (84/384) | **0.0 %** (0/384) |
| vs the 32.4 % expert | **retains 68 % -- ON TREND** | -- |

68 % retention of a state-observation expert is squarely normal, and the three seeds agree to
+-2 points. The second column is the transit problem priced exactly: it is the whole cost of
demonstrations that begin somewhere `env.reset()` never puts the arm. A caveat the headline
hides: among matched-protocol successes the worst clutter body has moved a **median 33.9 mm**,
and requiring it to stay within 10 mm drops success to 7.3 % -- about two thirds of the
successes involve some bulldozing.

**The expert's transit is the open problem, and its cause is now known.** Eight variants
measured (03 §3b). The manoeuvre *after* the pre-grasp works in every one of them —
`lifted-but-lost` and `over-box-not-inside` are **zero** throughout. What fails is reaching the
grasp pose: a position drive holds a torque-demanding pose only *with* an error, and `cem`
cannot see that because it evaluates candidates kinematically. Confirmed by raising the
compensator clamp — the grasp error falls 73.8 → 33.5 mm and the closed error 60.2 → 9.1 mm,
but the compensator saturates at both clamps tried and at the larger one drives the cube into
the desk. The fix (a plan-time bias, measured per pose with `hold_phys`) is implemented and
specified but does not yet pay; it is off by default behind `SETTLE_BIAS=1`.

### 2026-08-06 — B1 overturned, env built, reconstruction running

**The blocker the whole plan was organised around does not exist.** C9's "44 mm TCP floor" was
measured with the *fingers shut*; a grasp approaches with them open and the finger bodies reach
z ≈ 5 mm. Measured: the rubix cube grasps at **99 % with the TCP at 25 mm**. Full table and the
two probe bugs that produced two earlier wrong answers are in [HANDOFF.md](HANDOFF.md) §3.

**Task narrowed by the user:** only the cube is a grasp target; the roll of tape (91 mm across
vs an 89.1 mm opening) and the tape measure (184 g, at C2's failure cell) are **clutter**.

**Built:** `Rebot-Workstation-PickPlace1-v0` / `-Play-v0` / `-Strict-v0`, 41-D obs, authored
open box, strengthened `placed_mask` (passes all 8 negative controls including a rotated-box
case an axis-aligned test would fail). Expert scaffolding in `eva_bc/re3sim/` with `_kin.py`
ported.

**Env smoke test: ALL CHECKS PASS.** Took four rounds of spawn-sampler fixes; the last two are
documented in HANDOFF §8 because each wrong fix was instructive.

**Expert written, never run.** `eva_bc/re3sim/expert/{workstation_expert,collect_demos}.py`.
Running it is the next command — see HANDOFF §10.

**Reconstruction running:** all 5 captures framed (scene marker gate **262/300 = 87 %**),
scene COLMAP past matching and into incremental mapping. Four pycolmap API breaks patched
along the way (HANDOFF §7).

**Still placeholder:** object colliders are primitives, not reconstructed meshes; the desk
visual is the stock Isaac table, not the splats. Both are swaps, not rebuilds — see HANDOFF §6.

### 2026-08-05 — plan written, nothing built yet

Capture uploaded (5 videos + measurements). Read the repo's measured hardware envelope
(`docs/CHALLENGE_SUITE.md`) against the four captured objects and found **two blockers that
change the task design**, both recorded in [01_STEP1_PLAN.md](01_STEP1_PLAN.md):

1. **C9 (TCP floor ~44 mm) rules out 2 of the 3 intended grasp targets** at table height —
   the tape roll is 24 mm tall and the tape measure 36 mm, both entirely below the floor.
2. **The cardboard box reconstructs as a solid, not a container** — `extract_object.py`
   closes the mesh with a synthetic base, so nothing can be placed *into* it.

B2 was decided (author an open box, reconstructed mesh as visual only). **B1 is still open**
and the user has ruled that all three objects must be graspable, so it has to be solved rather
than designed around — most likely with a real riser, which may need one more short capture.

Also recorded: **size domain randomisation cannot be done at runtime** (C8 — a post-build
`xformOp:scale` never reaches the PhysX collider), so it becomes a build-time variant choice.

Step 2+ planned in [pickandplace1/](pickandplace1/), with the inherited-mistakes list in
[pickandplace1/LESSONS_INHERITED.md](pickandplace1/LESSONS_INHERITED.md). Full state in
[HANDOFF.md](HANDOFF.md).
