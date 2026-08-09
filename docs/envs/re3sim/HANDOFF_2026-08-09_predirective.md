# HANDOFF — re3sim photoreal workstation
*Written 2026-08-09, end of the 08-06→08-09 session. Supersedes `HANDOFF_2026-08-06.md`
(kept) and `HANDOFF_prev.md` (kept).*

---

## 0. ⭐ THE NEW DIRECTIVE — read this first, it changes the plan

The user issued a **course correction** at the end of this session. It overrides parts of the
plan in §8 of the previous handoff:

1. **Ditch the reconstructed gaussian/scanned OBJECTS.** Import a *proper* Rubix cube asset
   instead. Reconstruction stays for the **workstation/desk backdrop only**.
2. **Use an expert very similar to the ORIGINAL pick-and-place expert**, which achieved a
   really high success rate. See §6.1 — it is `eva_bc/expert/run_expert_v1.py`, a **cuRobo**
   planning expert, and **cuRobo is NOT installed on this box** (§6.2). That is the first
   thing to resolve.
3. **Move the workstation camera to be IN FRONT of the arm.** It is currently behind it
   (`eva_rl/scripts/render_workstation.py::VIEWS` and the `station_cam` pose in
   `collect_demos.py`, both at `(-0.62, -0.52, 0.46)` looking at `(0.24, 0.02, 0.04)` — that
   is over-the-shoulder from behind).
4. **Verify the expert exceeds 90 % success** before proceeding.
5. **Then push the env to main**, *ensuring no shared code files change*.

**Nothing in item 1–5 has been started.** The session ended immediately after this directive
was given; the remainder of this document is the state it must be started from.

---

## 1. State in one table

| thing | state |
|---|---|
| expert (CEM-based, current) | **71.9 %** at 64 envs, up from 40.6 % — §4.1 |
| BC policy | **19.5–20.1 %**, but trained on a BROKEN expert — must be redone, §4.3 |
| object meshes | rebuilt, gated, solid. Cube 40 holes → 2. §4.4. **To be replaced per the directive** |
| desk gaussians | healthy statistically, good from above, poor at grazing. Shipped unpruned. §4.5 |
| object rendering | ⭐ **WASHED OUT — open bug**, 2.6× saturation loss, cause narrowed but unfixed. §5.1 |
| videos | BC policy + expert, wrist and workstation views, all in `eva_bc/re3sim/runs/video/` |

---

## 2. Layout (unchanged)

```
eva_rl/source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/   env cfg + mdp
eva_rl/source/reBot_RL/data/workstation/                      objects/*.usda, splats.usd
eva_rl/scripts/render_workstation.py                          4-viewpoint renderer (num_envs=1!)
eva_bc/re3sim/expert/{workstation_expert,collect_demos}.py    CEM expert + executor
eva_bc/re3sim/act/{policy_runner,eval_flow,record_video}.py    BC
eva_bc/expert/run_expert_v1.py                                ⭐ THE "OG" EXPERT (cuRobo)
Re3Sim/workstation/tools/                                     reconstruction pipeline
Re3Sim/data/object2_*/  object3_*/                            two capture generations
Re3Sim/data/uploads/                                          where the user drops new videos
```

---

## 3. ⭐ WHAT WORKED — the two big wins of this session

### 3.1 The expert went 40.6 % → 71.9 %, and *why* is the transferable part

Two independent bugs, found by measurement, not by reasoning:

**(a) The gripper closed before the arm arrived.** The transit leg had a 40-step settle; the
grasp descent had **none**, and went straight from its ramp into the close. Measured with
`collect_demos.py --chatty`:

| leg | TCP error vs its own plan |
|---|---|
| `home` (has a settle) | **1.1 mm** median, 1.7 p90 |
| `grasp` (had none) | **11.1 mm** median, **60.0 mm** p90 |

The gripper opens to 89 mm on a 53–58 mm cube — ~17 mm margin a side. Adding `GRASP_SETTLE`
(default 40 env steps) collapsed the error to **0.9 mm** and took success **40.6 → 54.7 %**.
It saturates by 20 steps: this is tracking lag being allowed to decay, nothing subtler.

**(b) Grasp poses were RANKED by statistics that do not predict success.** `cem` scores
candidates through `write_joint_state_to_sim` — kinematically, no gravity, no contact — so it
returns poses the drive cannot hold. Three hypotheses were tested and **all refuted** from
existing data, no GPU needed:

| hypothesis | test | result |
|---|---|---|
| arm sags more when extended | success vs cube radius, 6 sextiles 150–280 mm | 33.0/34.1/31.1/32.5/35.3/37.2 % — flat |
| some envs got slippery cubes | mass+friction are randomised at **startup** → fixed per `env_index` across all 16 batches | per-env spread **0.99×** the binomial null. 0 envs at 0 %, 0 at ≥80 % |
| bad opening axis | success vs cube yaw mod 90°, 6 bins | 32.6/33.4/35.7/32.2/34.5/35.0 % — flat |

The clutter expert (94 % on its own task) had already written this down:
> *"no forward-kinematic statistic tried so far predicts it… So stop guessing: run each
> candidate's close in the sim and keep the one that measurably works."*

Porting that **in-sim close screen** (`screen()` in `collect_demos.py`, behind `SCREEN_ROUNDS`)
took **54.7 → 71.9 %** and planning 59/64 → 64/64. Its trace:
`54/64 lift → 63/64 after re-drawing 10 → 64/64`. `SCREEN_ROUNDS=2` and `=4` tie.

**Residual, stated honestly:** the screen certifies 64/64 grasps lift but only 46/64 succeed
end-to-end, because it *teleports* to the grasp while the real run *descends* 140 mm onto it.
Screening through the actual descent is the obvious next refinement.

### 3.2 Four object-pipeline bugs fixed, and a gate that can catch them

The user reported the objects looked "hollow" with wrong colours. Measurement confirmed far
worse: **the cube was a Rubix-cube-shaped LATTICE** — 40 distinct holes, the centre of every
sticker punched out, convex hull filling only 0.674 of its own bounding box.

Cause: MVS cannot match flat glossy sticker faces (a specular highlight moves between frames,
so the same physical point looks different in every one) while the high-contrast black grooves
match perfectly. `extract_object.py --density-quantile 0.06` then deleted exactly those
unmatched regions — and **`finish_object_asset.py` never passed the flag**, so every object
ever built used that default.

| bug | fix |
|---|---|
| density trim ate the unmatched faces; flag never passed through | `--density-quantile` passed, and **per-object** (cube 0.01, others 0.06 — one global value was wrong in *both* directions: at 0.01 the tape roll inflates 91 → 127 mm) |
| holes never filled | `--fill-holes` (30 mm) **after** decimation (decimating a filled mesh re-opens it), with nearest-original-vertex colour transfer so filled stickers are not black |
| `subdivisionScheme` never authored → USD fallback `catmullClark` | set to `none`. It also silently voided the authored normals, which USD ignores on subdivs |
| no material bound → RTX ignores a bare `displayColor` | binds `UsdPreviewSurface` + `UsdPrimvarReader_float3` |
| report crashed on ndarray **immediately before printing the gate verdict** | `json.dump(default=...)` — this is why no object ever reported PASS/FAIL |

**The gate that let the lattice through measured extents and mass.** *A hollow shell has
exactly the same bounding box as the solid it should be* — the lattice passed at **+4.8 %**, a
BETTER score than the fixed mesh gets. `finish_object_asset.py` now also reports
`holes / boundary% / hull-bbox fill` and warns above 5 holes. It fired correctly on its first
real use. `Re3Sim/workstation/tools/inspect_object_mesh.py` is the standalone version.

---

## 4. Measured results

### 4.1 Expert (64 envs, seed 11, paired, end-to-end, no teleport)

| configuration | success | planned |
|---|---|---|
| shipped | 40.6 % | 59/64 |
| `GRASP_SETTLE=40` | 54.7 % | 59/64 |
| **`+ SCREEN_ROUNDS=2`** | **71.9 %** | **64/64** |

Failure taxonomy across all 16 batches of the 2048-episode collection, **every batch**:
`plan-failed 8-21 | never-got-there 70-87 | lifted-but-lost 0 | over-box-not-inside 0`.
Once the cube leaves the table the expert places it **100 %** of the time. All loss is the grasp.

### 4.2 Grip-height sweep (still valid, end-to-end)

| grip z [mm] | 32 | 40 | 48 | **56** | 64 |
|---|---|---|---|---|---|
| success | 0.0 % | 7.8 % | 25.0 % | **40.6 %** | 40.6 % |

### 4.3 BC — MUST BE REDONE

| | 1024 eps | 2048 eps | 4096 eps |
|---|---|---|---|
| BC from cold reset | 7.8 % | **20.1 %** | 19.5 % |
| retention of expert | 24 % | 65 % | 63 % |

Curve is flat at ~20 %, **but every one of these was trained on demonstrations from an expert
that closed its gripper while the arm was still moving.** At ~64 % retention a 72 % expert
projects to ~45 %. The whole curve needs redoing.

### 4.4 Final object assets (before the directive to replace them)

| asset | source | worst axis | holes | fill |
|---|---|---|---|---|
| `cube.usda` | object2 | +7.3 % | **2** (was 40) | 0.657 |
| `rolloftape.usda` | object2 | **+4.1 %** | 6 | 0.550 |
| `tapemeasure.usda` | object3 | **+5.3 %** | 26 | 0.580 |

Backups of all three object2-derived versions in `data/workstation/objects_object2_backup/`.

### 4.5 Desk gaussians

356,768 gaussians, SH degree 3, median scale 1.13 mm, 0.27 % oversized, opacity median 0.51.
**Good from above** (top-down 3.25 % desk speckle), **poor at grazing** (floater curtain).
`prune_splats.py` was written and **deliberately not shipped**: removing the 9.3 % below-table
gaussians fixes grazing but degrades top-down 3.25 % → 8.00 %, because **gaussians are
volumetric, not surface samples** — one 30 mm below the desk still contributes from above.
`WRIST_CAM_CFG` looks steeply down, i.e. the regime where unpruned is best.

---

## 5. ⭐ OPEN BUGS

### 5.1 Objects render 2.6× desaturated — CAUSE NARROWED, NOT FIXED

| | saturation |
|---|---|
| cube mesh vertex colours | median **172**/255, p90 224, **75 %** vivid |
| rendered pixels at wrist range | median **67**/255, **6.2 %** vivid |

Excluded by measurement:
* **Not exposure** — rendered value mean 185/255, **0.0 % of pixels clipped**.
* **Not the mesh** — authored `displayColor` is vivid.
* **Probably not double-gamma** — pre-linearising (mesh S 172 → 232) moved rendered median
  only **69 → 71** (vivid 2.6 % → 11.3 %). Reverted, not shipped.

Remaining suspect: RTX may not honour `UsdPrimvarReader_float3 → UsdPreviewSurface.diffuseColor`
at full strength. Next: OmniPBR/MDL material, or bake vertex colours to a texture.
**Acceptance test: the mesh-vs-rendered saturation comparison that localised it.**
*Note: the directive to import a proper Rubix cube asset may make this moot for the cube — but
it will still affect the tape measure, the tape roll and any other scanned asset.*

### 5.2 Needs the user

* **box second outer dimension** — reconstruction says 229 × 261 × 90 vs sheet `longest: 218`.
  The env authors 218 × **150** and that 150 was always a guess.
* **arm mounting yaw** — 180° applied (91.3 % coverage); measured argmax 135° (97.1 %).

---

## 6. ⭐ THE "OG" EXPERT — what it is and what it needs

### 6.1 `eva_bc/expert/run_expert_v1.py`

A **cuRoboV2 planning expert** driving `Rebot-PickPlace-Play-v1`. Its per-episode loop, from
its own docstring:

```
for each can (nearest first):
  plan_grasp (goalset from proven-table candidates, azimuth-rotation construction)
  -> execute approach/descent at 50 Hz
  -> close + CLEAN-GRASP CHECK
  -> lift + HOLD CHECK (regrasp once on failure)
  -> attach -> plan transport to basket goalset -> release -> VERIFY PLACED -> home
```

**Why it is stronger than the current CEM expert**, and this is the design to copy:
* **cuRobo motion planning**, not CEM + hand-densified Cartesian waypoints.
* **Goalsets** — many candidate grasps handed to the planner, which picks a reachable one,
  instead of committing to one CEM solution and hoping.
* **Explicit verification at every stage** — clean-grasp check, hold check, verify-placed —
  with **regrasp once on failure**. The current expert has no recovery at all.
* Rich labelling for BC already built in (`segments`, `outcomes`, `train_mask` rules,
  perturb/diversify modes).

Sibling experts and their measured rates, for calibration: **clutter ~94 %**, **slot 91.4 %
pooled**. The 90 % bar in the directive is consistent with these.

### 6.2 ⭐ BLOCKER: cuRobo is NOT installed

```
env_isaaclab6: ModuleNotFoundError: No module named 'curobo'
tools:         ModuleNotFoundError: No module named 'curobo'
(no curobo directory anywhere on the box)
```

So "use a very similar expert" means one of:
* **(A) Install cuRobo** into `env_isaaclab6` and port `run_expert_v1.py` to the workstation
  task. Highest fidelity to the directive; cuRobo needs a CUDA toolchain (`~/miniconda3/envs/cudatk`
  has nvcc 12.8) and is a non-trivial build against torch 2.11.0+cu128.
* **(B) Port the ARCHITECTURE onto the existing `ArmKin`** — goalset grasps, clean-grasp
  check, hold check, regrasp-once, verify-placed — without cuRobo. The in-sim close screen
  already implemented (§3.1b) is effectively the "clean-grasp check" and is worth ~17 points
  on its own, which suggests the rest of the recovery machinery is where the remaining ~20
  points live.
* **(C) Both** — (B) first since it is unblocked and cheap, then (A) if (B) plateaus below 90 %.

**Recommendation: (B) first.** It needs no new dependency, reuses the screen, and directly
targets the measured failure (`never-got-there`, i.e. the grasp).

---

## 7. Every bug found this session, and the lesson

1. **Grasp descent had no settle** — the transit had one. Two legs differing in exactly one way
   measured 10× apart. *Lesson: when two similar code paths perform differently, diff them
   before theorising.*
2. **Grasp poses ranked by non-predictive statistics** — refuted three hypotheses with data
   already on disk before touching the GPU. *Lesson: the sister task had already solved this
   and written it down. Read the neighbouring code first.*
3. **The object meshes were lattices** — passed every gate at +4.8 %. *Lesson: a hollow shell
   has the same bbox as a solid. Gates measured what the pipeline already believed.*
4. **`subdivisionScheme` unauthored** → `catmullClark` fallback, which also voids normals.
5. **No material bound** → RTX ignores `displayColor`.
6. **`--density-quantile` never passed through**, and one global value is wrong in both
   directions across objects.
7. **Report crashed before printing the gate verdict** — which is why no object ever reported
   PASS/FAIL and the lattice shipped silently.
8. **`--target-height` scales by height above the FITTED PLANE** — the object3 captures put
   objects on a stack of flyers over carpet, RANSAC fit the carpet **12.8 mm** low, and
   everything came out 26 % small. Diagnosed because ~30 parameter combinations all *agreed*
   and the aspect ratio was right (1.98 vs a true 1.99). *Lesson: consistent wrong answers with
   correct shape = a scale problem, not a segmentation problem.*
9. ⭐ **`/World/Splats` is spawned once at the world origin but Isaac Lab CENTRES the env grid
   on it** — so env 0 coincides with the desk only at `num_envs == 1`. Measured at 8 envs, env 0
   is at `[2.5, -2.5, 0]`, i.e. the desk was **3.5 m away**. **Every render ever produced used
   `render_workstation.py`, which hardcodes `num_envs=1`** — the bug was structurally invisible
   to the only tool that looked. Would have surfaced as a vision student trained with no
   background. *Lesson: a tool that only ever runs one configuration cannot find configuration
   bugs.*
10. **The §5.4 rank bug recurred** in `inspect_object_mesh.py`, a tool written the same morning:
    dropping the unknown (`None`) axis before zipping scored the tape measure's 69 mm second
    axis against its 36 mm height and reported **+92 %** on a mesh within 3 %. *Lesson:
    catalogued bugs come back when a new tool re-implements the same comparison.*
11. **Hole COUNT is a bad metric** — object2's tape measure had 14 holes to object3's 26 and was
    strictly worse: a *ring* with one enormous void. *Use `hull/bbox fill` and the render.*
12. **Two changes in one experiment cannot be attributed** — I bundled `--below` with
    `--min-opacity`, blamed the opacity cut for desk speckle, and was wrong: it contributes 0.55
    of 4.75 points.
13. **A speckle metric that measured the wrong thing** — high-local-variance counting rose after
    pruning because removing the floaters *uncovered* the sharp-edged grid floor.
14. Smaller: `isaacsim.core.utils` does not exist in this build (use `omni.usd`);
    `AddTranslateOp` on a prim with an existing transform stack **appends** a composing op;
    `radiance:sphericalHarmonicsCoefficients` is **16 per gaussian** and must be filtered with
    stride or every gaussian wears another's colour; `Quatf` arrays do not survive a numpy
    round-trip; 32 envs × 2 cameras = 64 render products **OOMs a 10 GB card**.

---

## 8. ⭐ PLAN FROM HERE — *subject to change*

### Immediate, in order (this is the directive)

1. **Import a proper Rubix cube asset.** Replace `cube.usda` with a clean authored/downloaded
   56 mm cube with correct vivid sticker colours and a box collider. Keep the *reconstructed
   desk gaussians* as the backdrop. Check `Re3Sim/workstation/data/items/` and `data/items/`
   (`*.usdz` — there are existing authored props there) before authoring one.
   - Keep `REST_Z = size/2` and `placed_mask`'s `0.010 < z < 0.062` consistent (§ the
     centre-origin convention in `finish_object_asset.py`'s docstring).
   - This likely moots §5.1 for the cube but **not** for the scanned clutter.
2. **Move the workstation camera IN FRONT of the arm.** Currently `(-0.62, -0.52, 0.46)` →
   `(0.24, 0.02, 0.04)`, which is behind/over-the-shoulder. Front means roughly
   `(+0.9, 0.0, 0.35)` looking back at `(0.15, 0.0, 0.05)`. Two places: `render_workstation.py::VIEWS`
   and the `station_cam` pose in `collect_demos.py`. **Verify by rendering**, not by reasoning.
3. **Rebuild the expert on the `run_expert_v1` architecture** (§6.2, recommend option B first):
   goalset grasps → approach/descend → close + **clean-grasp check** → lift + **hold check**
   with **regrasp once** → transport → release → **verify placed**. The in-sim close screen is
   already the clean-grasp check; the missing pieces are the goalset and the recovery branches.
4. **Verify > 90 % success**, at ≥128 envs across ≥3 seeds, end-to-end from `env.reset()`,
   never under `--teleport-pregrasp`.
5. **Push the env to main**, ensuring no shared code files change. `git status` and review the
   diff file-by-file: this session touched `Re3Sim/workstation/tools/*` (pipeline, shared),
   `eva_bc/re3sim/*` (this env), and `eva_rl/.../re3sim/workstation_env_cfg.py` (this env).
   The `workstation_env_cfg.py` change (`RE3SIM_SPLATS` override) is additive and env-local.

### After that

6. **Regenerate demos and retrain BC** — every existing BC number came from the broken expert.
7. **DAgger**, now better targeted. Note its ceiling is the expert's own rate.
8. **Vision distillation** — blocked on §5.1 (washed-out colour) and now unblocked on the
   multi-env splat bug (§7.9).
9. **Real-hardware bring-up** — needs the two user answers in §5.2.

### Do NOT repeat

* Do **not** trust a sweep run under `--teleport-pregrasp` for any constant of the full manoeuvre.
* Do **not** re-derive the transit root cause — fixed, and the `high` vs `high_fk` diagnostic
  that found it is still wired into `collect_demos.py`.
* Do **not** re-enable either bias mechanism without re-measuring; both made things worse.
* Do **not** prune the splats without re-measuring both viewpoint regimes (§4.5).
* Do **not** render at `num_envs > 1` without checking the splats are aligned (§7.9).

---

## 9. Machine notes

* `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6` — **always**.
  Never invoke Isaac Sim through the env's python binary directly (CXXABI_1.3.15 segfault).
* 10 GB RTX 3080 is the binding constraint. **32 envs × 2 cameras OOMs.** 8 envs × 2 at
  640×360 works. Camera count, not resolution, is what kills it (per-render-product DLSS buffers).
* `ArmKin` uses **env count as CEM population** — fewer envs means a worse planner, so a
  render-constrained run is not a fair measure of expert quality (8 envs gave 50 %, 32 gave 72 %).
* Long jobs: Bash `run_in_background: true`. `python -u` always.
* Reconstruction of one object ≈ 78 min at 400 frames (COLMAP SIFT is **CPU-only** in this
  `pycolmap` wheel; the `tools` env's `colmap` CLI *does* have CUDA if this ever needs speeding up).

### Reproduce the current best expert

```
cd /home/eva/Desktop/isaacLab/eva_bc
GRASP_SETTLE=40 SCREEN_ROUNDS=2 BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py \
  --headless --num_envs 64 --batches 1 --seed 11 --out /tmp/x.hdf5
```

### Record the expert on video

```
GRASP_SETTLE=40 SCREEN_ROUNDS=3 BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py \
  --headless --num_envs 8 --batches 1 --seed 777 --out /tmp/x.hdf5 \
  --record-video re3sim/runs/video --record-width 640 --record-height 360
```

### Inspect any object mesh

```
python Re3Sim/workstation/tools/inspect_object_mesh.py \
  --mesh <a.ply> --mesh <b.ply> --label old --label new --expect-mm 56,56,56 --out cmp.png
```
