# 05 — Visual domain randomisation for the vision student

*Started 2026-08-09. Goal: a pixels-only policy on `Rebot-Workstation-PickPlace1-Vision-v0`
that survives a real deployment — weird lighting, a bumped camera, a noisy background —
trained straight from the verified scripted expert (no state-BC stage) plus DAgger.*

Status: **live working doc.** Findings are appended per phase as they are measured.

---

## 0. The design in one paragraph

The expert is state-based and never looks at pixels, so **visual DR is free**: every
camera/lighting/background perturbation costs the demo success rate nothing (measured
baseline to confirm, §2). Only arm-start jitter costs success (0.15 rad → 90.6 %, HANDOFF
§5c), and we pay it deliberately. DR parameters are sampled **per env per reset and held
for the episode** — a bumped camera stays bumped, which is what a real rig does. Sensor-level
effects (blur, noise, compression) are train-time image augmentations, not simulated,
because they are cheap, infinitely diverse and pixel-exact there.

## 1. Phases and gates

| phase | what | gate |
|---|---|---|
| 0 | bring-up on this machine (paths were `/home/eva`) | expert ≥ 90 % at 128 envs, seed 11 (dev machine: 96.9 %) |
| 1 | baseline visual dataset + student, **no new DR** | cold-reset eval, held-out seeds — the pre-DR reference |
| 2 | DR module: camera events + light events + backdrop/distractors + train-time augs | rendered stills grid, eyeballed; expert success unchanged vs §0 |
| 3 | DR dataset ~2–3k successful eps, 20 % nominal / 80 % DR | per-batch success ~90 % (arm jitter only) |
| 4 | train DR policy; eval vs baseline on nominal AND DR settings | pooled ≥128-ep, held-out spawn seeds |
| 5 | DAgger rounds under DR | retrain + re-eval; teacher is the true scripted expert |
| 6 | robustness matrix: per-axis eval at held-out magnitudes | the deploy-readiness report |

## 2. The DR axes

Sampled per env per reset unless stated. Ranges are the plan of record (Big Will approved
2026-08-09); measured adjustments get logged here with data.

**Camera extrinsics/intrinsics (in-sim, new event terms in `mdp/visual_dr.py`):**

| # | axis | range | note |
|---|---|---|---|
| C1 | station cam position | ±3 cm/axis | "shifted out of position" |
| C2 | station cam aim (target) | ±2–3 cm | tilt via look-at |
| C3 | station cam roll | ±4° | needs explicit look-at + roll quaternion — `set_world_poses_from_view` has no up/roll control |
| C4 | bumped-rig mode | ±8 cm / ±10°, ~15 % of episodes | the big out-of-position case |
| C5 | focal length | ±8 % | USD `focalLength` attr per env prim |
| C6 | wrist mount | ±2° rot, ±5 mm trans | local xform of `WristCam` prim under the gripper link |

**Lighting (in-sim):** the shipped scene has ONE global dome light (`/World/light`), so
per-env lighting needs per-env light prims — added in the DR env variant.

| # | axis | range |
|---|---|---|
| L1 | dome intensity | ×[0.3, 3.0] (global, per reset) |
| L2 | dome colour temperature | 2700–6500 K + tint |
| L3 | per-env key light direction/elevation | full azimuth, 20–80° elevation |
| L4 | per-env coloured fill light | 1–2 lights, ~25 % of episodes, saturated random hue |

**Background/scene (in-sim):**

| # | axis | note |
|---|---|---|
| B1 | distractor primitives | 0–4 per env, random hue/size/pose, **visual-only (no collider)** so physics and the expert are untouched |
| B2 | backdrop plane behind desk | random solid colour v1; noise textures v2 if needed |
| B3 | splat desk visibility | occasionally hidden per env → bare/backdrop background |
| B4 | box + table tint jitter | material colour writes |

**Sensor-level (train-time, `act/dataset_vision.py` opt-in augment flag):**
gaussian/defocus blur σ 0–2 px, sensor noise, gamma/exposure, brightness/contrast/
saturation/hue, JPEG q 30–95, small cutout patches, small random crop-shift-resize.

**Constraint that shapes all of it:** nothing in the DR may touch physics — colliders,
masses, spawn logic — or the expert's success would no longer be the known 95/90 %. DR
lives in a separate env variant (`...Vision-DR-v0`) so the shipped Vision env stays
bit-identical to its verified state.

## 3. Phase 0 — bring-up log (2026-08-09)

* Pulled `eva_rl` (here `reBot_RL`) and `eva_bc` (here `reBot_ACT`, was 12 commits behind —
  the pull brought the vision shards + student + DAgger stack, commit `057439e`).
* All `/home/eva/...` hardcoded paths in `re3sim/**/*.{py,sh}` replaced with
  script-relative resolution (`_ROOT` from `__file__`; `cd "$(dirname BASH_SOURCE)/.."`).
* This machine: RTX 4080 Laptop **12 GB** (dev machine was 10 GB — the documented
  render-pixel ceilings should hold with a little headroom), `env_isaaclab6` present,
  splats shipped as real files (88 MB, no dangling symlink).
* Expert verification run: **PASS — 124/128 = 96.9 %, reproducing the dev machine's seed-11
  number to the episode** (planned 128/128, grasped 124/128, closed-on-air 4, lost 0).
  "Deterministic in the seed" holds across machines, so every dev-machine number transfers.

## 4. Phase 2 — the DR module as built (2026-08-09)

Code: `mdp/visual_dr.py` + `workstation_vision_dr_env_cfg.py` + gym ids
`Rebot-Workstation-PickPlace1-VisionDR-v0` / `-Play-v0` (reBot_RL), `act/augment_vision.py`
+ `--augment` on `re3sim/act/train_flow_vision.py` (reBot_ACT). `collect_demos.py` now
skips its fixed station-cam re-aim when the task cfg declares `aim_station_cam` — otherwise
every batch would silently undo the camera DR; the default state task keeps the old path,
so the round-1 recipe is untouched.

**Camera-path gate: PASS.** `scripts/probe_vdr_camera.py` asserts the DR camera path
(look-at matrix + roll post-multiply + `set_world_poses(convention="opengl")`) reproduces
the stock `set_world_poses_from_view` pose at zero DR: max |Δpos| = 0.0 m, max |Δquat| =
1.5e-8. The roll math is safe to trust.

**Stills grid, round 1 — what looking at it caught** (8 envs × 4 resets, all warnings
silent so every DR prim was found):

* WORKING: per-env camera pose spread (incl. an obvious bump draw), per-env cube patterns,
  splats-hidden draws, backdrop colour variety, distractors on the desk, arm shadows from
  the per-env key lights, warm/cool dome shifts between resets.
* CAUGHT 1: **Isaac's black-with-gridlines ground plane is visible past the desk edge in
  100 % of station frames** — a sim-only texture the DR was supposed to be burying, present
  as a constant anchor in every image. Fix: a per-env 4×4 m visual-only "floor card" 1 m
  below the desk top, colour drawn per reset.
* CAUGHT 2: a distractor sphere hovered over the void past the splat crop's edge (x ≈ 0.9)
  — the placement bounds were the analytic slab, not the *visible* desk. Fix: x ∈ [−0.30,
  0.85], y ∈ [−0.42, 0.42].
* CAUGHT 3: the wrist backdrop read as a blown-out white sky in ~3/8 envs — the "muted"
  colour branch reached value 0.88. Fix: value capped at 0.75 (`_surface_color`).

`act/augment_vision.py` measured at 10.1 ms/frame (CPU), strength-0 verified a strict
no-op; use `--num-workers 8` so it stays off the training critical path.

**Stills grid, rounds 2–3** (after the round-1 fixes):

* Floor cards kill the grid-floor cue; overlapping neighbour cards get a 0.5 mm per-env z
  stagger and read as a patchwork floor — more background variety, no shimmer.
* CAUGHT 4: "splats hidden" left the cube and box FLOATING — the physics slab is authored
  invisible beneath the splats, so the draw deleted the desk from the image. Fixed by
  swapping instead of hiding: splats off ⇒ analytic slab visible with a random tint. Reads
  as "a different table", which is what a real redeploy looks like.
* CAUGHT 5: past the 4 m card edge the default ground plane still showed at grazing wrist
  angles → cards grown to 6 m + the global ground plane's shader tinted per full-batch
  reset. Its grid lines survive the tint but now read as far-floor tiling, kept.
* FALSE ALARM, worth recording: one env looked like splats AND tinted slab at once — it
  was the floor card seen past the splat crop's edge, 1 m below (parallax). Verified by
  the next reset's draws.
* Verdict: per-env variety is real across camera pose/roll, lighting, desk appearance,
  floor, wall, distractors; the invariants that remain are the cube, the box and the arm —
  which is the DR contract working, not a gap.

Still owed for the phase-2 gate: expert success on `-VisionDR-v0` ≈ the 90.6 %
arm-jittered baseline (visual DR is state-invisible, so any real drop = a physics leak).
Measured by the first DR collection batch in phase 3.

## 4b. ⭐ ONE Isaac instance at a time on this machine (learned the hard way)

Running the 128-env splats-per-env collection *concurrently* with an 8-env stills render
took the whole machine down (2026-08-10 ~06:10): last reading before the crash was **2.3 GB
free of 31 GB RAM, 0 MB free swap**, then both processes and /tmp were gone (reboot). The
GPU had headroom (4.4 GB/12 GB) — it was **system RAM** that ran out: two Kit processes,
each holding a 356k-gaussian stage cloned per env, plus the collector's in-RAM frame
buffers. The 10 GB dev-machine ceilings in HANDOFF §4 are all VRAM lessons; this is the
same trap one level up. Schedule renders/evals strictly after collections finish.

## 4c. ⭐ Why data collection froze the machine — the RAM budget, closed

Both freezes (2026-08-10 and -11) are the same arithmetic. `collect_demos.py --shards`
holds every frame of the batch in RAM until `shard_write`: 128 envs × 2 cams × 160×120×3
uint8 = **14.7 MB per step**, and demos run 1018–1118 steps → **~16 GB of buffers per
batch**, on top of ~12–14 GB for Isaac's 128-env scene with per-env splats. That is
28–30 GB on a 31 GB, 2 GB-swap machine: the kernel thrashes and the box freezes at the
*end* of batch 0 every time. The first freeze merely arrived earlier because a second Kit
instance was also resident (§4b). The dev machine this recipe shipped from had the RAM
for it; this one does not.

Fixes, both now standard here:
* **64 envs × 6 batches** per seed (≈ same episode count): buffers halve to ~7.4 GB and
  the scene shrinks with them. Planner population 64 is above the 32 floor; the phase-3
  success gate will price whatever it costs vs the 128-env 90.6 %.
* **Collection always runs under `systemd-run --user --scope -p MemoryMax=26G`** — worst
  case is a killed collector, never a frozen machine. Verified working on this box.

## 4d. Triple-confirm gates before any collection (2026-08-11, expert fix pending)

Big Will's directive: the pushed expert is being fixed; collect nothing until the fix is
pulled, and have everything verified for pull-and-go. Gates run and their verdicts:

| gate | method | verdict |
|---|---|---|
| dynamics invariance | `probe_vdr_invariance.py` ×2 tasks + `compare_vdr_traj.py`: same seed, same 200-step scripted action sweep, obs41 trajectories | **PASS — bit-exact (max Δ = 0.0)**. No collider/physics leak from any "visual-only" DR prim; expert numbers carry over by construction |
| DR determinism | re-seed torch, re-reset, compare camera draws | **PASS** (1.5e-8) — datasets regenerate exactly from the run seed |
| DR variety | consecutive resets differ; stills grids | **PASS** (grids are the real evidence; the numeric across-env stat is origin-dominated) |
| camera path | `probe_vdr_camera.py` zero-DR vs stock aim | **PASS — 0.0 m / 1.5e-8**, re-run after every module edit |
| end-to-end shards | 32-env smoke batch on `-VisionDR-v0`, RAM-capped + sampled; frames inspected, loader + leak-check + augment run on the output, then deleted | **PASS** — see below |

Smoke-batch verdicts (32 envs × 1 batch, seed 777, `--shard-include-failures`, deleted
after inspection; evidence frame grid kept at `renders/vdr_shard_frames.png`):

* expert on the DR task: **27/32 = 84.4 %** (planned 32/32, closed-on-air 2, lost-after-
  grasp 1, never-got-there 5) — the 32-env-planner + arm-jitter taxonomy, no DR-shaped
  failure mode. (Context: dev machine's 32-env jittered batch scored 90.6 %; n=32 spread
  is ±2 envs. The current expert also has known upstream issues being fixed.)
* `inspect_shards.py`: workspace near-black 1.6 % (the no-desk failure runs 40 %+),
  frame-precedes-action causality r = +0.842, no privileged keys in the student sample.
* mid-episode frames from the shards show the DR baked into the DATA: per-episode camera
  framing/tilt, floor/backdrop colours, lighting; cube stickers and box interiors stay
  legible; `--augment 1.0` previews add blur/noise/cutout/tint without erasing task
  content. One near-black wrist frame is the gripper/box occluding the lens mid-place — a
  real-rig phenomenon, present in the stock task too.
* RAM peak **16.9 GB** total system at 32 envs → ~21–23 GB projected at 64 envs (fits the
  26 GB cap), 128 confirmed infeasible (§4c).
* disk, corrected by measurement: **~94 MB/episode** (815-step mean × 115 KB), not the
  ~60 MB first estimated. Baseline (64×6, successes only) ≈ 30 GB; three DR seeds ≈
  95 GB. 146 GB free covers it with ~20 GB slack — if that pinches, drop DR seed 33
  first; every slice regenerates from its seed.

## 4e. Independent code review of the DR diffs (2026-08-11)

A high-effort review was run over every working-tree change in both repos. Confirmed and
fixed so far (final consolidated findings pending):

* `record_video.py` re-aimed the station camera with **no** `aim_station_cam` guard — the
  exact clobber-the-DR bug the other three call sites were guarded against. Guarded.
* `eval_vdr_matrix.sh`: relative `--ckpt`/`--out` silently re-resolved after the `cd`
  (now canonicalised first); a stale row from a *different* checkpoint would be silently
  reused on resume (now verified against the JSON's recorded ckpt); caller-exported
  `RE3SIM_VDR`/`RE3SIM_VDR_SCALE` leaked into every row (now pinned per row).
* `augment_vision.py`: at strength ≳2.9 the photometric factor draws went NEGATIVE and
  inverted the image — not a stronger corruption, a different and unphysical one, and the
  matrix sweeps strength upward. Factors floored; verified stable at strength 3.0.

## 6. Pull-and-go runbook (the moment the fixed expert lands)

```bash
# 0. pull — reBot_ACT is dirty with the DR wiring (aim-skip, augment); stash around it
cd ~/Desktop/isaacLab/reBot/reBot_ACT && git stash && git pull && git stash pop
cd ~/Desktop/isaacLab/reBot/reBot_RL  && git pull       # clean, new files only
# if the pull touched collect_demos.py / eval_flow_vision.py: re-apply the aim-skip guard
# (search "aim_station_cam" — 6 lines each) and re-run BOTH probes before collecting.

# 1. re-verify the expert on the fixed code (state task, no cameras — RAM-light)
conda activate env_isaaclab6 && cd ~/Desktop/isaacLab/reBot/reBot_ACT
python -u re3sim/expert/collect_demos.py --headless --num_envs 128 --batches 1 \
    --seed 11 --out /tmp/verify_s11.hdf5          # gate: ≥90 %, expect ~96.9 %

# 2. gates (cheap, serial): invariance pair + camera probe — all must PASS
# (commands in §4d; re-run because the pull may have moved shared code)

# 3. baseline (nominal) — ONE job at a time, always under the memory cap
systemd-run --user --scope -p MemoryMax=26G -- \
  bash re3sim/expert/collect_vision.sh re3sim/expert/data/vision_base_s21 64 6 21

# 4. DR set — seeds 31/32/33; baseline doubles as the ~25 % nominal mix fraction
for SEED in 31 32 33; do
  systemd-run --user --scope -p MemoryMax=26G -- \
    bash re3sim/expert/collect_vision.sh re3sim/expert/data/vision_vdr_s$SEED 64 6 $SEED \
      --task Rebot-Workstation-PickPlace1-VisionDR-v0
done
# phase-3 gate: per-batch expert success ≈ the arm-jittered baseline (visual DR is
# state-invisible — a drop means a physics leak, stop and diagnose)

# 5. train baseline, then DR policy (+ sensor augs), eval both on nominal AND DR
python -u re3sim/act/train_flow_vision.py --data re3sim/expert/data/vision_base_s21 \
    --out re3sim/runs/vbc_base --steps 100000 --seed 1 --num-workers 8
python -u re3sim/act/train_flow_vision.py --augment 1.0 --num-workers 8 \
    --data re3sim/expert/data/vision_base_s21 re3sim/expert/data/vision_vdr_s31 \
           re3sim/expert/data/vision_vdr_s32 re3sim/expert/data/vision_vdr_s33 \
    --out re3sim/runs/vbc_vdr --steps 100000 --seed 1
bash re3sim/act/eval_vdr_matrix.sh re3sim/runs/vbc_vdr/ckpt_final.pt re3sim/runs/vdr_matrix

# 6. DAgger under DR, retrain, re-run the matrix
systemd-run --user --scope -p MemoryMax=26G -- \
  bash re3sim/expert/collect_vision.sh re3sim/expert/data/vision_vdr_dagger1 64 3 41 \
    --task Rebot-Workstation-PickPlace1-VisionDR-v0 \
    --dagger-ckpt re3sim/runs/vbc_vdr/ckpt_final.pt
```

## 5. Scale revision (disk, measured)

146 GB free; shards run ~60 MB/episode (uint8 frames ×2 cams). The approved ~2–3 k DR
episodes would need ~180 GB. Revised: **baseline = seed 21 × 3 batches ≈ 345 successful
(~21 GB); DR set = 3 seeds × 3 batches ≈ ~1 050 successful (~62 GB).** Nothing is lost:
collection is deterministic in the seed, so any slice can be regenerated on demand.

*(phase 1/3 results land here)*
