# HANDOFF — re3sim photoreal workstation
*Written 2026-08-09, end of the directive session. Supersedes
`HANDOFF_2026-08-09_predirective.md`, `HANDOFF_2026-08-06.md` and `HANDOFF_prev.md` (all kept).*

Read `04_AUTHORED_CUBE_AND_RECOVERY_EXPERT.md` alongside this: it is the detailed record of
what was done and why. This is the state and the plan.

---

## 0. The directive, and where it stands

> Ditch the gaussians and just import a proper rubix cube — set this env up with just the
> workstation rendering in gaussians. Use a very similar expert to the og pick-and-place
> expert. Put the workstation camera in front of the arm. After verifying the expert can
> complete > 90 % of trajectories, push the env to main (ensuring it doesn't change any shared
> code files).
>
> — and mid-session: *make sure the rubix cube is robust to different colors (i.e. different
> patterns on the cube).*

| # | item | state |
|---|---|---|
| 1 | proper Rubix cube; gaussians for the workstation only | ✅ authored, 56 mm, exact box collider; clutter reverted to primitives |
| 1b | robust to different patterns | ✅ valid cube states, per-env randomised, verified by rendering |
| 2 | expert like `run_expert_v1` | ✅ goalset → descend → close → hold check → regrasp → verify placed |
| 3 | workstation camera in front of the arm | ✅ `STATION_CAM_EYE`, chosen by rendering candidates |
| 4 | verify > 90 % | ✅ **see §2** |
| 5 | push the env to main, no shared-file changes | see §5 — **structurally clean**, no tracked file in either repo is modified |

---

## 1. Layout

```
eva_rl/source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/
    workstation_env_cfg.py       scene, cameras, events        ← STATION_CAM_EYE lives here
    rubiks.py                    ⭐ the cube's STATE model (shared with the author script)
    mdp/{common,events,observations,rewards,terminations}.py
eva_rl/source/reBot_RL/data/workstation/     box.usda, objects/*.usda, splats symlink (ignored)
eva_rl/scripts/author_rubiks_cube_usd.py     ⭐ writes cube.usda + pattern variants
eva_rl/scripts/author_workstation_box_usd.py
eva_rl/scripts/render_workstation.py         now takes --num_envs / --env
eva_bc/re3sim/expert/workstation_expert.py   planner: grasp_choices(), plan_episode(choices=)
eva_bc/re3sim/expert/collect_demos.py        ⭐ the executor: goalset, screen, retry, masks
eva_bc/re3sim/probes/_kin.py                 ArmKin (CEM population == env count!)
eva_bc/expert/run_expert_v1.py               the "og" cuRobo expert this was ported from
```

---

## 2. Measured state

### The expert — **VERIFIED, 128 envs, 3 seeds, end-to-end from `env.reset()`**

| seed | success | planned | ≥1 lifting candidate | grasped | lost after grasping |
|---|---|---|---|---|---|
| 11 | **96.9 %** | 128/128 | 122→126/128 | 124/128 | 0 |
| 12 | **93.0 %** | 128/128 | 126→128/128 | 121/128 | 2 |
| 13 | **96.1 %** | 128/128 | 120→127/128 | 124/128 | 1 |
| | **mean 95.3 %, min 93.0 %** | | | | |

Never under `--teleport-pregrasp`. Runs are **deterministic in the seed** — seed 11 was re-run
after an unrelated fix and reproduced to the episode.

How it got there, from 71.9 % at the start of the session:

| | |
|---|---|
| shipped expert, scanned cube, 64 envs | 71.9 % |
| authored cube, same expert, 64 envs | 73.4 % — **the cube swap changed nothing** |
| goalset + retry, teleport screen, 32 envs | 65.6 % |
| + screen replays the real DESCENT | 75.0 % |
| + re-draw envs with no lifting candidate | 78.1 % |
| + restore the cube's SPAWN ORIENTATION | 87.5 % |
| + rank candidates by measured LIFT HEIGHT, 128 envs | **95.3 %** |

The two changes that mattered most were both bugs of *measurement*, not of control: a screen
that teleported past the part of the manoeuvre that fails, and an object restore that left the
cube rotated away from the yaw every plan had been solved against. See
`04_AUTHORED_CUBE_AND_RECOVERY_EXPERT.md` §5.

### Everything else

| thing | state |
|---|---|
| cube | authored, 56 mm, exact box collider, vivid materials, per-env valid patterns |
| clutter | analytic primitives (`RE3SIM_SCANNED_CLUTTER=1` restores the scans) |
| desk | reconstructed gaussians, unpruned, aligned at any `num_envs` |
| workstation camera | in FRONT of the arm, one pose shared by the renderer and the recorder |
| BC policy | **19.5–20.1 %, and every number is void** — trained on the broken expert, §6 |

---

## 3. How to reproduce

```
cd /home/eva/Desktop/isaacLab/eva_bc
python -u re3sim/expert/collect_demos.py --headless --num_envs 128 --batches 1 --seed 11 \
  --out /tmp/verify.hdf5
```

Film the expert (wrist + workstation), one view per run:

```
cd /home/eva/Desktop/isaacLab/eva_bc
for CAM in station wrist; do
  python -u re3sim/expert/collect_demos.py --headless --num_envs 32 --batches 1 --seed 777 \
    --out /tmp/vid_$CAM.hdf5 --record-video re3sim/runs/video_hq --record-cams $CAM \
    --record-width 960 --record-height 540 --record-stride 1 --record-fps 50 --record-quality 10
done
```

⚠ **Filming is memory-bound, and the constraint is total render-product PIXELS.** `Camera`
requires one prim per env (it raises if `_view.count != num_envs`), so N envs always means N
cameras per view, and the RTX path tiles them into ONE atlas — a 32-env × 2-cam × 720p run
tries to allocate a single 7680×4320 texture and dies. `--record-cams` films one view per run
so the run can keep enough envs to plan with: at 16 envs the goalset planner solves only 8/16.
Runs are deterministic in the seed, so two single-view runs at the same seed are frame-aligned
and can be composited.

Measured ceilings on the 10 GB card: **32 envs × 1 cam × 960×540 works; 32 × 1 × 1280×720
OOMs; 16 × 2 × 1280×720 fits at 8.97 GB but starves the planner.**

Render the env (and check the splats are aligned at >1 env):

```
cd /home/eva/Desktop/isaacLab/eva_rl
python -u scripts/render_workstation.py --headless --num_envs 4 --env 0 --out /tmp/renders
```

Rebuild the cube assets:

```
python scripts/author_rubiks_cube_usd.py --variants 8
```

### Knobs that matter

**The defaults ARE the verified configuration** — `--num_envs 128` with no environment
variables reproduces the measured expert. `SCREEN_ROUNDS` used to default to 0 (no screen) while
every quoted number was measured with one; that trap is closed.

| var | default | what it does |
|---|---|---|
| `GOALSET` | 4 | candidate grasps solved per env |
| `RETRIES` | 2 | in-episode regrasp attempts after the first |
| `SCREEN_ROUNDS` | 4 | 0 = no screen; ≥1 screens every goalset slot, then `n−1` re-draw rounds |
| `SCREEN_DESCEND` | 1 | ⭐ screen by replaying the real descent, not by teleporting |
| `GRASP_SETTLE` | 40 | env steps held at the grasp pose before closing |
| `HOLD_RISE` | 0.030 | cube rise that counts as "the grasp worked" |
| `RE3SIM_CUBE_PATTERN` | — | integer picks `cube_pNN.usda` (`p00` = solved) |
| `RE3SIM_CUBE_PATTERN_DR` | 1 | per-env pattern randomisation |
| `RE3SIM_SCANNED_CLUTTER` | 0 | restore the photogrammetry clutter |
| `RE3SIM_SPLATS` | — | point at a different gaussian field |
| `RE3SIM_ARM_START_JITTER` | 0.0 | per-joint uniform jitter [rad] on the arm's start pose |
| `--record-cams` | both | film one view per run; halves the render-product pixels |
| `--record-keep-pad` | off | film env 0 even while it is only being held for the batch |
| `--record-warmup` | 180 | discarded pre-roll steps before the first kept frame |

`GOALSET=1 RETRIES=0 SCREEN_ROUNDS=0 BIAS_MAX=0.25` reproduces the pre-2026-08-09 executor.

---

## 4. ⭐ Traps, in the order they will bite

1. **`ArmKin` uses the env count as its CEM population.** A small run is a *worse planner*, not
   just a smaller sample. 8 envs returned `0/8 planned` and looked like a broken patch. Never
   smoke-test this expert below 32 envs, and never quote a success rate from a
   camera-constrained run.
2. **Never invoke Isaac Sim through the env's python binary directly** — CXXABI_1.3.15
   segfault a second into startup, real cause ~400 lines into the Kit log. Always
   `conda activate env_isaaclab6` first.
3. **10 GB is the binding constraint. 32 envs × 2 cameras OOMs.** 8 envs × 2 at 640×360 works.
   Camera *count*, not resolution, is what kills it.
4. **`/World/Splats` is spawned once at the world origin** and Isaac Lab centres the env grid
   on it, so it only lines up with env 0 at `num_envs == 1`. Both `render_workstation.py` and
   `collect_demos.py` now move it onto the env being filmed.
5. **`set_world_poses_from_view(env_ids=None)` does not broadcast** a single pose to N cameras.
6. **`startup`-mode events get `env_ids=None`**, not a tensor.
7. **`cfg.prim_path` is already resolved** — `/World/envs/env_.*/Cube`, not `{ENV_REGEX_NS}/…`.
8. **Isaac Sim's first import takes minutes.** Slow startup is not a hang.
9. **A render outside a simulation step produces no frame.** `sim.render()` in a loop --
   with or without an annotator read, with or without a preceding `sim.forward()` -- does not
   warm the renderer up. Three attempts failed on that assumption. What works is *stepping*:
   `render_workstation.py` steps 60 times before rendering, and `collect_demos.py` now
   pre-rolls real env steps and discards the frames. Symptom when it is missing: the first
   frame is at the camera's spawn pose and the 356k-gaussian desk is absent for ~120 frames,
   so the arm and the cube float on a bare grid floor while everything else looks correct.
10. **Filming is bounded by total render-product PIXELS, not resolution.** `Camera` raises if
   `_view.count != num_envs`, so N envs is always N cameras per view, and RTX tiles them into
   ONE atlas -- 32 envs x 2 cams x 720p asks for a single 7680x4320 texture and dies. Measured:
   32x1x960x540 works, 32x1x1280x720 OOMs, 16x2x1280x720 fits at 8.97 GB but starves the
   planner (8/16 envs solved). Film one view per run with `--record-cams`; runs are
   deterministic in the seed, so two single-view runs composite frame-for-frame.
11. Long jobs: Bash `run_in_background: true`, `python -u`, always.

---

## 5. Pushing to main (item 5)

**No tracked file in `eva_rl` or `eva_bc` is modified.** `git diff --stat` is empty in both;
everything for this env is new/untracked, and `tasks/__init__.py` discovers subpackages via
`import_packages`, so registering the env needs no edit to a shared file. The directive's
constraint is satisfied structurally, not by care.

Excluded deliberately (see `data/workstation/.gitignore` and its README):
* `splats.usd`, `splats_splats.usd` — absolute symlinks into the *Re3Sim* repo's build output;
  committing them hands everyone else a dangling link. The env degrades to a placeholder desk
  and says so at load time.
* `objects_object2_backup/` — superseded photogrammetry assets kept locally for A/B.

⚠ **`Re3Sim` has four MODIFIED TRACKED files** (`re3sim/scripts/reconstruct.py`,
`workstation/tools/{align_to_marker,extract_object,mesh_to_rigid_usd}.py`) plus a dozen new
tools. Those are the *shared reconstruction pipeline* and are **not** part of this env's push —
the env no longer depends on any of them, since the cube is authored and the clutter is
primitives. They are left uncommitted for a separate, deliberate decision.

---

## 5b. ⭐ The expert "pauses a lot" — measured, and one of the causes was a BC data bug

Asked why the filmed expert pauses so much. Measured on the recorded episode (env 0, 1018
steps) rather than guessed:

```
phase       steps  % of ep   still  still %
approach      668    65.6%     393    58.8%
close          70     6.9%       0     0.0%
lift           50     4.9%       0     0.0%
carry         140    13.8%       0     0.0%
release        30     2.9%      17    56.7%
retreat        60     5.9%       9    15.0%

TOTAL STILL (max |joint vel| < 0.02):            419/1018 = 41.2 % of the episode
steps where the COMMAND did not change at all:   721/1017 = 70.9 %
longest unchanged-command runs (steps): [412, 112, 72, 42, 12, 7, ...]
```

There are **two different causes** and they want opposite treatment:

**(a) Deliberate settles — keep them.** `stay(TRANSIT_SETTLE=40)` before the descent,
`stay(GRASP_SETTLE=40)` before the close, `stay(70)` for the close itself, `stay(30)` to
release, `stay(40)` to let the cube settle before judging. These are load-bearing: adding
`GRASP_SETTLE` alone took the expert 40.6 % → 54.7 %, because a position drive that is still
travelling when the fingers shut closes on a pose the arm has not reached. ~220 steps total.

**(b) ⭐ Batch padding — a bug, and not only a cosmetic one.** The single longest run of an
unchanged command is **412 steps — 8.2 s of a 20.4 s video**. That is env 0 finishing its
transit early and then being held while the slowest of 32 envs catches up, because `pad()`
extends every segment to the batch maximum by repeating its last waypoint.

Those steps were being **recorded as demonstrations**. A policy trained on them sees, at a
state where the correct action is "descend", a large number of samples that say "freeze" —
contradictory supervision at the same state, and a plausible contributor to the BC plateau at
~20 %. `pad()` now also returns `own[t, i]` ("is env i still on its own waypoints"), and those
steps are masked out of `train_mask` exactly as the retry-idle steps already were. The
deliberate settles stay unmasked, because the policy *should* reproduce them.

For filming, `grab()` now skips frames where env 0 is only being held (`--record-keep-pad`
restores them). The video went **1018 → 608 frames**, i.e. 20.4 s → 12.2 s, with the same
manoeuvre.

*Lesson: "the demonstrations contain a lot of holding" and "the video looks slow" were the
same defect seen from two ends, and the visible one is what surfaced it.*

## 5c. Randomised arm start positions — implemented, sweep in flight

Every episode this env has ever produced began at exactly one arm pose, and the expert's
transit is solved *from* that pose, so neither the policy nor the expert had ever been asked
whether they depend on it.

* `mdp.randomize_arm_start` — a `reset` event applying uniform per-joint jitter to the six arm
  joints, clamped into the soft limits. Fingers are left alone: the gripper is a binary
  command and starting it half-closed is not a different initial condition, it is an illegal
  one. **Off by default**; `RE3SIM_ARM_START_JITTER=<radians>` turns it on, so every number
  measured before this still describes the shipped env.
* `plan_episode(q_start=...)` solves the transit chain and the home TCP from the pose the env
  is **actually** in, not from `kin.q_arm0`.
* The executor captures `q_start` right after `env.reset()` (before any planning teleport) and
  restores the arm to *that*, not to the shared default — otherwise the restore would silently
  undo the randomisation the plans were just solved against.

⚠ **The sweep (`jitter = 0.0 / 0.15 / 0.30 rad`, 128 envs, seed 13) was still running at
compaction.** Results land in `/tmp/.../scratchpad/jit_*.hdf5` and the log; re-run with:

```
cd /home/eva/Desktop/isaacLab/eva_bc
for J in 0.0 0.15 0.30; do
  RE3SIM_ARM_START_JITTER=$J python -u re3sim/expert/collect_demos.py --headless \
    --num_envs 128 --batches 1 --seed 13 --out /tmp/jit_$J.hdf5
done
```

**`jitter=0.0` is the control and must reproduce 96.1 % / 123 of 128 on seed 13.** If it does
not, the regression is in the `q_home = q_start` change or the padding mask, not in the
jitter. Expect degradation with jitter for two reasons worth separating in the taxonomy:
`plan-failed` rising means the transit could not be solved from the new start (a planner
limit), while `never-got-there` rising means it was solved and not followed (a control limit).
A jittered start can also spawn the gripper intersecting an object, since `reset_objects` keeps
clear of the box but not of the arm.

## 5d. The expert videos

`eva_bc/re3sim/runs/video_hq/` (gitignored — 160 MB):

| file | view |
|---|---|
| `expert_sidebyside.mp4` | both, labelled — **start here** |
| `expert_station.mp4` | workstation cam, in front of the arm |
| `expert_wrist.mp4` | wrist cam (D405 mount) |

960×540, 50 fps, quality 10, **every** step recorded, 608 frames / 12.2 s, env 0 = SUCCESS.
Filmed at 32 envs (batch scored 90.6 %) — **not** the 128-env configuration, because filming
is pixel-bound (§4.10). Quote the 128-env number for the expert, not this one.

Reproduce: see §3. The two views come from two runs at one seed and composite frame-for-frame
because runs are deterministic in the seed.

## 6. Plan from here — *subject to change*

1. **Finish the arm-start sweep** (§5c) — it was running at compaction. Check the control
   first.
2. **Regenerate demos and retrain BC.** Every BC number on record (7.8 / 20.1 / 19.5 % for
   1024 / 2048 / 4096 episodes) came from an expert that closed its gripper while the arm was
   still moving, ran against a cube whose yaw did not match its plan, **and was trained on
   demonstrations in which ~40 % of the steps were the arm holding still for the batch's
   sake** (§5b). At ~64 % retention a 95 % expert projects to ~60 %. **The whole curve needs
   redoing, and the padding fix alone may move it.**
   * `train_mask` is now **per step**, and demos carry an `attempts` attribute. Decide
     deliberately whether recovery episodes (`attempts > 0`) belong in the training set — they
     are a genuinely harder, multimodal behaviour to imitate.
   * Consider collecting with a **non-zero start jitter** once §5c says what it costs: a
     policy whose training set contains exactly one initial arm pose has no reason to be robust
     to any other, and that is the first thing real hardware will violate.
3. **DAgger**, better targeted now. Its ceiling is the expert's own rate.
4. **Vision distillation.** Now unblocked on both of its old blockers: the multi-env splat bug
   (§4.4) and the washed-out object colour (the cube is authored with real materials). Note the
   camera constraint in §4.3 and the planner-population trap in §4.1 — a vision run *must* be
   small, so its expert will be weaker than the 128-env number.
5. **Real-hardware bring-up.** Still needs the two user answers below.

### Still needs the user

* **the box's second outer dimension** — reconstruction says 229 × 261 × 90 mm against a
  measured sheet giving `longest: 218`. The env authors 218 × **150** and that 150 was always a
  guess.
* **arm mounting yaw** — 180° is applied (91.3 % workspace coverage); the measured argmax was
  135° (97.1 %).

### Open bug, now much smaller in scope

* **Scanned assets render ~2.6× desaturated** (mesh saturation 172/255 in, 67/255 out; 0.0 % of
  pixels clipped, so not exposure). This no longer affects anything the env loads by default —
  the cube is authored and the clutter is primitive — but it will return the moment another
  photogrammetry asset is introduced. Remaining suspect: RTX not honouring
  `UsdPrimvarReader_float3 → UsdPreviewSurface.diffuseColor` at full strength. The authored
  cube's fix (constant `diffuseColor` per material, near-pure hues with a small floor) is the
  workaround that works.

### Do NOT repeat

* Do **not** quote a success rate from a run under `--teleport-pregrasp`, or below 32 envs.
* Do **not** re-enable the settle-bias or the runtime integrator without re-measuring; both
  made things worse.
* Do **not** prune the splats without re-measuring *both* viewpoint regimes — removing the
  9.3 % below-table gaussians fixes grazing views and degrades top-down speckle 3.25 → 8.00 %,
  because gaussians are volumetric, not surface samples.
* Do **not** trust a screen, gate or renderer that cannot reach the configuration you are
  worried about. Three separate bugs this session were invisible for exactly that reason.
* Do **not** assume a render outside a simulation step produces a frame. It does not, and
  three warm-up attempts failed on that assumption before a stepped pre-roll fixed it (§4.9).
* Do **not** read "the arm is holding still" as "the expert is being careful" without checking
  `pad()` first — 412 consecutive steps of it were batch padding (§5b).
