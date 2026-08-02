# Handoff — reBot challenge suite

Session of 2026-08-02. Read this first, then `docs/CHALLENGE_SUITE.md` (measured hardware
envelope + design rationale) and `docs/envs/README.md` (per-env reference).

## The goal

Extend `reBot_RL` beyond its single pick-and-place task with a suite of environments that
each isolate a **different manipulation skill** — not just different objects. Emphasis on
**precision placement** (a tight target, not a forgiving basket). Every env must be
validated as actually achievable on this hardware, and every env gets a video.

---

## Environment setup (do this first, every time)

```
cd /home/rei/Desktop/isaaclab/eva_rl
source $(conda info --base)/etc/profile.d/conda.sh && conda activate env_isaaclab6
```

- Always run python from `/home/rei/Desktop/isaaclab/eva_rl` (the shell cwd resets).
- **Do not pass `--headless`** — deprecated in Isaac Lab 3.0; headless is the default.
- Filter noise: `2>&1 | grep -vE "Warning\]|DeprecationWarning|absl::|descriptor_database"`
- GPU is an **11 GB RTX 2080 Ti**. Camera rendering above ~256 envs throws
  `CUDA error: an illegal memory access was encountered`. Use ≤32 envs for video, up to
  512+ for headless measurement.

---

## Read this before touching anything

**This session was mostly a measurement session, and it invalidated several numbers the
previous handoff treated as settled.** Four of them changed env designs. If you are picking
up from the older handoff, these supersede it:

| # | what was believed | what is measured | consequence |
|---|---|---|---|
| C10 | TCP is 75 mm behind `gripper_end` | **41.9 mm** | every scripted grasp closed 33 mm past the object and shut on air; the `ee_frame` in all four envs was 33 mm off |
| C3 | gripper stroke is 45 mm | **89.1 mm** commanded, **~120 mm** forced open | pre-grasp's whole premise collapses |
| C9 | (unknown) | **TCP cannot go below ~44 mm** above the table | drawer handle sat at 26 mm — unreachable; task was impossible |
| C8 | per-env size via USD scale works | post-build scale **never reaches the collider** | the old grasp-width sweep is unproven |

The lesson that generalises: **a self-consistent harness proves nothing.** The slot probe
placed the block at its own computed TCP, so it looked correct under *any* offset. Validate
a measurement against something that does not share its assumptions.

---

## What is DONE

### Measurement (new this session, all with raw data in `logs/analysis/`)

| harness | answers |
|---|---|
| `scripts/analysis/gripper_stroke.py` | opening = `1.0035*(q_l+q_r) − 1.25 mm`, resid 0.035 mm; 89.1 mm commanded, ~120 mm forced; and that post-build USD scale is ignored by the collider |
| `scripts/analysis/tcp_floor.py` | TCP floor ~44 mm; usable band x ≈ 0.22–0.26, z ≈ 0.045–0.10 |
| `scripts/analysis/grasp_geometry.py` | grasp envelope vs object height (one process per height — see C8) |
| `scripts/challenge/pregrasp_probe.py` | the two-sided ungraspability test; **disproves the pre-grasp premise** |

Carried over and still good: C1 (no top-down grasp below z = 0.19 m, 0.00 % of table-height
voxels), C2 (USD finger drive 25× too weak; all envs override to 2000/40), C5–C7.

### Code

All four envs build, render, are registered, and now bind `ee_frame` to the measured
`mdp.TCP_OFFSET`. Do not touch `pick_place/` — its module-global `OBJECT_NAMES` and
hardcoded 2-object observation sit behind the existing 87.9 % result.

| env | smoke test | status |
|---|---|---|
| `Rebot-PrecisionSlot-{,Loose-,Tight-,Play-}v0` | `test_precision_slot_env.py` **PASS** | achievability **unproven** — probe needs re-tuning |
| `Rebot-PreGrasp-{,Play-}v0` | `test_pregrasp_env.py` **PASS** | **⚠ premise disproven — needs redesign** |
| `Rebot-ClutterExtract-{,Play-,Tight-}v0` | `test_clutter_env.py` **PASS** | topple threshold proven reachable |
| `Rebot-DrawerOrder-{,Play-}v0` | `test_drawer_env.py` **PASS** | handle raised onto a plinth; joint travel unverified |

Each smoke test collects failures in a list rather than asserting, so one run reports
everything, and each fires **at least three negative controls**.

Videos in `logs/videos/`, all four re-recorded this session. The drawer is now legible — it
needed a bound `UsdPreviewSurface` material (RTX ignores `displayColor`) and a side-on camera.

---

## What is NOT done, in priority order

### 1. Pre-grasp needs a new mechanism (or replacing)

Width cannot make anything ungraspable on this arm: the fingers force open to ~120 mm, and a
120 mm object spans the whole usable workspace. Measured — the lying 100 mm block is lifted
**100 %** of the time.

`docs/envs/pregrasp.md` lists the options. The most promising is **height, not width**: the
44 mm TCP floor is a hard barrier, so an object lying flat and ~25 mm tall genuinely cannot
be reached, while the same object stood on end (100 mm tall, 40 mm wide) is grasped 100 % of
the time. The open question is whether the gripper can *push* something that short, since the
same floor limits contact — that is the first thing to measure.

`pregrasp_probe.py` already implements the test harness this needs.

### 2. Re-tune and re-run the slot insertion probe

`PRE_X` and `GRIP_Z` were both tuned against the wrong TCP. At the old `GRIP_Z = 0.072` the
stroke now stalls at 20.7 mm (0 % inserted); at `--grip_z 0.084` it reports 68.8 % but the
block starts at 42 mm depth against a 40 mm threshold, so it measures a ~1 mm stroke. Fix the
start pose so the block begins just inside the mouth, then re-sweep grip height. The
`--grip_z` flag is already wired up.

### 3. Verify the drawer joint travels under a scripted pull

The joint is free (zero drive stiffness, 8.0 damping), the handle is now at z = 71 mm and
reachable, and the predicates all pass. Nobody has actually pulled it with the arm.

### 4. Clutter extraction at the tight end

Reset jitter produces free gaps of **7.0–18.8 mm** (measured). A scripted extraction at the
narrow end would close the achievability question.

### 5. Reward weights are untuned

**No training run has been done on any challenge env.** A short PPO run on
`Rebot-PrecisionSlot-v0` to confirm a learning curve exists is the cheapest sanity check.

### 6. Explicitly out of scope

The **end-to-end scripted expert**. Big Will has a separate pipeline for expert generation —
do not build one. (An agent was spawned for this in an earlier session and has been stopped.)

---

## Hard-won gotchas — read before debugging

1. **Finish every CEM/IK search before placing the object.** These searches evaluate
   candidates with `write_joint_state_to_sim`, which teleports the arm and re-opens the
   fingers hundreds of times. Searching for a lift path *after* closing silently drops the
   object, and the grasp reads as a slip. This one cost the most time of anything here.
2. **A retention test must be a real lift.** "Is the object within 80 mm of the TCP" is
   satisfied by an untouched object lying on the table under the gripper — it scored a 107 mm
   block in an 89 mm gripper at 100 %. Better still, check the **finger separation after
   closing**: it cannot be faked (`gap = 1.0035*(q_l+q_r) − 1.25 mm`).
3. **`ProxyArray`, not tensors.** Isaac Lab 3.0 accessors need `.torch`.
4. **Observation terms must return (N, 1)**, not (N,).
5. **PhysX articulations cannot contain kinematic links.** Anchor with a `FixedJoint`.
6. **RTX ignores `displayColor`.** Bind a `UsdPreviewSurface` or the asset renders flat white.
7. **`UsdGeom.BBoxCache` reads the USD xform hierarchy, which PhysX never writes back to.**
   It returns a constant bound at every joint value. Use physics body poses.
8. **Position-only IK is not a sufficient specification for this arm** — constrain the
   finger-separation axis or the wrist arrives unable to hold the object.
9. **Interpolating between two IK solutions in joint space does not keep the TCP straight.**
   Solve a Cartesian waypoint chain with a tight search radius instead.
10. **`scene.write_data_to_sim()` before every `sim.step()`** in standalone scripts. Better:
    drive motion through `env.step()` so the real action scaling and decimation apply.
11. **`PhysxCfg` comes from `isaaclab_physx.physics`**, assigned to `sim.physics`.
12. **Argparse names collide with SimulationApp** — `--width` and `--height` are both
    rejected; use `--obj_width`, `--block_h`.
13. Isaac asset server has only **4 YCB props**; all new geometry must be procedural.

---

## Suggested next session, in order

1. Measure whether the gripper can push an object shorter than the 44 mm TCP floor. That one
   number decides whether pre-grasp can be rescued or should be replaced.
2. Re-tune `slot_insertion_probe.py`'s start pose and grip height; re-establish the insertion
   rate under the corrected TCP.
3. Scripted drawer pull.
4. One short PPO run on `Rebot-PrecisionSlot-v0` (1024 envs) purely to confirm the reward
   produces a learning signal.
