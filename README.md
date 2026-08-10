# reBot_RL — RS-rebot-dev-arm x Isaac Lab

Reinforcement-learning environments and training pipeline for the Seeed
**RS-rebot-dev-arm** (8 DOF: 6 revolute + 2-finger prismatic gripper) in
**Isaac Sim 6.0.1** / **Isaac Lab 3.0**. The flagship task is two-object
pick-and-place: grasp a scaled YCB soup can and sugar box from the table and drop
them into an open-top basket, using side-on grasps only (near-table top-down
grasps are kinematically impossible for this arm).

The repo contains:

- manager-based Isaac Lab environments for **lift-to-pose**, **vision (two-camera)
  lift**, and **pick-and-place** (v0 fixed basket, v1 randomized basket),
- **rl_games PPO** training and playback scripts,
- an honest **task-completion evaluator** for pick-and-place,
- a **scripted/hybrid expert** demo generator + behavior cloning (BC → rl_games
  checkpoint transplant for RL fine-tuning),
- a **teacher→student vision distillation** pipeline (DAgger-capable),
- the vendored robot USD asset (Git LFS) with its validation reports.

## Results

The best privileged-state PPO teacher reached **87.9% task success** on the
randomized pick-and-place evaluation (`scripts/evaluate_pick_place.py`: both
objects in the basket at episode end, nothing ever dropped, mid-episode
object-nudge perturbations on). Dense-shaping variants plateaued in the high-80s;
the project has since pivoted to a motion-planner → ACT → residual-RL pipeline,
which lives in the sibling `reBot_ACT/` repository. The v1 environment here
(`Rebot-PickPlace-v1`) is Stage 0 of that pipeline.

## Requirements

- Linux, NVIDIA GPU (developed on a single 12 GB GPU; reduce `--num_envs` if you
  have less memory)
- Isaac Sim 6.0.1 — the vendored USD asset needs Isaac Sim 6.0's physics parser;
  under 5.1 the articulation comes up with 0 DOFs
- Isaac Lab 3.0
- Git LFS (the robot USD, demo lookup tables, and basket asset are LFS-tracked)

## Installation

Reference setup (conda env `env_isaaclab6`, python 3.12, Isaac Lab 3.0 checked
out at `~/Desktop/isaacLab/IsaacLab`):

```bash
conda create -n env_isaaclab6 python=3.12 && conda activate env_isaaclab6
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --pre
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.3.1 scipy==1.17.0          # isaacsim-core pins these
cd ~/Desktop/isaacLab/IsaacLab && ./isaaclab.sh -i
cd <this repo> && python -m pip install -e source/reBot_RL
```

Isaac Lab 3.0 runs **headless by default** — add `--viz kit` to any command below
to get the interactive viewer (costs throughput, so leave it off for training).

## Registered tasks

| Task ID | Description |
|---|---|
| `Rebot-Lift-Cube-v0` | Lift-to-pose: grasp a cube, lift it, carry it to a randomized commanded pose. Privileged state obs (33-D), actions 7-D (6 joint positions + binary gripper). |
| `Rebot-Lift-Cube-Play-v0` | Same, 16 envs, observation noise off — for watching checkpoints. |
| `Rebot-Lift-Cube-Vision-v0` / `-Play-v0` | Lift task observed through two rig cameras (front + wrist) for the distillation student. |
| `Rebot-Lift-Cube-Vision-Rand-v0` / `-Play-v0` | Vision lift with visual domain randomization (lighting, materials, camera jitter). |
| `Rebot-PickPlace-v0` / `-Play-v0` | Two-object pick-and-place into a fixed basket. Scaled YCB soup can + sugar box spawn on the +y side, basket on the -y side; success = both objects inside at episode end. 39-D privileged obs, dense shaping + curriculum + mid-episode nudge perturbations. |
| `Rebot-PrecisionSlot-v0` / `-Loose-v0` / `-Tight-v0` / `-Play-v0` | **Challenge suite.** Grasp a block and slide it horizontally into a snug slot at 3.0 / 1.5 / 0.5 mm per-side clearance. Where the basket accepts ±50 mm and any orientation, this accepts ±1.5 mm and ±6.9°. 34-D obs, Factory-style multi-scale keypoint shaping. See [docs/envs/precision-slot.md](docs/envs/precision-slot.md). |
| `Rebot-ClutterExtract-v0` / `-Tight-v0` / `-Play-v0` | **Challenge suite.** Extract a target block from a row of four lighter distractors and set it down in a goal zone *without toppling a neighbour*. Free gaps measure 7–19 mm after jitter. The greedy reach-and-close action is frequently wrong. 42-D obs. See [docs/envs/clutter-extract.md](docs/envs/clutter-extract.md). |
| `Rebot-DrawerOrder-v0` / `-Play-v0` | **Challenge suite.** Pull a drawer open by its handle, *then* stow a block in it. The carry reward is hard-gated to exactly zero until the drawer is open, so no gradient bridges the two stages. Procedurally authored one-DOF cabinet. 33-D obs. See [docs/envs/drawer-order.md](docs/envs/drawer-order.md). |
| `Rebot-PreGrasp-v0` / `-Play-v0` | **Challenge suite — ⚠ premise disproven, needs redesign.** Intended as extrinsic dexterity (tip an ungraspable block up, then grasp it). Measurement showed the fingers force open to ~120 mm, so the start state is graspable after all. Builds and tests clean; see [docs/envs/pregrasp.md](docs/envs/pregrasp.md) for the numbers and redesign options. |
| `Rebot-Workstation-PickPlace1-v0` / `-Play-v0` | **Reconstructed real workstation.** Pick a 56 mm Rubik's cube off a photoreal 3D-Gaussian desk and place it in a measured cardboard box, among clutter. Geometry is *measured* (calipers + scale), appearance is a 3DGS reconstruction of an actual desk. 41-D privileged obs. See [docs/envs/re3sim/](docs/envs/re3sim/). |
| `Rebot-Workstation-PickPlace1-Vision-v0` / `-Vision-Play-v0` | ⭐ The same task **with the two cameras the real rig has** — a wrist D405 (measured intrinsics, feed-calibrated tilt) and a workstation camera 0.66 m in front of the base, 1.00 m up, 75° down. Also clones the gaussian desk per env, which any multi-env render needs. This is the task to train a pixels-only policy on. |
| `Rebot-Workstation-PickPlace1-Strict-v0` | Harder rung: disturbing the clutter ends the episode. Not for development — a 73.3 % expert scored 16.4 % once displacement was actually tested. |
| `Rebot-PickPlace-v1` / `-Play-v1` | v1: randomized movable basket (kinematic rigid body repositioned every reset), wider object spawns with lying-on-side probability, per-env object scale / mass / gain / start-pose diversity, 41-D obs (basket center xy appended), sparse rewards only. Stage 0 of the planner→ACT→residual pipeline. |

All tasks register an `rl_games_cfg_entry_point`; the lift task additionally has
an RSL-RL config.

## Quickstart — the reconstructed workstation

Everything this task needs is in the repo, including the 88 MB gaussian desk (Git LFS). From a
clean clone:

```bash
git lfs pull                                   # the desk, the robot USD, the lookup tables
python -m pip install -e source/reBot_RL
```

Look at the scene before training anything — the splats are visual-only, so *every* automated
check passes whether they are placed correctly or not rendering at all:

```bash
python -u scripts/render_workstation.py --enable_cameras --headless
python -u scripts/station_cam_tilt_sweep.py --enable_cameras --headless   # camera placement
```

State-based training sees no pixels and wants many envs:

```bash
python -u scripts/rl_games/train.py --task Rebot-Workstation-PickPlace1-v0 --headless --num_envs 1024
```

Pixels-only training uses the `-Vision-v0` ids, which put both cameras in the scene and give
every env its own desk. Rendering is bound by TOTAL render-product pixels across all envs, not
by resolution alone: 128 envs x 2 cameras x 160x120 is fine on 10 GB, 64 envs x 1 camera x
640x480 comes back blank.

```bash
python -u scripts/render_workstation.py --enable_cameras --headless \
    --task Rebot-Workstation-PickPlace1-Vision-Play-v0 --num_envs 8 --env 3
```

Environment variables that change the task:

| variable | default | effect |
|---|---|---|
| `RE3SIM_SPLATS_PER_ENV` | `0` | clone the gaussian desk per env. **Required for any multi-env render** on the non-vision tasks; the `-Vision-v0` ids set it themselves. |
| `RE3SIM_ARM_START_JITTER` | `0.0` | uniform per-joint jitter (radians) on the arm's start pose. Measured expert cost: 96.1 % at 0, 90.6 % at 0.15, 75.0 % at 0.30. |
| `RE3SIM_CUBE_PATTERN` | per-env random | pin every env to one of the 8 authored cube patterns. |
| `RE3SIM_SCANNED_CLUTTER` | `0` | use the photogrammetry clutter instead of analytic primitives. |
| `RE3SIM_CAM_W` / `RE3SIM_CAM_H` | `160` / `120` | camera resolution on the `-Vision-v0` tasks. Keep it 4:3 — the wrist intrinsics are calibrated at 640x480 and 16:9 changes the vertical FOV. |
| `RE3SIM_SPLATS` | — | point at a different gaussian field, for A/B. |

The scripted expert, demo collection and the BC / DAgger stack live in the sibling `eva_bc`
repo (`eva_bc/re3sim/`).

## Training

```bash
# pick-and-place (rl_games PPO; logs + checkpoints under logs/rl_games/rebot_pick_place/)
python scripts/rl_games/train.py --task Rebot-PickPlace-v0 --num_envs 4096

# lift task
python scripts/rl_games/train.py --task Rebot-Lift-Cube-v0 --num_envs 4096

# fewer envs if the GPU runs out of memory (batch sizes still divide evenly)
python scripts/rl_games/train.py --task Rebot-PickPlace-v0 --num_envs 2048

# resume / fine-tune from a checkpoint (e.g. a BC-initialized one), optionally with W&B
python scripts/rl_games/train.py --task Rebot-PickPlace-v0 --checkpoint <ckpt>.pth --track

# watch a trained checkpoint
python scripts/rl_games/play.py --task Rebot-PickPlace-Play-v0 --viz kit \
    --checkpoint logs/rl_games/rebot_pick_place/<run>/nn/<ckpt>.pth

# training curves (per-term episode rewards under Episode_Reward/...)
tensorboard --logdir logs/rl_games/rebot_pick_place
```

## Evaluation

`scripts/evaluate_pick_place.py` is the honest gate metric: fixed-length episodes
(no time-out auto-reset racing), success = both objects placed at the end AND no
object ever dropped; reports per-object placed rates, drop rate, and mean time to
completion. Perturbations stay on unless `--no_perturb`.

```bash
python scripts/evaluate_pick_place.py --headless \
    --checkpoint logs/rl_games/rebot_pick_place/<run>/nn/<ckpt>.pth
```

`scripts/probe_pick_place_policy.py` diagnoses *where* a policy fails (does the
carried object cross the basket footprint? at what height? does the gripper ever
open there?).

## Scripted expert + behavior cloning

`scripts/scripted_expert/` generates demonstrations with a hybrid expert — the RL
policy reaches and grasps (its proven primitive), a scripted position-IK sequence
carries, lowers, and releases over the basket — or with a pure-kinematic grasp
state machine (`--grasp_mode kinematic`). Actions are recorded in the env's
native action space; only fully successful episodes are saved.

```bash
python scripts/scripted_expert/generate_pick_place.py --headless ...   # demos → data/pick_place_demos/<run>/
python scripts/scripted_expert/train_bc.py --data_dir data/pick_place_demos/<run>
python scripts/scripted_expert/bc_to_rlgames.py --bc_checkpoint logs/bc/<run>/best.pt --out <rlgames_init>.pth
```

`bc_to_rlgames.py` transplants the BC trunk/mu-head/obs-normalizer into an
rl_games checkpoint template so the cloned policy can be RL-fine-tuned in place.
The small lookup tables the expert uses (`data/pick_place_demos/grasp_table.pt`,
`carry_waypoints.pt`) ship with the repo; the demo shards themselves are
gitignored.

## Vision distillation (teacher → student)

`scripts/distillation/` distills the state-based lift teacher into a two-camera
vision student:

```bash
# roll the teacher with cameras + visual randomization, save episode shards
python scripts/distillation/collect_episodes.py --checkpoint <teacher>.pth --episodes 500
# behavior-clone the student (pure PyTorch, no Isaac Sim needed)
python scripts/distillation/train_student.py --data_dir data/distillation/<timestamp>
# evaluate in sim; close the DAgger loop with collect_episodes.py --student_checkpoint
python scripts/distillation/play_student.py --checkpoint logs/distillation/<run>/best.pt
```

## Challenge suite

A second family of tasks, each isolating a manipulation skill the pick-and-place task does
not exercise. Design rationale, the measured hardware envelope every task respects, and the
per-task validation ladder are in [docs/CHALLENGE_SUITE.md](docs/CHALLENGE_SUITE.md);
per-env reference docs are in [docs/envs/](docs/envs/README.md).

| env | skill under test | status |
|---|---|---|
| `Rebot-PrecisionSlot-*` | tight-tolerance horizontal insertion | tests pass; achievability probe needs re-tuning |
| `Rebot-ClutterExtract-*` | constrained retrieval without disturbing neighbours | tests pass; topple threshold proven reachable |
| `Rebot-DrawerOrder-*` | articulated joint + irreversible precedence | tests pass; scripted pull unverified |
| `Rebot-PreGrasp-*` | non-prehensile reconfiguration | ⚠ premise disproven, needs redesign |

### Measured constraints — read these before designing anything for this arm

Every one of these was *believed* before it was measured, and every one broke something.
Raw data lands in `logs/analysis/`.

| constraint | value | harness |
|---|---|---|
| **No top-down grasp below z = 0.19 m** | 0.00 % of table-height voxels admit a finger axis within 26° of vertical (819k configs) | `analysis/reachability_map.py` |
| **TCP is 41.9 mm behind `gripper_end`** — *not* the 75 mm used elsewhere in this repo | a 33 mm error; makes any scripted grasp close on air | see below |
| **Gripper opens 89.1 mm on command, ~120 mm if forced** | `_GRIPPER_OPEN = 0.045` is a *per-finger* joint value and both fingers move | `analysis/gripper_stroke.py` |
| **The TCP cannot go below ~44 mm above the table** | the gripper bottoms out on the table first; usable band x ≈ 0.22–0.26 m | `analysis/tcp_floor.py` |
| **A post-build USD `xformOp:scale` never reaches the collider** | only the render mesh scales | `analysis/gripper_stroke.py` |
| **The USD authors the finger drive at 100 N/m** | caps squeeze at ~1.7 N; challenge envs override to 2000 N/m | `analysis/gripper_stiffness_sweep.py` |

The TCP figure is the one to internalise: with the fingers shut, the two finger-body origins
coincide, and *that* point is the grasp point. A constant offset error is nearly invisible in
a reach reward — the policy just learns a shifted target — but it is fatal to any scripted
grasp. Challenge envs bind their `ee_frame` to `mdp.TCP_OFFSET`; the older lift/pick-place
envs still use the inherited `-0.075` and were left untouched.

```bash
# measurement harnesses
python scripts/analysis/reachability_map.py --num_envs 2048 --batches 400   # workspace map
python scripts/analysis/query_reachability.py                               # query it
python scripts/analysis/gripper_stroke.py                                   # opening + TCP calibration
python scripts/analysis/tcp_floor.py                                        # how low the TCP can go
python scripts/analysis/grasp_geometry.py --block_h 0.07                    # grasp envelope vs height

# achievability probes
python scripts/challenge/slot_insertion_probe.py --grip_z 0.084
python scripts/challenge/pregrasp_probe.py --num_envs 96

# assets and visualisation
python scripts/challenge/author_drawer_usd.py                               # (re)author the cabinet
python scripts/challenge/record_env_video.py --task Rebot-DrawerOrder-Play-v0 \
    --cam_eye 0.10 -0.40 0.28 --cam_target 0.245 0.0 0.075
```

## Smoke tests

Every challenge test collects failures in a list rather than asserting, so one run reports
everything, and each fires at least three negative controls.

```bash
python scripts/test_precision_slot_env.py   # geometry, success predicate + 4 negative controls
python scripts/test_clutter_env.py          # row pitch, topple constraint + 3 negative controls
python scripts/test_drawer_env.py           # articulation, handle reach, precedence gate + 4 controls
python scripts/test_pregrasp_env.py         # shadow width, uprighting gradient + 3 negative controls

python scripts/test_pick_place_env.py       # v0 env: shapes, predicates, terminations, rewards
python scripts/test_pick_place_env_v1.py    # v1 env: 41-D obs, randomized basket, spawn geometry
python scripts/author_basket_usd.py         # (re)author the movable basket USD (plain pxr, no sim)
```

## Repository layout

```
scripts/
  rl_games/train.py, play.py         rl_games PPO training / playback
  evaluate_pick_place.py             task-completion evaluator (gate metric)
  probe_pick_place_policy.py         failure-mode diagnostics
  scripted_expert/                   hybrid-expert demos, BC, BC→rl_games transplant
  distillation/                      teacher→student vision distillation (+DAgger)
  analysis/                          hardware measurement harnesses (workspace, gripper, TCP)
  challenge/                         achievability probes, asset authoring, video recording
  test_pick_place_env*.py            env smoke tests
  test_{precision_slot,clutter,drawer,pregrasp}_env.py   challenge env smoke tests
  author_basket_usd.py               one-off basket USD authoring
source/reBot_RL/                     pip-installable Isaac Lab extension
  reBot_RL/assets/rebot_arm.py       arm articulation config
  reBot_RL/tasks/manager_based/
    lift/                            lift-to-pose env + agent configs
    lift_vision/                     two-camera lift env + visual randomization
    pick_place/                      pick-and-place v0 + v1 envs, mdp terms
    challenge/                       challenge suite envs + their own mdp/ (kept separate
                                     from pick_place/mdp on purpose — see docs)
  data/RS-rebot-dev-arm/             vendored robot USD asset (LFS) + validation docs
  data/basket/basket.usda            movable basket asset for v1
  data/drawer/drawer.usda            procedurally authored cabinet for the drawer task
docs/CHALLENGE_SUITE.md              measured hardware envelope + design rationale
docs/envs/                           one reference doc per environment
docs/HANDOFF.md                      current state, open questions, gotchas
data/pick_place_demos/               expert lookup tables + demo metadata
```

## Things worth knowing about this arm

- **Drive gains:** the USD authors angular gains per *degree*; Isaac Lab actuator
  configs are per *radian*. The asset config (`reBot_RL/assets/rebot_arm.py`)
  deliberately leaves stiffness/damping `None` so the parser's converted values
  survive. Never copy the raw USD numbers into an actuator config — that
  de-scales every revolute gain 57.3x.
- **Home pose:** the authored home pose hangs the gripper 0.135 m *below* the
  base. Any scene that mounts the base on a surface must override the start pose
  (these envs do).
- **No top-down grasps at table level:** the wrist pitch chain (j2/j3/j4) cannot
  point the fingers down next to the table, so policies grasp side-on. Fingers
  extend 0.089 m along `gripper_end` local **-X**.
- **The true TCP offset is (-0.0419, 0, 0)**, measured: with the fingers shut the two
  finger-body origins coincide, and that point is the grasp point. The `(-0.075, 0, 0)`
  used by the lift and pick-place envs is **33 mm too far forward**. Those envs were left
  as they are — a constant offset just shifts the target their policies learned, and
  changing it would invalidate the 87.9 % result — but do not carry that number into
  anything new, and never into a scripted grasp. Challenge envs use `mdp.TCP_OFFSET`.
- **The gripper opens 89.1 mm on command, ~120 mm if forced.** `_GRIPPER_OPEN = 0.045` is a
  per-finger prismatic joint value, not a stroke, and both fingers move.
- **The TCP cannot be placed below ~44 mm above the table** — the gripper bottoms out first.
  Any fixture feature the gripper must touch has to sit above that.
- **Quaternions are (x, y, z, w)** in Isaac Lab 3.0 — identity is `(0, 0, 0, 1)`.
  A 2.x-style `(1, 0, 0, 0)` silently flips 180° about X.

The vendored asset in `source/reBot_RL/data/RS-rebot-dev-arm/` is used
unmodified; see its `VALIDATION.md` / `ISAAC_SIM_CODE.md` for upstream
provenance.

## Related work

Pure-RL training on this task is paused at 87.9%. Active development continues in
the sibling **`reBot_ACT/`** project: a cuRobo motion-planner expert generates
demonstrations in the v1 env, an ACT policy is trained on them, refined with
DAgger, and finished with residual RL.
