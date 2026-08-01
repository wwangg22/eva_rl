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
out at `~/Desktop/isaacLab/IsaacLab-3.0`):

```bash
conda create -n env_isaaclab6 python=3.12 && conda activate env_isaaclab6
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --pre
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.3.1 scipy==1.17.0          # isaacsim-core pins these
cd ~/Desktop/isaacLab/IsaacLab-3.0 && ./isaaclab.sh -i
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
| `Rebot-PickPlace-v1` / `-Play-v1` | v1: randomized movable basket (kinematic rigid body repositioned every reset), wider object spawns with lying-on-side probability, per-env object scale / mass / gain / start-pose diversity, 41-D obs (basket center xy appended), sparse rewards only. Stage 0 of the planner→ACT→residual pipeline. |

All tasks register an `rl_games_cfg_entry_point`; the lift task additionally has
an RSL-RL config.

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

## Smoke tests

```bash
python scripts/test_pick_place_env.py --headless      # v0 env: shapes, predicates, terminations, rewards
python scripts/test_pick_place_env_v1.py --headless   # v1 env: 41-D obs, randomized basket, spawn geometry
python scripts/author_basket_usd.py                   # (re)author the movable basket USD (plain pxr, no sim)
```

## Repository layout

```
scripts/
  rl_games/train.py, play.py         rl_games PPO training / playback
  evaluate_pick_place.py             task-completion evaluator (gate metric)
  probe_pick_place_policy.py         failure-mode diagnostics
  scripted_expert/                   hybrid-expert demos, BC, BC→rl_games transplant
  distillation/                      teacher→student vision distillation (+DAgger)
  test_pick_place_env*.py            env smoke tests
  author_basket_usd.py               one-off basket USD authoring
source/reBot_RL/                     pip-installable Isaac Lab extension
  reBot_RL/assets/rebot_arm.py       arm articulation config
  reBot_RL/tasks/manager_based/
    lift/                            lift-to-pose env + agent configs
    lift_vision/                     two-camera lift env + visual randomization
    pick_place/                      pick-and-place v0 + v1 envs, mdp terms
  data/RS-rebot-dev-arm/             vendored robot USD asset (LFS) + validation docs
  data/basket/basket.usda            movable basket asset for v1
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
  extend 0.089 m along `gripper_end` local **-X**; the TCP frame sits at offset
  (-0.075, 0, 0).
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
