# `Rebot-PrecisionSlot-*` — tight-tolerance horizontal insertion

**Skill under test:** sub-centimetre alignment and a committed straight-line insertion.

**Source:** `source/reBot_RL/reBot_RL/tasks/manager_based/challenge/precision_slot_env_cfg.py`
and the `challenge/mdp/` package.

## Gym IDs

| ID | clearance (per side) | envs | note |
|---|---|---|---|
| `Rebot-PrecisionSlot-Loose-v0` | 3.0 mm | 1024 | easy rung |
| `Rebot-PrecisionSlot-v0` | **1.5 mm** | 1024 | default |
| `Rebot-PrecisionSlot-Tight-v0` | 0.5 mm | 1024 | matches IndustReal's tightest shipped peg |
| `Rebot-PrecisionSlot-Play-v0` | 1.5 mm | 16 | for watching checkpoints |

## Video

| clip | file |
|---|---|
| settle, 4 envs tiled (1.5 mm) | `logs/videos/Rebot-PrecisionSlot-Play-v0_settle.mp4` |
| settle, 4 envs tiled (3.0 mm) | `logs/videos/Rebot-PrecisionSlot-Loose-v0_settle.mp4` |
| settle, 4 envs tiled (0.5 mm) | `logs/videos/Rebot-PrecisionSlot-Tight-v0_settle.mp4` |
| random actions, workspace sweep | `logs/videos/Rebot-PrecisionSlot-Play-v0_random.mp4` |

Regenerate any of them with:

```
python scripts/challenge/record_env_video.py --task Rebot-PrecisionSlot-Play-v0
python scripts/challenge/record_env_video.py --task Rebot-PrecisionSlot-Tight-v0 --mode random
```

What to look for: the block stands upright, the slot walls are visibly **shorter than the
block** (that is deliberate — see the kinematics note below), and the red back stop marks
the far end of the bore.

Visual inspection of the recorded clips confirms: the block settles upright with no
interpenetration; the slot channel runs along +x with the red back stop at its far end, so
the mouth faces the robot as designed; and the block spawns on the −y side, clear of the
fixture and inside the graspable envelope.

> The repo's stock `WORKSPACE_CAM_CFG` pose was framed for the lift/pick-place scene and
> puts this fixture entirely out of shot. `record_env_video.py` therefore re-aims the
> camera with a look-at (`--cam_eye` / `--cam_target`) rather than using the cfg pose.

## Why the task looks the way it does

Two measured facts about this arm, not design preferences. Full data in
[../CHALLENGE_SUITE.md](../CHALLENGE_SUITE.md).

1. **There is no top-down grasp below z = 0.19 m.** Across 819,200 sampled joint
   configurations, **0.00 %** of table-height voxels admit a finger axis within 26 deg of
   vertical; the steepest approach available anywhere in that band is 42.3 deg off
   vertical. So the insertion cannot be a vertical peg-in-hole the way Factory, RLBench and
   ManiSkill formulate it. It is a **horizontal** slide along +x.
2. **The walls must be shorter than the block.** The gripper holds the block's upper half
   while its lower half sits between the walls. If the walls were block-height the fingers
   would collide with them, and (because of fact 1) the arm cannot release from above to
   avoid that.

The point of the task is the tolerance. The existing pick-and-place basket accepts the
object anywhere in a ±50 mm footprint with no orientation constraint; this slot accepts
±1.5 mm and ±6.9 deg. ManiSkill2's own ablation is the cleanest evidence that this is the
axis that matters: with task, policy and objects all held fixed, `PegInsertionSide` goes
from **0.01 to 0.74** when clearance is widened 10×.

## Scene

All values env-local, metres. Robot base and table top at the origin.

| entity | prim | geometry | pose |
|---|---|---|---|
| robot | `{ENV_REGEX_NS}/Robot` | RS-rebot-dev-arm, 6R + 2 prismatic fingers | base at origin, `_START_POSE` |
| block | `{ENV_REGEX_NS}/Block` | cuboid **45 × 30 × 70 mm**, 0.04 kg | `(0.220, −0.130, 0.035)` upright |
| slot floor | `{ENV_REGEX_NS}/SlotFloor` | `(0.070, outer, 0.020)` | `(0.245, 0, 0.010)` |
| slot wall +y | `{ENV_REGEX_NS}/SlotWallPY` | `(0.070, 0.008, 0.030)` | `y = +(half + 0.004)` |
| slot wall −y | `{ENV_REGEX_NS}/SlotWallNY` | `(0.070, 0.008, 0.030)` | `y = −(half + 0.004)` |
| back stop | `{ENV_REGEX_NS}/SlotBack` | `(0.008, outer, 0.030)` | `x = 0.284` |
| table | `{ENV_REGEX_NS}/Table` | SeattleLabTable USD | `(0.5, 0, 0)`, yaw 180° |

`half = 0.015 + clearance`; `outer = 2 × (half + 0.008)`.

Derived slot landmarks: **mouth plane x = 0.210**, **back stop face x = 0.280**, centreline
`y = 0`, floor top `z = 0.020`.

The 30 mm graspable width sits inside the measured 26–42 mm grasp sweet spot, and 0.04 kg
is inside the payload the finger drive can actually hold. The spawn point is at r ≈ 0.26,
azimuth ≈ −31 deg — inside the r ≤ 0.32 envelope the repo's trained policies actually
grasped in.

**Single source of truth:** the scene builds its collision cuboids from the same constants
in `challenge/mdp/common.py` that the success predicate reads, so geometry and predicate
cannot drift apart. (The v0/v1 basket does not have this property — its dimensions live
both in the env cfg and in a separately generated USD.)

## Actions — 7-D

| term | type | dim | detail |
|---|---|---|---|
| `arm_action` | `JointPositionActionCfg` | 6 | `joint[1-6]`, `scale=0.5`, `use_default_offset=True` |
| `gripper_action` | `BinaryJointPositionActionCfg` | 1 | `< 0` closes, `>= 0` opens; open 0.045 m |

Commanded joint target = `default_joint_pos + 0.5 × action`, so
`action = (q_desired − q_default) / 0.5`.

## Observations — 34-D

| # | term | dim | slice | function |
|---|---|---|---|---|
| 1 | `joint_pos` | 8 | `0:8` | `mdp.joint_pos_rel` |
| 2 | `joint_vel` | 8 | `8:16` | `mdp.joint_vel_rel` |
| 3 | `block_pose` | 7 | `16:23` | block pos+quat in the robot root frame |
| 4 | `slot_error` | 4 | `23:27` | `(depth, lateral, yaw, inserted)` in the slot frame |
| 5 | `actions` | 7 | `27:34` | `mdp.last_action` |

Term 4 is the task-frame error the policy actually has to null — the same idea as Factory's
`fingertip_pos_rel_fixed`, and why a 21-dim observation suffices there.

## Rewards

Factory's multi-scale squashed-keypoint recipe (`1/(exp(a·d) + b + exp(−a·d))`) rather than
a hand-tuned ladder. The three length scales *are* the curriculum — coarse pulls the block
in from anywhere, fine only pays inside the last millimetres. A single `tanh` cannot express
both, which is why the pick-place shaping does not transfer to a tolerance task.

| term | weight | a, b | what it does |
|---|---|---|---|
| `reaching` | 2.0 | std 0.1 | get the gripper to the block at all |
| `lifting` | 8.0 | — | block above 0.045 m **and** within 0.08 m of the TCP |
| `kp_baseline` | 6.0 | 5, 4 | coarse pull toward the slot |
| `kp_coarse` | 12.0 | 50, 2 | alignment, ~1 cm scale |
| `kp_fine` | 20.0 | 300, 0 | last-millimetre term |
| `engaged` | 15.0 | — | nose between the walls (depth ≥ 10 mm) |
| `inserted` | 60.0 | — | sparse success, **per step** |
| `dropping_penalty` | −30.0 | — | on `block_dropped` |
| `toppling_penalty` | −10.0 | — | on `block_toppled` |
| `action_rate` | −2e-2 | — | smoothness |
| `joint_vel` | −5e-3 | — | smoothness |

Keypoints are the block's four **corners**, not Factory's collinear ones: Factory locks
roll and pitch out of its action space, this arm does not, so the reward has to see all six
DoF of error or a rolled block reads as perfectly placed.

`inserted` pays every step the block *stays* home rather than terminating on success —
terminating makes a fast sloppy insert reward-rate-optimal, which is a failure this repo
already hit on the lift task.

## Terminations

| term | trigger |
|---|---|
| `time_out` | 12 s (600 policy steps) |
| `block_dropped` | block z < −0.05 |
| `block_toppled` | block's own +z axis has world-z < 0.6 |

## Events

| term | mode | detail |
|---|---|---|
| `reset_all` | reset | `reset_scene_to_default` |
| `reset_block` | reset | x ±0.02, y ±0.03, **yaw ±0.35 rad** |
| `block_material` | startup | static friction 0.9–1.1, dynamic 0.75–0.95, 16 buckets |

Yaw randomization is safe here (unlike in `pick_place`) because the block is authored
Z-up, so a body-frame yaw delta is a genuine yaw rather than a topple.

## Success criterion

`mdp.is_inserted`: depth ≥ **40 mm** past the mouth (block centre x ≥ 0.250) **and**
lateral offset within the slot half-width **and** |yaw| ≤ **0.12 rad (6.9 deg)** **and**
above the slot floor.

## Sim settings

`dt = 1/400`, `decimation = 8` (50 Hz policy), `episode_length_s = 12`, `num_envs = 1024`,
PhysX with `gpu_max_rigid_patch_count = 2**19` and
`friction_correlation_distance = 0.00625`. Block colliders use `contact_offset = 0.002`,
`rest_offset = 0.0`.

**Finger drive override.** The single `.*` actuator group is split in two. The revolute
joints keep `stiffness = damping = None` so Isaac Lab preserves the USD's validated gains
(hardcoding the raw USD numbers de-scales every revolute gain 57.3×). The **fingers are
overridden to stiffness 2000 / damping 40**, because the USD authors them at 100 N/m, which
caps the squeeze at ~1.7 N and the payload at ~0.05 kg — see `CHALLENGE_SUITE.md` C2.

## Running it

```
python scripts/test_precision_slot_env.py                                  # smoke test
python scripts/test_precision_slot_env.py --task Rebot-PrecisionSlot-Tight-v0 --num_envs 8
python scripts/challenge/record_env_video.py --task Rebot-PrecisionSlot-Play-v0
python scripts/rl_games/train.py --task Rebot-PrecisionSlot-v0 --num_envs 1024
```

## Validation evidence

`scripts/test_precision_slot_env.py` **passes on all three clearance variants.**

**V1 — geometry read back off the USD stage** with `UsdGeom.BBoxCache`:

| variant | authored gap | block | measured clearance |
|---|---|---|---|
| Loose | 36.000 mm | 30.000 mm | **3.000 mm** |
| default | 33.000 mm | 30.000 mm | **1.500 mm** |
| Tight | 31.000 mm | 30.000 mm | **0.500 mm** |

**V3 — physics:** block settles upright (z ∈ (0.020, 0.050), up-axis > 0.95) with no
interpenetration.

**V5 — the predicate has teeth.** Fires 16/16 at the home pose (depth 40.0 mm) and stays
off for four negative controls: laterally offset, over-yawed, barely entered, still on the
table.

And the geometry itself rejects misalignment. A block force-teleported in at 0.6 rad of
yaw and allowed to settle relaxes to:

| clearance | residual yaw | vs. the 0.12 rad tolerance |
|---|---|---|
| 3.0 mm | 0.433 rad | far outside |
| 1.5 mm | 0.154 rad | outside |
| 0.5 mm | 0.000 rad | forced perfectly square |

That monotonic progression is the evidence that clearance is a real physical difficulty
axis rather than a number in a config: the tight slot physically cannot accept a yawed
block, so a misaligned policy is *rejected* rather than funnelled. It is the opposite of a
basket.

**V6 — MDP plumbing:** 34-D observation, all terms finite, every reward term finite, both
termination terms fire on their trigger states.

**V4 — achievability: the insertion stroke. Superseded by the TCP correction, see below.**
`scripts/challenge/slot_insertion_probe.py` isolates the insertion from the grasp: the
block starts held in the closed gripper resting on the slot floor with its nose just inside
the mouth (±2 mm of lateral scatter so it is not one lucky alignment), and the arm then
executes *only* the insertion stroke, open-loop, in joint space.

No IK solver is used. The two joint configurations — at the mouth and fully inserted — are
found by cross-entropy search over the 6 arm joints, scored by forward kinematics evaluated
in the sim itself. That cannot silently converge on an unreachable pose, because the
achieved TCP error is measured and reported.

> ### ⚠ These figures are superseded and must be re-measured.
>
> They were produced with `TCP_OFFSET = (-0.075, 0, 0)`. The true tool centre point is
> **41.9 mm** behind `gripper_end`, not 75 mm (`CHALLENGE_SUITE.md` C10). The probe was
> *self-consistent* under the wrong value — it placed the block at its own computed TCP, so
> it would have looked fine for any offset — which is precisely why the error survived.
>
> With the corrected offset the arm reaches the same TCP target in a different posture, with
> the wrist 33 mm deeper, and the fingers now clip the slot wall tops. Re-running at the old
> `GRIP_Z = 0.072` stalls the stroke at 20.7 mm of depth (insert rate 0 %). Raising the grip
> to `--grip_z 0.084` recovers 68.8 %, but in that run the block *starts* at 42 mm of depth
> against a 40 mm success threshold, so the stroke it measures is ~1 mm long and the number
> means nothing.
>
> **What is still solid:** the block is held 64/64 after the gripper closes, and the CEM
> reaches its targets to 0.13–2.1 mm. What needs redoing is `PRE_X` and `GRIP_Z`, both of
> which were tuned against the wrong TCP. Until then this env's achievability is *unproven*,
> not disproven.

Historical figures, for reference only:

| quantity | 512-env run | 32-env run (video) |
|---|---|---|
| TCP error, pre-insertion pose | 0.17 mm | 0.22 mm |
| TCP error, inserted pose | 0.05 mm | 0.46 mm |
| finger-axis misalignment | 6e-5 | 1e-4 |
| block held after gripper close | 253/512 | 30/32 |
| inserted | 284/512 (55.5 %) | 26/32 (81.3 %) |
| mean final depth | 31.0 mm | 40.6 mm |

Video of the stroke: `logs/videos/slot_insertion_probe.mp4`.

**Two real defects this probe caught**, both worth recording because each produced a
confident-looking wrong answer first:

1. *Block placed floating.* The slot floor exists only inside the slot (x ∈ [0.210, 0.280]).
   A start pose outside the mouth left the block hovering 20 mm above the table; it fell
   during the settle and the gripper closed on empty air. Measured signature: the block
   finished at depth −56 mm, having moved *backwards* 26 mm.
2. *Position-only IK.* The first CEM optimised TCP position alone, leaving wrist
   orientation free. The fingers then closed along the block's 45 mm length instead of
   across its 30 mm width and squirted it out of the slot. Measured signature: 34 mm
   backwards travel and 23 mm of lateral error during the close, with insert rate 0.0.
   Constraining the finger-separation axis (`gripper_left` − `gripper_right`) to lie along
   world y took the insert rate from **0 % to 55–81 %**.

Defect 2 is the substantive one for anyone building on this: **a position-only IK target is
not a sufficient specification for this arm.** The grasp axis has to be constrained
explicitly or the wrist will arrive in an orientation that cannot hold the object.

**Also passes:** `scripts/test_precision_slot_env.py` still passes end to end after the TCP
correction — slot gap 33.000 mm against a 30.000 mm block (1.500 mm per side), the success
predicate fires 16/16 at the home pose, and all four negative controls reject. A block forced
in at 0.60 rad relaxes to 0.122 rad, so the walls really do reject yaw.

**Still outstanding:** re-tuning the insertion probe, and the full grasp-carry-insert expert. The probe establishes that the
geometry and the insertion are executable; it does not yet establish that the *approach and
grasp from the table spawn* can be scripted. So the env is validated as correctly specified
and physically solvable at the insertion stage, and the end-to-end success rate is not yet
measured.

## Known limitations

- The reachability map behind the design is pure kinematics (no table, no collision), so it
  is an *upper* bound. A pose it marks unreachable is definitively unreachable; one it marks
  reachable may still be blocked.
- 2.5 kg payloads are not certified in sim even with the stiffness override; see
  `CHALLENGE_SUITE.md` C2.
- Reward weights are transplanted from Factory's published recipe and this repo's
  pick-place equilibrium; they have not yet been tuned against a training run.
