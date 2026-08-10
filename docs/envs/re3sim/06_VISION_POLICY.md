# The pixels-only student — round 1, round 2, and where it actually breaks

Status at 2026-08-10: **1.6 % (round 1), 0.0 % (round 2 with DAgger)** against a 93.0 % expert.
Not solved. This file exists so the next attempt does not re-derive the eliminations.

## 1. What was built

| piece | where |
|---|---|
| vision shard recording | `eva_bc/re3sim/expert/collect_demos.py --shards` |
| student (2 cameras + 23-D proprio, no privileged input) | `eva_bc/re3sim/act/train_flow_vision.py` |
| launcher-free runtime (for DAgger) | `eva_bc/re3sim/act/policy_runner_vision.py` |
| evaluation + failure taxonomy | `eva_bc/re3sim/act/eval_flow_vision.py` |
| DAgger | `collect_demos.py --dagger-ckpt` |
| data audits | `inspect_shards.py`, `probe_wrist_blackout.py` |
| the task with cameras in the scene | `Rebot-Workstation-PickPlace1-Vision-v0` |

Datasets: `vision_r1` 238 episodes (93.0 % expert), `vision_d1` 231 DAgger episodes
(90.2 % expert from student-drifted starts). ~830 steps/episode, 160x120 per camera.

## 2. ⭐ Where it breaks — measured, not inferred

Hand the policy the episode part-way through, replaying the expert's recorded actions up to
that point (`scratchpad/handoff_probe.py`):

| policy takes over at | gripper at handoff | placed |
|---|---|---|
| 30 % | open, approaching | **0/3** |
| 55 % | open / just closed | **0/3** |
| 75 % | **closed, cube in hand** | **2/3** |

**The student can carry and place. It cannot grasp.** That is the whole finding. The box is
218 x 150 mm and tolerates centimetre error; closing on a 56 mm cube needs millimetres.

Precision check: action MSE against the expert is 0.013, and one action unit is 0.5 rad, so
that is ~0.057 rad ~= **3.3 deg of joint error per joint** -- centimetres at the gripper.

## 3. Eliminated, with the evidence (do not re-test these)

| hypothesis | test | verdict |
|---|---|---|
| the model did not learn | predict on training frames | **rejected** -- corr +0.874 with expert motion, MSE 0.013 vs 0.090 do-nothing baseline |
| copycat / causal confusion (`last_action` is in the obs, so echoing it is a low-loss do-nothing solution) | MSE to `last_action` | **rejected** -- 0.098, it is not echoing |
| the gripper head cannot close | predict on frames where the expert closes | **rejected** -- 41/48 correct, mean -0.702 |
| the images never reach the network | swap in another episode's frames / black | **rejected** -- output changes by up to 0.93 |
| generalisation from 238 episodes | evaluate on a TRAINING seed | **rejected** -- 0.0 % there too |
| padding-pruning broke the action stream | replay recorded actions into a fresh env | **rejected** -- 3/3 place the cube |
| the eval harness drives the env wrongly | same replay | **rejected** -- same replay reproduces the demo |

That last one is the important negative: **the data, the env and the evaluator are all
correct**, so what remains is the policy's own precision.

## 4. One real mismatch found, and it was not the cause

At t=0 the live env and the recorded shard agree to 4 decimals on joint positions, cube pose,
box pose and image means -- except `last_action`, which is **zero after `env.reset()`** and
equals the expert's hold command in every recorded first frame (`collect_demos.py` pre-rolls
120 steps holding `q_home` and never resets afterwards). In this action space zero means
"drive to the default pose", so the student was handed a false claim at step 0, and with
15-step chunk commitment that is enough to throw an episode.

`eval_flow_vision.py` now pre-rolls the hold command and does not reset afterwards. It changed
nothing (1.6 % -> 1.6 %, 0.0 % -> 0.0 %), but it is a genuine train/eval discrepancy and it
would have masked any later fix.

## 5. Why DAgger did not help

DAgger was aimed at compounding error, on the reasoning that the expert is **open-loop** -- it
solves a plan and replays it, so it never corrects and its demonstrations contain no
corrective behaviour. That reasoning still looks right in general, and the collection worked
(students drifted to 0.727 rad off default; the expert still scored 90.2 % planning from
there). But it moved nothing, because the binding constraint is not *which states* the student
visits -- it is that the student cannot hit a 56 mm target from any of them.

**Process note: the handoff sweep and the action-replay cost ~40 minutes and would have said
this before the ~4.5 hours DAgger cost.** Localise the failure before scaling the data.

## 6. Ranked suspects for the grasp precision

1. **The wrist camera goes blind during the descent.** 1.29 % of wrist frames are fully black
   overall but they cluster in deciles 4-6 -- the grasp -- affecting 78 of 238 episodes with
   runs up to 316 steps (6.3 s). Those samples are dropped from training, so the student has
   *less* supervision exactly where it needs most, and at rollout it is blind there too.
   The near plane is NOT the cause (replaying the pose renders content at 0.02 / 0.005 /
   0.001 m), so this is still undiagnosed.
2. **The workstation camera resolves the cube at 6.8 px** (8.2 mm/px at 1.07 m). Fine
   localisation has to come from the wrist camera, which sees 18-37 px -- and which is the one
   that goes black.
3. **Action representation.** Absolute joint targets require the policy to know both the
   current pose and the goal to millimetre accuracy; a residual/delta parameterisation is the
   usual answer.

## 7. Plan

1. Diagnose the wrist blackout properly (it is transient, not geometric).
2. Re-collect at higher camera resolution, wrist especially.
3. Only then more data / DAgger.

A realistic note on the 95 % target: DAgger's ceiling is the demonstrator, so 95 % needs the
expert above 95 % on the eval distribution (it is 96.1 % at jitter 0, 93.0 % at 0.15) *and*
near-perfect retention. The precedent on this codebase is `exp08`, where a pixels-only student
retained 87.5 % of a 91.4 % champion -- which would land here around 84 %.
