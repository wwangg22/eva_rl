# `Rebot-ClutterExtract-v0` — constrained retrieval from clutter

**Skill under test:** collision-aware approach and gentle contact, where the correct first
action is frequently *not* a grasp.

**Source:** `challenge/clutter_env_cfg.py`, `challenge/mdp/clutter.py`
**Gym IDs:** `Rebot-ClutterExtract-v0` (2048), `-Play-v0` (16), `-Tight-v0` (harder pitch),
`-Lenient-v0` (the pre-2026-08-03 rule, for re-running old baselines only)
**Video:** `logs/videos/Rebot-ClutterExtract-Play-v0_settle.mp4`
**Test:** `scripts/test_clutter_env.py` — **passes**, four negative controls

## ⚠ The success constraint was tightened on 2026-08-03

The task used to require only that no neighbour **toppled** (`up_z < 0.75`, ≈41° of tilt). A
neighbour dragged the length of the table and set down upright passed that predicate cleanly.
It now also requires that no neighbour **moved** more than `DISTURB_TOL = 2 mm`.

This is not a cosmetic tightening. Measured on 768 held-out episodes of a scripted expert:

| | topple-only | + 2 mm displacement |
|---|---|---|
| scripted expert success | **73.3 %** | **16.4 %** |

The median episode the old rule scored as a success displaced a neighbour by **13.7 mm** —
more than the row's own 12 mm free gap — and in **22–25 % of all episodes a neighbour was
carried into the goal zone along with the target** (p90 of the displacement ≈ 206 mm, against
120–277 mm from the row to the goal).

The quantity that catches this was already being computed on every step: `distractors_disturbed`
was wired to a shaping reward and to nothing else. Full account, including how it went
unnoticed through ~5 000 scored episodes: `eva_bc/clutter/docs/15_STRICT_METRIC.md`.

`DISTURB_TOL` is calibrated, not assumed. Under a null action for a full 700-step episode the
blocks drift by **1 µm** (768 episodes, P35), so the threshold sits **≈2 000×** above the
solver's own noise; the free gap is 12 mm, so it also sits well below "nudged but still in its
slot".

## The task

A blue target block sits in the middle of a row of four lighter, identical distractors.
Extract it and set it down in the green goal zone **without disturbing any neighbour** —
moving any distractor more than 2 mm, or toppling it, terminates the episode with a −40
penalty.

| variant | row pitch | free gap between 30 mm blocks |
|---|---|---|
| `Rebot-ClutterExtract-v0` | 42 mm | **12 mm** |
| `Rebot-ClutterExtract-Tight-v0` | 36 mm | **6 mm** |

## Why this is a different skill

In the basket task the greedy reach-and-close action is *always* correct. Here it often
is not: the free gap is a fraction of the gripper's outer width when open (the fingers alone
separate to 89 mm on command, docs C3), so the policy must either thread the fingers in
precisely or push a neighbour aside first — a non-greedy action whose payoff is entirely
deferred. That is a credit-assignment structure plain pick-and-place never poses.

The distractors are deliberately **lighter than the target** (0.025 kg vs 0.04 kg) so they
topple readily and the no-topple constraint actually bites, rather than being satisfied by
accident.

Reference points: VPG (Zeng et al. 2018) measured **82.7 %** completion for a policy that
learns pushing and grasping jointly against **40.6 %** for grasping alone — on a UR5 with a
parallel jaw, the same gripper class as this arm. ManiSkill's `PickClutterYCB` scores
**0.00–0.08** for challenge entrants, so the axis has substantial headroom.

## Scene

| entity | geometry | pose |
|---|---|---|
| target | cuboid **36 × 30 × 70 mm**, 0.04 kg, blue | `(0.250, 0.0, 0.035)` |
| distractor 0–3 | same shape, **0.025 kg**, grey | y = ∓84 mm, ∓42 mm from the target |
| goal marker | cylinder r = 45 mm, **visual only** | `(0.185, −0.185)` |

The goal marker has no collider on purpose — a collider there would fight the block being
set down on it.

Row radius r ≈ 0.25 keeps every block inside the measured graspable envelope
(r ≤ 0.32, docs C4).

## MDP

**Actions — 7-D**, as all challenge envs.

**Observations — 42-D**

| # | term | dim | slice |
|---|---|---|---|
| 1 | `joint_pos` | 8 | `0:8` |
| 2 | `joint_vel` | 8 | `8:16` |
| 3 | `target_pose` (root frame) | 7 | `16:23` |
| 4 | `clutter` — per distractor: (dx, dy relative to target, up-axis) | 12 | `23:35` |
| 5 | `actions` | 7 | `35:42` |

**Rewards**

| term | weight | note |
|---|---|---|
| `reaching` | 2.0 | std 0.10 |
| `lifting` | 8.0 | target above 0.055 m and near the TCP |
| `extracted` | 15.0 | target clear of the row (z > 0.090) **and** nothing toppled |
| `carrying` | 12.0 | gated: only pays once the target is above 0.070 m |
| `success` | 60.0 | sparse, per step, in the goal zone with the row undisturbed |
| `disturbance` | **−3.0** | *summed* planar displacement of the distractors — "be gentle" |
| `topple_penalty` | **−40.0** | on termination |
| `disturb_penalty` | **−40.0** | on termination |
| `action_rate` / `joint_vel` | −2e-2 / −5e-3 | |

`disturbance` stays continuous and stays a *sum*: it is the dense gradient toward gentleness,
and it has to pay before the 2 mm cliff is reached to be any use. The constraint itself uses
the **max** over the four blocks instead — "no neighbour was moved" must not be satisfiable by
nudging four blocks 0.5 mm each.

`disturb_penalty` is not optional. `distractor_disturbed` ends the episode, and ending an
episode early is worth a large *positive* amount to an agent accumulating negative shaping
terms — without the matching penalty, adding the termination would have made shoving a
neighbour the profitable move.

**Terminations:** `time_out` (14 s), `target_dropped`, **`distractor_toppled`** (any
distractor's up-axis below 0.75), **`distractor_disturbed`** (any distractor more than 2 mm
from its spawn). In practice the disturbance term fires first essentially always — a block
must slide before it can tip — so `distractor_toppled` now reads **0.0 %** on expert rollouts
that previously reported ~20 % topples.

**Events:** the target and all four distractors are jittered *independently* on reset — a
uniformly-spaced row is a much easier problem than one with an uneven, sometimes-tighter
gap. A final `record_spawn` event stores the settled distractor positions, which is what
`disturbance` measures against; it must run last.

## Validation status

`scripts/test_clutter_env.py` — **passes**:

| check | result |
|---|---|
| 42-D observation concatenates, finite; `clutter_obs` is `(N, 12)` | ✓ |
| row settles on the table, all upright | ✓ |
| **measured** free gap between adjacent 30 mm blocks after reset jitter | **7.0 – 18.8 mm** |
| nothing toppled at reset; disturbance 0.00 mm | ✓ |
| **`TOPPLE_DOT = 0.75` is reachable** — a distractor laid on its side | toppled 16/16 |
| a distractor shoved 30 mm but left upright | **not** toppled, disturbance 33.9 mm |
| **`DISTURB_TOL` is silent at reset** | max displacement 0.000 mm |
| **a 1 mm nudge (below tol)** | **not** disturbed |
| **a 4 mm shove, still upright** | disturbed 16/16, toppled 0/16 |
| `target_extracted` for a target lifted clear | ✓ |
| `target_at_goal` for a target set down in the zone | 16/16 |
| (a) at the goal in xy but still held high | rejected |
| (b) set down outside the goal radius | rejected |
| **(c) perfectly at the goal but a distractor is down** | rejected — the topple constraint binds success |
| **(d) perfectly at the goal but an upright distractor dragged 85 mm with it** | rejected — the disturbance constraint binds success |
| nothing terminates on the first step after reset | ✓ |

Control **(d) is the one this suite used to be missing**, and its absence is why the metric
error survived ~5 000 scored episodes. The row above it — "a distractor shoved 30 mm but left
upright: **not** toppled" — was written as a *passing* check, and it is a correct statement
about `any_distractor_toppled`. It is silent on whether toppling was the right constraint,
and nothing in the suite asked.

The measured gap range also shows the per-block reset jitter is doing real work: episodes
range from a comfortable 18.8 mm to a 5.6 mm squeeze, so difficulty varies rather than being
fixed at the nominal 12 mm.

**Verified since:** a scripted expert extracts the target on **16.4 %** of held-out episodes
under the 2 mm rule (768 episodes, `eva_bc/clutter`), so the task is achievable but not
comfortably — the residual failure is that the finger blades sweep the neighbours during the
close, in 83.6 % of attempts.
