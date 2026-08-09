# `workstation_pickandplace1` — expert → BC → RL

Everything **after** the environment exists. The env itself is planned in
[../01_STEP1_PLAN.md](../01_STEP1_PLAN.md).

| doc | covers |
|---|---|
| [01_POST_ENV_PLAN.md](01_POST_ENV_PLAN.md) | the staged ladder S1–S8, with the gate each stage must pass |
| [LESSONS_INHERITED.md](LESSONS_INHERITED.md) | the measured mistakes from `eva_bc` and `eva_rl` we are not allowed to repeat |

**Task.** Three reconstructed objects (rubix cube, roll of tape, tape measure) spawn at
random reachable poses with random yaw on the user's reconstructed desk. A stationary open
box sits at a random reachable location. Place all three into the box.

**Pipeline.** Scripted expert → flow-matching chunk BC → batched sim eval → grasp-success
bit → x0-steering RL. Optionally then teacher→student vision distillation, which is the
whole reason for building a photoreal env in the first place.

**Status:** planning. Nothing built.

## The one-paragraph version of why this plan looks like it does

`eva_bc` ran this exact ladder on a 2-can pick-and-place task and left a full lab notebook.
The short version: BC plateaued at **64.1 %** pooled, additive residual RL came out **exactly
flat** (55.5 % → 55.5 %), and **x0-steering took it to 91.4 %**. The diagnosis was that the
base's failures were *wrong mode choices*, not imprecise execution — the flow head already
contained successful behaviour for ~91 % of spawns and what was missing was a
state-conditioned selector. So this plan goes straight to x0-steering and keeps the additive
residual only as a cheap ablation. It also front-loads three things that cost `eva_bc` weeks:
seed discipline, bit-exact wrapper gates, and a strengthened success predicate.
