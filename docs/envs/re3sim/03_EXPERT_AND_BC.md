# 03 — Scripted expert → demonstrations → flow-matching BC

**Started 2026-08-06.** Live document; numbers are filled in as they are measured, and every
number says which env it was measured in (primitive colliders vs reconstructed meshes), because
those are **not the same task** and their success rates are not comparable.

---

## 1. The manoeuvre

```
home ──(action-driven)──► pre-grasp ──► grasp @ z=25 mm ──► CLOSE ──► lift ──► carry ──► release ──► retreat
      fingers open                                       70 steps    fingers closed        fingers open
```

Constants, each with the measurement that set it:

| | value | why |
|---|---|---|
| grip height | **25 mm** | p01 measured 99 % held here; 5 % at 20 mm, 77 % at 30 mm — a genuine optimum, not a plateau |
| opening axis | nearest cube **face normal**, both candidates tried | a parallel jaw closing on a cube *corner* rotates it instead of squeezing |
| approach axis | **unconstrained** | pinning it horizontal collapsed `o_align` to 0.40–0.69 and failed p01's positive control |
| floor penalty | `floor_z = 0.002` | **not** the clutter expert's 0.012 — a 56 mm cube gripped at 25 mm needs finger bodies at z ≈ 5 mm, and a 12 mm floor penalises the exact pose the grasp requires |
| pose gate | `pos ≤ 1.5 mm`, `o_align ≥ 0.90`, `low_z ≥ 2 mm`, `pen = 0` | `o_align ≥ 0.99` (clutter's gate) is unattainable at this grip height |
| carry height | 175 mm | rim 93 + cube half 28 + margin |
| waypoint spacing | ≤ 30 mm Cartesian | clutter P17: joint-space lines between independently solved waypoints swept 108 mm off their Cartesian path; dense local solves took success 18 % → 62.5 % |
| ordering | **solve everything before closing** | `cem`/`refine` teleport the arm and re-open the fingers; searching after the close silently drops the cube and reads as a slip. Worth 0 % → 100 % once already |

---

## 2. Run 1 — the expert works, the *planner* is the bottleneck

First execution ever, 32 envs, primitive colliders:

```
[batch 0] planning: 17/32 solved in 51s
[batch 0] SUCCESS 10/32 = 31.2%   (planned 17/32)
    taxonomy: plan-failed  15 | never-got-there   7 | lifted-but-lost   0 | over-box-not-inside   0
```

**Read the taxonomy, not the headline.** 31.2 % looks bad. But *conditional on a plan being
found*, the expert succeeded **10/17 = 59 %**, and — the important part — **`lifted-but-lost`
and `over-box-not-inside` are both zero.** Nothing was dropped in the carry and nothing was
released in the wrong place. The whole loss is upstream of execution: 15 envs the planner
refused, and 7 more it planned but never reached the cube in.

This is the failure taxonomy earning its keep on its first run. A success rate alone would
have sent me tuning the carry height or the finger stiffness, neither of which is broken.

### 2.1 Why the planner refused — a cost-function imbalance

Every rejection was `o_align`, and by a *small* margin:

```
[plan] grasp gate FAILED: pos 0.74 mm, o_align 0.850, low_z  6.7 mm
[plan] grasp gate FAILED: pos 0.99 mm, o_align 0.899, low_z  0.0 mm      <-- 0.001 short
[plan] grasp gate FAILED: pos 0.30 mm, o_align 0.761, low_z 18.2 mm
```

Position was never the problem — 0.3–1.1 mm against a 1.5 mm tolerance. The cause is in
`_kin.cem`'s cost:

```python
c = w_pos * ((tcp - pos).norm() - pos_tol).clamp(min=0)   # w_pos = 200, pos_tol = 1 mm
c = c + w_o * (1 - |o_hat · o_des|)                       # w_o   = 0.25
```

At `w_o = 0.25`, **losing the opening axis entirely costs 0.25**, which the position hinge
matches after 1.25 mm. So the search will happily give up the whole jaw orientation to buy a
fifth of a millimetre it did not need. The defaults are the clutter task's, where a 70 mm
block is far more forgiving of a rolled wrist.

### 2.2 The four changes

1. **Re-price orientation:** `w_o` 0.25 → **4.0**. A drop to `o_align = 0.90` now costs 0.40.
2. **Re-price position to match:** `w_pos` 200 → **600**. These two are only meaningful against
   each other — at the stock `w_pos`, `w_o = 4.0` would let the search spend the entire
   position budget (`POS_TOL − pos_tol` = 0.5 mm) to buy 0.025 of `o_align`, trading one gate
   failure for a different one. At 600, alignment can buy about a millimetre of slack and no
   more, which is the balance the hinge was designed for.
3. **Call `refine`.** The first version stopped at the CEM. `_kin.refine` finite-differences
   one Jacobian at a pose the CEM has already proved executable and drives
   `[tcp ; k_rot·o_hat]` at the target — precisely the pair the gate tests. Kept only when it
   improves the gate margin, since DLS near a rank-deficient configuration can walk away.
4. **Search both opening axes and both grip heights.** A cube is symmetric under 90°, so
   `yaw` and `yaw + π/2` are *both* valid face-normal grasps — and they are not equally
   **reachable**, because they demand different wrist rolls at the same TCP. The radial
   heuristic picks one; there was no reason to stop there. `z = 30 mm` (p01's 77 % cell) is
   tried only after both axes fail at the 99 % cell.

Candidates are ranked by **gate margin** — the worst of the three normalised margins — rather
than by `cost`. `cost` mixes in the avoid and floor penalties, so a beautifully aligned pose
4 mm off target can outrank one missing `o_align` by 0.01.

---

## 3. The defect that would have wasted the training run

`collect_demos.py` teleported the arm to the pre-grasp pose and started recording there.
`env.reset()` leaves the arm at its **default pose**, so every recorded episode began in a
state the environment never produces, and the policy would have met an unseen observation at
step 0 of every evaluation — reading as "BC does not transfer" rather than as a data bug.

Upstream had already measured this and its price (clutter P29): an action-driven approach from
the reset pose scored **73.0 % against a 74.2 % teleport baseline** — 1.2 points, and it is
what ships, because the teleport number is not a number about a deployable policy.

Fixed: `plan_episode` now solves a `home` segment — a *Cartesian* chain from the home TCP to
the pre-grasp TCP, densified and clutter-avoiding — and the collector drives it with actions.
The arm is still teleported once, back to `q_arm0` after planning, because `cem`/`refine`
scatter it across the workspace; that teleport restores the state `reset` produced rather than
inventing a new one. Object positions are restored at the same time, since the planning
teleports sweep the arm through them.

Solved as a Cartesian chain rather than a joint-space line for the P17 reason: independently
solved endpoints can sit in different IK branches, and the straight joint path between them
left the Cartesian route by 108 mm.

---

## 3a. ⭐ The real bottleneck: CEM solves poses the arm cannot *hold*

Adding the home approach dropped success from 31 % to **0 %**, with the taxonomy reading
`never-got-there 23`. Rather than guess, I instrumented each phase with the achieved TCP
against the target the planner had recorded — and the numbers pointed somewhere I had not been
looking.

| phase | TCP error vs plan (median) | p90 |
|---|---|---|
| after `home` | 69.0 mm | 185 mm |
| after `home` + 60 more held steps | 61.2 mm | 178 mm |
| after `grasp` | 53.4 mm | 123 mm |
| after `close` (70 steps holding still) | 46.2 mm | 124 mm |
| after `lift` | **8.0 mm** | 59 mm |
| after `carry` | **1.3 mm** | 62 mm |

Two readings of that table are wrong and one is right.

* *"Tracking lag."* No — 60 extra steps at a fixed command bought 8 mm, and 70 more bought
  another 7. A lag converges when the command stops moving.
* *"Blocked by the table."* I added lowest-body-z to check, and it read 0.0 mm in every
  phase — including the carry, where the TCP is 175 mm up and lands within 1.3 mm of target.
  That was **my probe being wrong**, not evidence: it took the minimum over *all* bodies,
  which includes the base plate that sits on the desk by design. `_kin.floor_bodies` restricts
  to `link6`/`gripper_*` for exactly this reason.

The right reading is the **asymmetry**. Low, far poses are missed by tens of millimetres;
high, retracted ones are hit to a millimetre. Isolating it settled the matter — teleporting
the arm directly onto a solved pre-grasp and stepping, it settles **19.8 mm away from the pose
it was placed on**.

**This is a position drive holding against gravity.** Drive torque is stiffness × error, so a
pose that requires torque can only be held *with* an error, and a low extended reach over the
table requires the most torque of anything in the manoeuvre.

**`cem` is structurally blind to this.** It evaluates candidates through
`write_joint_state_to_sim`, which places the arm kinematically — no gravity, no contact. Every
pose it returns is reachable in that sense; some of them are simply not **holdable**. The
existing gates (`pos_err`, `o_align`, `low_z`, `pen`) are all computed from that same
gravity-free FK, so none of them can catch it either. This is the same family of mistake as
the two probe bugs in §2 of [HANDOFF.md](HANDOFF.md): trusting a search's own report of a pose
instead of measuring the pose the robot actually reaches.

**Fix: command past the target by the steady-state error**, so the drive's own error lands the
arm *on* the target. An integrator finds that offset per joint without anyone modelling the
arm:

```python
a = kin.act(q + bias, close)          # what is actually submitted, and what is recorded
env.step(a)
bias = (bias + 0.15 * (q - q_now)).clamp(-0.25, 0.25)
```

The recorded action is the **compensated** one. That is correct rather than a compromise: it
is what was submitted to `env.step`, and it is what a policy must emit to hold the same pose.
The clamp is anti-windup — `q - q_now` also contains ordinary tracking lag while a waypoint is
being traversed — and 0.25 rad sits well inside the ±0.5 rad the action encoding spans before
`|a|` exceeds 1.

### Isolation table

Run at 32 envs, primitive colliders, same spawn seed:

| variant | planned | success | conditional |
|---|---|---|---|
| original planner, teleport to pre-grasp | 17/32 | 10/32 | 59 % |
| re-priced planner, action-driven home | 23/32 | **0/32** | 0 % |
| re-priced planner, teleport to pre-grasp | 23/32 | 11/32 | 48 % |

The re-pricing did what it was meant to (17 → 23 planned) and the home approach is where the
success went. Note also that the re-priced planner's *conditional* rate is slightly lower —
consistent with it now admitting harder layouts that were previously refused outright.

---

## 3b. ⚠ THE OPEN PROBLEM: the home→pre-grasp transit

**Status: unsolved after eight measured variants. This is the top item for the next session,
and everything needed to continue is below.**

The task decomposes cleanly into a *transit* (reset pose → pre-grasp, free space) and a
*manoeuvre* (pre-grasp → grasp → lift → carry → release). **The manoeuvre works.** Across
every run in this section the failure taxonomy reads `lifted-but-lost 0` and
`over-box-not-inside 0` — nothing is ever dropped in the carry and nothing is ever released in
the wrong place. Only the transit fails, and it fails by leaving the arm somewhere the
descent cannot recover from.

### What was tried, and what each one measured

| # | transit | planned | dips rejected | end-of-transit error | success |
|---|---|---|---|---|---|
| 1 | Cartesian CEM chain, home → pre-grasp | 23/32 | — | 60.8 mm, plateaus | 0/32 |
| 2 | as 1, + 60 extra held steps | 23/32 | — | 61.2 mm | 0/32 |
| 3 | as 1, + steady-state bias compensator | 23/32 | — | 60.8 mm | 1/32 |
| 4 | one joint-space line, home → pre-grasp | **2/32** | 30 | **4.5 mm** ✅ | 0/32 |
| 5 | joint line via a separately-solved `q_high` | 12/32 | 4 | 56 mm, **+53 mm ABOVE** | 0/32 |
| 6 | as 5, base-yaw first then shape | 4/32 | 18 | 103 mm | 0/32 |
| 7 | backwards Cartesian chain out to the home TCP | 10/32 | **0** ✅ | 101 mm | 0/32 |
| 8 | as 7 but forward and orientation-free, + wrist twist at height | 14/32 | 5 | 87 mm | 2/32 |

Five things are now **established** rather than suspected:

1. **A joint-space line is followable and a Cartesian CEM chain is not.** Variant 4 tracked to
   **4.5 mm** where variant 1 plateaued at 60.8 mm. Independently-solved Cartesian waypoints
   hop IK branches, and the commanded joint target then jumps faster than the arm can move.
2. **A joint-space line between two high poses is not itself high.** Variants 4–6 dipped to
   **−51 … −86 mm** — the wrist swings clean under the table — even though both endpoints sit
   at z ≈ 165 mm. Interpolating a redundant chain moves through configurations, not positions.
3. **Solving the transit backwards from the grasp fixes the dip completely** (variant 7,
   `dips = 0`) — but carrying the grasp's `o_des` out to the home TCP forces a wrist twist that
   lands in a different branch from the reset pose. Measured start hops: **2.15–2.52 rad**,
   which rejected every surviving plan.
4. **The chains do solve.** Variant 8 plus a `_chain` solve-error gate reports `chain = 0`
   rejections at a 10 mm tolerance, so `q_high` really does achieve its Cartesian target. The
   arm nonetheless settles ~98 mm below it.
5. **It is a converged steady state, not a lag.** Adding 40 held steps at the transit pose
   changed the result *bit for bit* (87.0 mm, dz −97.8 mm, identical to three decimal places).
   The arm parks at z ≈ 72–84 mm and can reach neither the transit pose above it nor the grasp
   pose below it.

### RESOLVED: it is (A), and the fix is a plan-time bias — not a runtime integrator

The saturation test settles it. Instrumenting the compensator's peak and raising its clamp
from ±0.25 rad to ±0.80 rad (gain 0.15 → 0.35):

| | bias ±0.25, gain 0.15 | bias ±0.80, gain 0.35 |
|---|---|---|
| peak bias reached | (unmeasured) | **0.800 rad — SATURATED** |
| error after `grasp` | 73.8 mm | **33.5 mm** |
| error after `close` | 60.2 mm | **9.1 mm** |
| error after `carry` | 1.2 mm | 0.8 mm |
| cube z at close | 28.0 mm (untouched) | **11.5 mm — crushed** |
| success | 2/32 | 2/32 |

So: **the arm can reach these poses, if commanded hard enough.** That is hypothesis (A) —
a position drive that can only hold a torque-demanding pose *with* an error — and the
compensator was never allowed to correct it, saturating at both clamps tried.

It also shows why a *runtime integrator* is the wrong instrument. At ±0.80 it overshoots into
the object: the cube ends at z = 11.5 mm instead of 28 mm, i.e. the gripper is driving it into
the desk. An integrator tuned to converge fast enough during a 30-step descent is necessarily
tuned to overshoot once it arrives.

**The fix to implement next is a plan-time bias, converged per pose:** teleport onto the
solved pose, drive it with `hold_phys` for a few dozen steps, read the achieved joints, and
take `bias = q_target − q_achieved`; iterate twice. That measures the exact steady-state
offset for that pose with no dynamics to tune, and `_kin` already has every piece
(`teleport_arm`, `hold_phys`, the joint read-back). Planning already teleports freely, so it
costs nothing that is not already being paid.

The remaining ~112 mm at the *transit* pose is the same phenomenon at the most
torque-demanding point of the trajectory and should fall out of the same fix.

### The two hypotheses as they stood before that test

Point 5 is the one to attack, and it has exactly two explanations left:

* **(A) the drive cannot hold the pose.** Argues for it: the arm holds the *carry* pose to
  1.2 mm and the *lift* pose to 4.9 mm, both high and retracted, while missing low or extended
  ones by tens of millimetres — the signature of a position drive whose torque is
  stiffness × error. Argues against it: 98 mm is a very large sag, and the bias compensator
  (§3a) should have absorbed it unless it saturated at its ±0.25 rad clamp.
* **(B) the arm is obstructed at z ≈ 80 mm.** Argues for it: the parking height is suspiciously
  close to the box rim (93 mm), and the desk collider was changed to a 1.60 × 1.00 m analytic
  slab when the splats landed — larger than the stock table it replaced, so the arm may now
  collide where it previously swung freely.

**The experiment that separates them takes one run:** teleport the arm onto `q_high`, hold,
and measure. If it sags there, it is (A); if it holds, the transit *path* is at fault and (B)
is live. Then log `bias.abs().max()` to see whether the compensator is saturating, and re-run
with `REBOT_WORKSTATION_PRIMITIVES=1` (which restores the stock table) to test (B) directly.

Both are cheap. Neither was run tonight because the reconstruction track is the priority the
user set, and because a wrong guess here costs a full re-measurement of every number below.

### One more measured result: the runtime compensator was making things *worse*

Turning the runtime integrator off entirely, at 128 envs on the demo-start protocol:

| | batch 0 | batch 1 |
|---|---|---|
| with runtime bias (±0.25, gain 0.15) | 16/128 = **12.5 %** | — |
| **without any runtime bias** | 42/128 = **32.8 %** | 41/128 = **32.0 %** |

Conditional on a plan being found: **53 % and 58 %**. So the integrator was not merely failing
to help, it was costing about twenty points — it perturbs poses the arm was already hitting in
order to chase ones it was not. Both bias mechanisms are now **off by default**; the plan-time
one is kept behind `SETTLE_BIAS=1` because it addresses the right cause and is where the next
session should resume.

### What is being trained in the meantime, and its exact limitation

Demonstrations are generated with `--teleport-pregrasp`, which starts each episode at the
**pre-grasp** rather than at the reset pose. **This is stated plainly because it matters:**
such a policy learns the grasp-and-place manoeuvre and *not* the approach, so it must be
evaluated on the same protocol and is **not deployable from a cold reset**. The flag's own
help text says so, and `eval_flow.py` reports the protocol it used.

This is a smaller claim than "a trained pick-and-place policy", and it is the honest one. The
manoeuvre it covers is the part every measurement in this document says already works.

**Both numbers are reported, deliberately.** `eval_flow.py` is run twice:

* `--match-demo-start` — starts each episode at the expert's pre-grasp, as the demonstrations
  do. This measures the **grasp-and-place manoeuvre**, against an expert reference of 32.4 %
  on the identical protocol.
* no flag — starts from the env's own `reset()`. This is the **deployable** number, and this
  checkpoint is expected to read very low because it has never seen that state.

**The gap between the two IS the transit problem, quantified.** Reporting only the first would
overclaim; reporting only the second would throw away the evidence that the BC stack works at
all. The printed header of each run says which protocol it used.

---

## 4. The BC stack

Ported near-verbatim from `eva_bc/clutter/act/`, because every hyper-parameter in it was paid
for by a measurement:

* **rectified flow, not a CVAE** — `x_τ = (1−τ)x₀ + τx₁`, target `v = x₁ − x₀`, 10 Euler steps
  at inference. The flow noise is the stochasticity source, so `use_vae` is off.
* **chunk 50, commit 15.** Chunk commitment is load-bearing, not a tunable: 59.4 / 32.8 / 3.1 /
  0 / 0 % at `n_action_steps` 15 / 8 / 4 / 2 / 1 (EXP02).
* **temporal ensembling OFF** — stale pre-event chunks suppress corrections.
* **lr 1e-4**, an order above ACT's 1e-5; flow/π₀-style heads want it.

### 4.1 The 41-D observation split

```
[ 0:16]  observation.state              joint_pos_rel(8) + joint_vel_rel(8)
[16:41]  observation.environment_state  target_pose(7) + box_pose(4) + clutter(6)
                                        + placed(1) + last_action(7)
```

**A trap worth naming:** this is `16 + 25 + 7`, which is *numerically identical* to
pick-place's split. A pick-place checkpoint therefore loads into this task without a murmur
and simply behaves badly, which reads as a training problem rather than as the wrong file.
`policy_runner.load_checkpoint` asserts on a `task` tag, not on the widths.

`compute_stats` floors every std at 1e-4. In a pool filtered to successes, the `placed` channel
is very nearly constant and `last_action[6]` nearly so; an unfloored std divides to infinity
and produces a NaN loss on step 1.

### 4.2 Evaluation

`re3sim/act/eval_flow.py`, buckets latched **before** each env's first termination because
`env.step` auto-resets a done env and everything read after that call describes a freshly
re-spawned scene:

| bucket | meaning |
|---|---|
| `success` | `placed_mask` ever true — the headline |
| `lifted` | cube ever above 60 mm — did the grasp take? |
| `over_box` | `over_box` ever true — did the carry arrive? |
| `near_miss` | lifted **and** over the box, never placed → release/settle problem |
| `carried_astray` | lifted but never over the box → carry problem |
| `no_lift` | never lifted → grasp problem |

Plus clutter displacement, separately: `placed_mask` deliberately does not ask whether the arm
shoved the tape measure across the desk on the way, and an episode that does is not the same
quality of success as one that does not.

Judged as **retention of the expert's own rate on the same protocol**, not against an absolute
number — the absolute figure moves with the expert and says nothing alone.

---

## 4a. ⭐ MEASURED: the BC stack works

**896 episodes collected, 278 successful (31.0 %)** — consistent across seven batches
(32.8 / 32.0 / 29.7 / 35.2 / 32.0 / 33.6 / 21.9 %). Trained 60 k steps in 19 min, final loss
0.059, 23.3 M parameters, 137,634 samples from 278 demos of 488–508 steps each.

Evaluated on **three held-out spawn seeds** (88000–88002 × 128 envs = 384 episodes; collection
used seeds 1000–1006, so there is no overlap), on **both** protocols:

| | matched protocol (demo start) | cold reset |
|---|---|---|
| seed 88000 | 21.1 % | 0.0 % |
| seed 88001 | 24.2 % | 0.0 % |
| seed 88002 | 20.3 % | 0.0 % |
| **pooled** | **21.9 %** (84/384) | **0.0 %** (0/384) |
| vs the 32.4 % expert | **retains 68 % — ON TREND** | — |
| `no_lift` | 55.2 % | 87.5 % |
| `carried_astray` | 19.5 % | 12.5 % |
| `near_miss` | 3.4 % | 0.0 % |
| `dropped` | 4.9 % | 0.0 % |

**The BC port is healthy.** 68 % retention of a state-observation expert is squarely in the
normal band, the three seeds agree to ±2 points, and `success` and `success_final` are
identical (delta +0.00), so nothing is being flattered by the latch.

**The cold-reset column is the transit problem, priced.** 21.9 → 0.0 is the entire cost of
demonstrations that begin somewhere `env.reset()` never puts the arm. Note the policy does
still *lift* the cube in 11.7–14.1 % of cold episodes — it has learned something transferable
about grasping — but it never once carries it to the box.

### One caveat that the headline hides, and it is not small

Among the episodes the matched protocol calls SUCCESS, the worst clutter body has moved a
**median of 33.9 mm** from its spawn (p90 87.5 mm, max 501.7 mm). Tightening success to
require the clutter to stay put:

| clutter must stay within | success |
|---|---|
| (unconstrained — the headline) | 21.9 % |
| 10 mm | 7.3 % |
| 5 mm | 4.7 % |
| 2 mm | 3.6 % |

So roughly two thirds of the successes are achieved partly by **bulldozing**. `placed_mask`
deliberately does not ask about this, which is exactly why `eval_flow.py` reports it
separately — and it is what the `-Strict-v0` variant exists to train against once the default
task is solved. It is also consistent with the clutter task's own history: a 73.3 % expert
there scored 16.4 % once displacement was actually tested.

---

## 4b. ⭐ MEASURED on the FINAL photoreal env — better on every axis

Everything in §4a was measured against primitive colliders and a desk that was 90° out of
place. Re-run against the finished environment (reconstructed cube / tape roll / tape measure
colliders, reconstructed desk, yaw corrected, grip height re-measured at 32 mm):

| | primitive env | **photoreal env** |
|---|---|---|
| demonstrations | 896 episodes, 278 good | **1024 episodes, 323 good** |
| expert success | 31.0 % | **31.5 %** |
| planner solve rate | 63–80 / 128 | **79–94 / 128** |
| BC pooled, matched protocol | 21.9 % | **24.7 %** (95/384) |
| **retention of the expert** | 68 % | **77 % — ON TREND** |
| clutter displacement, median over successes | 33.9 mm | **6.55 mm** |
| BC pooled, cold reset | 0.0 % | 2.3 % |

Three things worth drawing out.

**The collider swap is free once the grip height is re-measured.** Swapping in the
reconstructed cube alone cost 31 % → 19.5 %; re-sweeping `GRIP_Z` (25 → 32 mm) took it back to
31.5 %. The lesson is not "reconstructed meshes are fine" but that **a constant measured on a
different object is an assumption**, and this one was load-bearing.

**The planner got materially happier** — 62–73 % of layouts solved, against 49–63 % before.
The reconstructed cube is slightly smaller than the nominal 56 mm and the grasp sits higher,
so more spawn poses are reachable.

**The clutter displacement collapsed 5×**, from a 33.9 mm median to 6.55 mm. The policy is no
longer buying its successes by shoving the scene around, which is the difference between a
number and a usable behaviour — and it is what makes the `-Strict-v0` variant plausible next.

The cold-reset column moved 0.0 % → 2.3 %, which is real but is not a solution: the transit
(§3b) is still the one thing between this and a policy that runs from a cold `env.reset()`,
which is what a physical rollout needs.

---

## 4c. ⭐ THE TRANSIT IS SOLVED — and it was never the drive

Nine variants and a day of physics hypotheses, and the answer was three characters of
`reversed()`.

### How it was finally caught

By reporting the achieved TCP against **two** references at the same instant — the Cartesian
target, and the forward kinematics of the joint pose actually commanded:

| compared against | error |
|---|---|
| `high` — the Cartesian target | **101.6 mm** (dz −107.0) |
| `high_fk` — FK of the commanded pose | **7.5 mm** |

The arm was going exactly where it was sent. **The plan was sending it 107 mm too low.** Every
"the position drive cannot hold this pose" hypothesis was chasing a planner bug.

### The bug

```python
up_pts = list(reversed(_densify(grip, pre))) + list(reversed(_densify(pre, high)))
```

`_densify(a, b)` already runs *a → b*, so marching outward from `q_grip` the order is simply
grip → pre → high. Reversing each half — copied from an earlier version that solved a single
leg backwards — put `high` in the **middle** of the chain and left `q_up[-1]` near the
pre-grasp. The `CHAIN_TOL` gate never fired because the chain *does* reach every waypoint it is
given; it was given them in the wrong order.

End-of-transit error **101.6 mm → 1.0 mm**. Planner solve rate 35/64 → 48/64.

### What it had been hiding: the grip height was measured twice, both times on the wrong thing

With an honest transit the real sag appeared cleanly — ~35 mm over the 140 mm descent — and it
is compensable simply by commanding higher. Re-swept end to end, from the env's own reset:

| grip z [mm] | 32 | 40 | 48 | **56** | 64 |
|---|---|---|---|---|---|
| success | 0.0 % | 7.8 % | 25.0 % | **40.6 %** | 40.6 % |
| planned /64 | 45 | 55 | 60 | 59 | 58 |

**56 mm is above the cube's own 54.5 mm top, and that is the point:** the *settled* tool centre
is what lands on the cube, not the commanded one. On a position-driven arm, commanding the
height you want is commanding the wrong thing.

Both earlier values were honest measurements of the wrong thing:

* **25 mm** — `p01_grasp_feasibility.py`, on an **analytic** cube;
* **32 mm** — a sweep using `--teleport-pregrasp`, which under the buggy ordering put the arm
  effectively **at** the cube, so it measured a grasp with no descent and therefore no sag.

**Generalisable:** a protocol that skips part of the manoeuvre does not measure a constant of
the manoeuvre. Both sweeps were careful, reproducible, and answered a question nobody had
asked.

### The expert now runs from a cold reset, and better than the shortcut ever did

| | teleport shortcut | **end-to-end from `env.reset()`** |
|---|---|---|
| expert success | 31.5 % | **31.9 %** (327/1024) |
| planner solve rate | 79–94 /128 | **107–126 /128** |

The two-number hedge in §4a is gone: there is one protocol now, and it is the deployable one.

### A third measurement that was of the wrong thing: the episode horizon

The first end-to-end evaluation scored **3.4 %**, with `no_lift` at 91.9 % and a clutter
displacement of 0.04 mm — the policy had barely moved anything. Cause: demonstrations that
include the transit run **1018–1118 env steps**, while `episode_length_s = 16.0` gave a horizon
of **800**. Every episode was truncated mid-transit.

Raised to 26 s (1300 steps, ~16 % headroom): **3.4 % → 7.8 %**.

**The horizon is a property of the manoeuvre**, so it has to be re-derived whenever the
manoeuvre changes length — it is not a config constant that stays put.

### Where the remaining gap is, precisely

At 7.8 % against a 31.9 % expert the retention is 24 %, and the taxonomy localises it exactly:

```
seed 88000: SUCCESS 7.0% | lifted  7.0% | over-box  7.0% | no-lift 93.0%
seed 88001: SUCCESS 4.7% | lifted  7.0% | over-box  5.5% | no-lift 93.0%
seed 88002: SUCCESS 11.7% | lifted 14.8% | over-box 12.5% | no-lift 85.2%
```

`success` tracks `lifted` almost exactly. **The carry and the place are learned; the whole loss
is the grasp.**

### It was the sample budget, and the scaling is steep

Adding the transit doubled every episode (488 → 1057 steps), so the same ~327 demonstrations
had to cover twice as much behaviour. Doubling the data and the training confirms it:

| | 1024 eps / 60 k steps | **2048 eps / 100 k steps** |
|---|---|---|
| expert (cold reset) | 31.9 % | 30.6 % |
| **BC (cold reset)** | 7.8 % | **20.1 %** |
| retention of expert | 24 % | **65 %** |
| `no_lift` | 90.4 % | 78.4 % |
| clutter displacement, median over successes | 0.04 mm | 0.04 mm |

**2.6× on 2× the data**, with no change to the architecture or the hyper-parameters. The
"PORTING DEFECT" banner was right to fire at 24 % and right to stop firing at 65 %.

Two things to note about the shape of this result. `success` still tracks `lifted` to within a
couple of points on every seed, so the carry and the place remain essentially perfect and every
remaining point is the grasp — which is exactly the part of the trajectory that needs coverage
of the spawn distribution, i.e. the part more data helps most. And the clutter displacement
median is **0.04 mm**: this policy is not buying its successes by shoving the scene, which is
the difference between a number and a behaviour worth deploying.

### The curve flattens at ~20 %

Third pass, 4096 episodes (1327 usable demos, 1.38 M samples) / 140 k steps:

| | 1024 / 60 k | **2048 / 100 k** | 4096 / 140 k |
|---|---|---|---|
| BC from cold reset | 7.8 % | **20.1 %** | 19.5 % |
| retention | 24 % | 65 % | 63 % |
| `lifted` | 7–15 % | 20–24 % | **21–32 %** |
| `near_miss` | 0.5 % | 0.8 % | **6.8 %** |
| `no_lift` | 90.4 % | 78.4 % | **72.9 %** |

**Data is no longer the lever.** But the composition moved in a way the headline hides: the
policy now *lifts* considerably more often (`no_lift` 78.4 → 72.9 %, `lifted` up to 32 % on the
best seed) and gives some of it back at the release (`near_miss` 0.8 → 6.8 %). Net flat, but
the failure has partly migrated from the grasp to the place — which is a different, and
easier, problem than the one it started with.

Three consequences for what to do next:

1. **The expert is now the binding constraint.** BC retains ~64 % of it across two independent
   runs, so at a 31 % expert the policy is near its ceiling. Every expert point is worth
   ~0.64 policy points, and the expert is the cheaper thing to improve.
2. **DAgger targets exactly the right thing.** The failures are concentrated (`no_lift` plus a
   new `near_miss` mode), the expert is available to relabel, and both modes are reachable
   states the policy actually visits.
3. **`near_miss` at 6.8 % is a release-height problem**, and it is cheap to test: sweep
   `CARRY_Z` (0.175) end-to-end the way `GRIP_Z` was swept.

---

## 5. Status

| step | state |
|---|---|
| expert planner re-priced (`w_o`, `w_pos`, `refine`, two axes, two heights) | **done** — 17/32 → 23/32 planned |
| `_chain` reports and gates on its own solve error | **done** — was silently discarding it |
| per-phase execution diagnostics (TCP vs plan, signed, gap, cube z) | **done** — this is what localised everything |
| action-driven home transit | **OPEN — §3b.** Eight variants measured, root cause identified |
| steady-state bias: root cause confirmed, fix specified | **done**, plan-time version not yet written |
| demo collection on the PHOTOREAL env | **done** — 1024 episodes, 323 successful (31.5 %) |
| flow-matching BC, 60 k steps | **done** — final loss 0.066 |
| evaluation, both protocols, 3 held-out seeds | **done** — 24.7 % matched (**77 %** of expert), 2.3 % cold |
| grip height re-measured for the real mesh | **done** — 32 mm, not 25 (§4b) |

**Two things to carry forward, in priority order:**

1. **The plan-time bias (§3b).** It is specified, the evidence is unambiguous, and `_kin`
   already has every piece it needs. This is what unblocks the transit and therefore the
   deployable policy.
2. **Regenerate demos against the reconstructed cube.** Everything measured in this document
   used the *primitive* collider, and is labelled as such. A 56 mm analytic cube and a
   reconstructed one are not the same grasp.

---

## 6. ⭐ ROOT CAUSE OF THE 31 % EXPERT — it closes the gripper before the arm arrives

Everything above treats the 31 % expert as a fact of the task. It is not. The sister tasks in
this same repo measure **~94 %** (`clutter/expert/clutter_expert.py`) and **91.4 %** pooled
(`slot`). 31 % is anomalous, and this section is the root-cause hunt that found why.

### 6.1 The failure is entirely at the grasp, and nowhere else

`_taxonomy` prints per batch. Across all 16 batches of the 2048-episode collection:

```
taxonomy: plan-failed 8-21 | never-got-there 70-87 | lifted-but-lost 0 | over-box-not-inside 0
```

`lifted-but-lost` and `over-box-not-inside` are **exactly zero in every single batch**. Once
the cube leaves the table the expert places it in the box 100 % of the time; ~92 % of layouts
get a valid plan. The whole ~69 % loss is the close itself.

### 6.2 What the gripper is doing — success vs failure, from the recorded demos

Mean gripper joint (`obs[:, 6:8]`, i.e. `joint_pos_rel`) at the close, 120 demos each:

| | grip joint at close | cube z, max over episode |
|---|---|---|
| **success** | **−0.015** — blocked by the cube | 151.6 mm |
| **failure** | **−0.040** — closes straight past it | 29.6 mm (median) |

In failures the fingers travel **2.7× further** and shut on empty air. This is not slip, not a
drop, not friction. The cube is never between the fingers.

### 6.3 No layout variable predicts it — three hypotheses killed

All from `runs/final2/demos.hdf5`, 1851 planned episodes, no GPU needed:

| hypothesis | test | result |
|---|---|---|
| arm sags more when extended | success vs cube radius in root frame, 6 sextiles over 150–280 mm | **33.0 / 34.1 / 31.1 / 32.5 / 35.3 / 37.2 %** — flat. REFUTED |
| some envs got slippery/heavy cubes | mass + friction are randomised at **startup**, so they are fixed per `env_index` across all 16 batches. Per-env success spread vs the binomial null | std 0.123 vs null 0.124 → **ratio 0.99×**. 0 envs at 0 %, 0 envs ≥ 80 %. REFUTED |
| the opening axis is chosen badly | success vs cube yaw mod 90°, 6 bins | **32.6 / 33.4 / 35.7 / 32.2 / 34.5 / 35.0 %** — flat. REFUTED |

Only the polar angle showed anything (23.1 % near θ = 0° vs 41.4 % near θ = +32°), and that is
a minority effect, not a 69 % one.

**This exact situation is already written down in the clutter expert**, which reached 94 %:

> *"The pose draw is the largest remaining source of variance — 32.8 % to 74.2 % success
> between draws of the same configuration (P25 arm A, sd 17.5 %) — and no forward-kinematic
> statistic tried so far predicts it. `o_align`, keep-out penetration and wrist height are all
> near-identical across a good draw and a bad one. So stop guessing: run each candidate's
> close in the sim and keep the one that measurably works."*

### 6.4 ⭐ The measurement that localised it

`collect_demos.py --chatty`, 64 envs, seed 11, `BIAS_MAX=0.0` (production settings):

| leg | TCP error vs plan | gap | cube z |
|---|---|---|---|
| `home` (transit) | **1.1 mm** median, **1.7 mm** p90 | 89.1 mm | 27.2 mm |
| **`grasp`** | **11.1 mm** median, **60.0 mm** p90 | 89.1 mm | 27.2 mm |
| `closed` | 2.6 mm, p90 56.3 mm | 44.5 mm | 27.2 mm |
| `lift` | 5.4 mm | **−1.2 mm** — shut on air | 27.2 mm |

The transit tracks its plan to **a millimetre**. The grasp descent arrives **11 mm out at the
median and 60 mm at p90**. The gripper opens to 89 mm on a 53–58 mm cube, so the margin is
~17 mm a side: 11 mm spends most of it, 60 mm is a clean miss.

### 6.5 The cause, and why it went unseen for so long

Two compounding reasons, and the second is embarrassing:

1. **A position drive holds a loaded pose only *with* an error.** Torque is stiffness × error,
   so any pose that needs torque is held off-target. `cem` cannot see this at all: it scores
   candidates through `write_joint_state_to_sim`, which places the arm kinematically — no
   gravity, no contact. Every pose it returns is reachable in that sense and some are simply
   not *holdable*. This was already written down in §3a and acted on only for the transit.

2. **The grasp descent had no settle, and the transit did.** Look at the two legs side by side
   in `main`: the transit ends with `for _ in range(40): step(prev, ...)` — a hold that lets
   the arm converge before the descent begins — and measures 1.1 mm. The grasp descent ramps
   down and goes **straight into `for _ in range(70): step(prev, True, ...)`**, closing the
   fingers while the arm is still travelling. It measures 11.1 mm. The settle was added to the
   transit when it was found to be lagging; nobody added the same thing to the grasp.

The two legs differ in exactly one way and measure 10× apart. That is the hypothesis.

### 6.6 Fixes, in the order they are being tested

1. **`GRASP_SETTLE`** (`collect_demos.py`, default 40 env steps) — hold at the grasp pose,
   fingers still open, before closing. Exact parallel to the transit's settle. Swept by
   `probes/sweep_grasp_settle.sh` over `0 20 40 80 160`, where **0 reproduces the shipped
   behaviour exactly and is the control**. Paired: same seed, same layouts, end-to-end.
2. **In-sim close screen**, ported from `clutter_expert.py::_screen` — solve K candidate grasp
   poses, actually close on each in the simulator, and keep the one that measurably encloses.
   This is the measured answer to §6.3 on the sister task and does not depend on the mechanism
   being understood. Note `_solve_grasp(tries=1)` currently declines to draw a second
   candidate on the argument that restarts already supply the diversity — the clutter
   measurement (32.8 % → 74.2 % between draws) says that argument is wrong.

Also worth noting for §5.8's sake: `STANDOFF`, `CARRY_Z` and `O_ALIGN_MIN` are now
env-overridable so they can be swept the way `GRIP_Z` was, without editing the module.

### 6.7 ⭐ MEASURED: the missing settle was worth **+14.1 points**

`probes/sweep_grasp_settle.sh`, 64 envs, seed 11, paired (identical layouts), end-to-end with
no `--teleport-pregrasp`. `GRASP_SETTLE=0` reproduces the shipped behaviour and is the control.

| `GRASP_SETTLE` | success | grasp TCP err median | p90 | `never-got-there` |
|---|---|---|---|---|
| **0** (shipped) | **40.6 %** (26/64) | **11.1 mm** | 60.0 mm | 33 |
| 20 | **54.7 %** (35/64) | **0.9 mm** | 56.4 mm | 24 |
| 40 | 54.7 % | 0.9 mm | 56.2 mm | 24 |
| 80 | 54.7 % | 0.9 mm | 56.2 mm | 24 |
| 160 | 54.7 % | 0.9 mm | 56.2 mm | 24 |

The median arrival error collapses **11.1 mm → 0.9 mm**, i.e. to the same 1 mm the transit leg
already achieved with its own settle, and the effect **saturates by 20 steps** — this is
tracking lag being allowed to decay, nothing more. Default left at 40 for margin; 20 is
sufficient on this evidence.

The signed error is the tell: `+6.6 / +0.8 / +2.2 mm` at settle 0 becomes `−0.1 / +0.0 /
−0.5 mm` at settle 20. A *systematic* offset in the direction of travel is what a lag looks
like; it is not what gravity sag looks like (that would be a persistent −z).

**What this did NOT fix, and it is the interesting part.** The p90 barely moves: 60.0 → 56.2 mm.
So there are two populations, and only one of them was lag:

* **the median pose** — arrives fine once given ~20 steps to converge. Was ~2/3 of the loss.
* **a residual ~25 %** — still 56 mm from a plan whose own gate accepted it at ≤ 1.5 mm, no
  matter how long it is given to settle. These are poses the drive genuinely cannot hold, and
  they are exactly the case §3a described and §6.3 showed no kinematic statistic can predict.

More settling cannot help the second population. Selecting a different pose can — which is
what the in-sim close screen (§6.6 item 2) is for: it does not make an unholdable pose
holdable, it detects that this one is and re-draws.

**Consequence for everything downstream.** Every expert and BC number in this document before
2026-08-06 was produced by an expert that closed its gripper while the arm was still moving.
The BC scaling curve (7.8 % → 20.1 % → 19.5 %) was measured against a 31 % expert and retained
~64 % of it; at 55 % that projects to ~35 %, and the flat curve may simply have been the
policy faithfully reproducing a demonstrator that missed.

### 6.8 ⭐ MEASURED: the in-sim close screen takes the expert to **71.9 %**

Ported from `clutter/expert/clutter_expert.py::_screen` as `screen()` in `collect_demos.py`,
behind `SCREEN_ROUNDS` (0 = shipped planner). Batched: every env screens ITS OWN candidate in
the same physics steps, so a round costs one close regardless of `n`. It teleports to the
candidate grasp, settles OPEN under gravity, closes, lifts, and asks the only question that
matters — **did the cube come up?** Envs that fail get a fresh CEM draw and are screened again.

Same 64 envs, same seed 11, same layouts, end-to-end:

| configuration | success | planned |
|---|---|---|
| shipped | 40.6 % (26/64) | 59/64 |
| `GRASP_SETTLE=40` | 54.7 % (35/64) | 59/64 |
| **`+ SCREEN_ROUNDS=2`** | **71.9 % (46/64)** | **64/64** |

**40.6 % → 71.9 %, +31.3 points, 1.77×.** The screen's own trace shows why:

```
screen round 0: 54/64 grasps actually lift
screen round 1: 63/64 (re-drew 10)
screen round 2: 64/64 (re-drew 1)
```

`SCREEN_ROUNDS=2` and `=4` tie at 71.9 %, because round 1 already reaches 63/64. Two rounds is
the setting. Planning also went 59/64 → 64/64: re-drawing a failed env gives the gate another
independent chance, so layouts that used to be unplannable now solve.

**The residual, stated honestly.** The screen certifies 64/64 grasps lift, but end-to-end only
46/64 succeed. So ~18 envs pass the screen and still fail. That is expected and it localises
the next piece of work: the screen *teleports* to the grasp pose, while the real run
**descends** ~140 mm onto it, and the descent introduces error the screen never sees. Screening
through the actual descent (`run_phys` down the last leg instead of `teleport_arm`) is the
obvious next refinement.

**Why this worked when six sweeps of `GRIP_Z`, `STANDOFF` and `w_o` did not.** Every one of
those tuned a *statistic the planner can compute*. §6.3 measured that no such statistic
predicts the outcome. The screen does not rank candidates better — it stops ranking them and
runs the experiment.
