# Step 2+ — from a working env to a policy

**Precondition:** Step 1 complete — `Rebot-Workstation-PickPlace1-v0` builds, randomises,
and every object in it is *proven* graspable (`../01_STEP1_PLAN.md`).

**Shape of the plan:** the same staged ladder `eva_bc` ran on the 2-can task, with each stage
gated and the gate written down before the stage runs. Two deliberate deviations from that
history, both justified by what it measured:

1. **The expert is FK-scored search, not cuRobo and not raw DLS IK** — cuRobo is not
   installed, and DLS provably diverges in exactly our height band (LESSONS A1–A3).
2. **RL goes straight to x0-steering.** The additive residual was measured *exactly flat*
   on this arm and task family; it is kept only as a cheap ablation (LESSONS D2).

Read [LESSONS_INHERITED.md](LESSONS_INHERITED.md) before writing any code in any stage.

---

## What the env must expose (design this in during Step 1)

The BC/RL stack in `eva_bc` binds to a small `mdp` interface. Retrofitting it is more
expensive than providing it up front, so Step 1's P2 should already emit:

| touchpoint | meaning |
|---|---|
| `mdp.placed_mask(env) -> (N,) bool` | the **strengthened** success predicate (LESSONS B1) |
| `mdp.object_pos_local(env, name) -> (N, 3)` | per-object env-local position |
| `mdp.box_centers_local(env) -> (N, 2)` | container centre, env-local (the `basket_centers_local` analogue) |
| `mdp.OBJECT_NAMES` | canonical name list |
| an `objects_canonical` obs term | target-first canonical ordering; the residual/steer obs builders mirror its target-selection logic exactly |

**Action interface** (ports verbatim from `run_expert_v1.py:428-455`, keep it identical):

```python
a = torch.zeros(1, 7, device="cuda")
a[0, :6] = (dq - q_default) / 0.5   # 1 action unit = 0.5 rad, relative to default joint pos
a[0, 6]  = grip                     # +1 = OPEN, -1 = CLOSE
```

Record the `(o_t, a_t)` pair **before** the step and refresh `last_obs` after, so the
obs/action alignment stays causal.

Note this task has **three** objects, not two — the observation layout, the canonical
ordering and the target-selection logic all widen, and `act/dataset.py`'s hardcoded 41-D
split has to be re-derived from the env config rather than guessed.

---

## S1 — Feasibility spike (GO / NO-GO)

**Purpose:** prove a grasp exists for all three objects *before* building an 800-line expert
runner (LESSONS A10).

- FK-scored CEM over the 6 arm joints, cost including the finger-separation-axis term
  (A5), evaluated in the sim so the achieved TCP error is reported (A3).
- Sweep: each object × a grid of yaws × a sample of spawn poses across the reach annulus,
  plus the box at a sample of its own placements.
- Reuse `scripts/analysis/grasp_geometry.py` and `scripts/challenge/slot_insertion_probe.py`
  as the reference implementations.

**Gate:** a per-object, per-yaw success map with achieved TCP error reported, and a stated
minimum feasible fraction. **If an object's map is empty, stop and fix the env** — that is
B1 resurfacing, and no amount of expert engineering fixes an infeasible grasp.

**Also measure here:** the clearance for a gripper *holding* each object to enter the box
mouth. Slot task lesson B5/§1a — an empty gripper and a loaded one clear a channel at
completely different heights, and getting this wrong invalidated a whole session.

---

## S2 — Scripted expert + demo generation

**Purpose:** an expert that drives the env with privileged state and records demonstrations
to HDF5 with per-segment phase labels and a `train_mask` that loss-censors the expert's own
failed sub-attempts.

Structure ports from `eva_bc/expert/run_expert_v1.py` almost entirely — the 811-line runner
is generic apart from the planner bridge. The minimal planner interface to implement:

1. `plan_grasp(q6, target) -> {approach_qs, grasp_qs, lift_qs}` or a failure token
2. `plan_place(q6, box_xy, heights) -> qs` or failure
3. `plan_to_config(q6, q_goal) -> qs` or failure (the retreat)
4. `update_world(state)` — no-op if paths are analytic and collision-free by construction
5. `attach` / `detach` — no-op if the payload is not collision-checked

Everything else ports as-is: `build_train_mask`, env setup, state readers, perturbation
primitives, `step_action` / `mark` / `run_traj` / `hold`, the `fetch_one` retry-and-accounting
control flow, `episode()`, and the HDF5 writer.

**Ordering rule that is not optional:** finish every search before closing the gripper (A6).

**Gate:** expert success rate and **failure anatomy** on a fixed suite, reported against the
strengthened predicate. `eva_bc` measured ~94 % nominal / 77.4 % perturbed; the perturbed
number matters because it caps what DAgger can later teach. Report per-object success — with
three heterogeneous objects a pooled rate mixes materially different difficulties, exactly the
mistake `clutter` documents for its per-slot statistics.

**Then:** seeded batch demo generation (`gen_demos_nominal.sh` pattern) and a coverage /
stratification audit (`act/report_coverage.py`).

---

## S3 — Flow-matching chunk BC

Train a rectified-flow chunk policy on the demos: flow head on the vendored LeRobot ACT
transformer, CVAE/KL deleted, `action_is_pad` censoring kept exact via a decoder
`key_padding_mask`. Chunk 50, execute 15, temporal ensembling OFF, 10 Euler steps, seedable
x0. External mean/std normalization.

**Do not shorten the execution horizon** (C2).

**Gate:** held-out batched sim eval (S4), **≥3 training seeds**, champion selected on a
*held-out spawn seed* and confirmed on pooled ≥128-episode numbers. **Single-run comparisons
are void** (C1). Make sure the trainer is seeded — `eva_bc`'s was not when v1–v3 were trained,
which is what made three published verdicts unsupportable.

**Calibrate the churn floor early.** Before comparing any two policies, train 3 replicas of
one config and measure how many episodes flip between them. That number is the noise floor
every later A/B has to beat.

---

## S4 — Batched sim evaluation

Roll a checkpoint across N parallel envs with per-env action queues (receding horizon) and a
privileged flush trigger on detected object discontinuity. Fully vectorized controller state —
a tensor queue, not python deques (E2).

**Gate:** this *is* the gate for every other stage. It must be deterministic per spawn seed
(same-checkpoint re-run churn = 0), or per-episode diffs mean nothing.

**Build the failure taxonomy here**, not later: never-lifted / lifted-but-not-placed /
placed-some-stuck / dropped-after-place, per object. Aggregate success hides real change (B2).

---

## S5 — DAgger (optional, low priority)

HG-DAgger-style gated collection: frozen BC policy drives, failure gates (miss / stall / drop
/ timeout) trip an expert takeover from the current sim state, only the *recovery* is labelled
trainable.

**Evidence from `eva_bc` says to deprioritise this.** Its measured verdict was "no measurable
interference, weak evidence of stabilization" — arm means 56.7 % (+dagger, n=3) vs 47.4 %
(nominal, n=3) — and the DAgger *teacher itself* had a 68 % takeover ceiling concentrated
exactly on the states the policy visits most. Run it only if S4's taxonomy shows a failure
bucket that is clearly covariate shift rather than mode selection.

---

## S6 — Grasp-success bit probe

A small MLP on 5 raw obs dims (finger pos + finger vel + last commanded grip) emitting a
validated grasp-success bit, added as an observation feature for the RL stages.

**Gate (pre-register it):** **0 % false-positive rate** on on-policy closed-on-air freeze
states AND ≥95 % accuracy on expert post-close frames. Runtime semantics: probe output ANDed
with commanded-closed.

Do not substitute history for this (refuted, E3), and do not drop the commanded-grip channel
(physical finger joints alone: better AUC, 27.1 % FPR).

---

## S7 — x0-steering RL on the frozen BC base

Chunk-level RL that picks the flow base's integration noise x0 per execution window:
`x0 = alpha_x0 * tanh(z)` held for the window, window-summed reward. Outputs are always
on-manifold — the base's own decoder turns x0 into a coherent chunk.

**Gates, in order:**

1. **Bit-exactness.** `z = 0` must reproduce the x0-zeros base episode-for-episode. A
   comparator that exits nonzero and aborts the chain (E1).
2. **Early health.** Per-term reward channels sane from epoch 1 (D5), knowing the logger's
   cadence.
3. **Adoption.** Pooled ≥128-episode held-out result vs the deterministic base with a
   **pre-registered** improvement threshold, plus a mandatory taxonomy diff: did the targeted
   failure bucket collapse *without* symmetric new breakage?

**Before training:** measure the frozen base's noise tolerance and set the initial sigma from
that measurement, not by default (D3). And check `clip_actions: 1.0` (D1) — this one has
already cost two full training runs once.

**Ablation (cheap, run it):** the additive per-step residual, purely to confirm it is flat
here too. Its infrastructure is shared with the steering path anyway.

---

## S8 — Vision (the actual point of a photoreal env)

Everything above is state-based. The reason to reconstruct a real desk is that a policy
trained in it can see. `eva_bc` has `EXP08_visual_policy.md` / `EXP09_visual_steering.md` and
`eva_rl` has `lift_vision/` and `pick_place_vision/` including
`lift_vision/visual_randomization.py`.

Planned as teacher→student distillation: freeze the state-based teacher from S7, render the
NuRec splats through the env's cameras, and distil into a vision student with visual DR
(lighting, materials, camera jitter). DAgger-capable.

This is where the Re³Sim premise gets tested: does a policy trained on reconstructed pixels
transfer? That question is out of scope until S7 clears its gate, but the env should be built
with camera slots present from the start so this does not require a rebuild.

---

## Risks specific to this task

| risk | why it is new here | mitigation |
|---|---|---|
| Three heterogeneous objects, not two identical cans | pooled success mixes different difficulties; obs layout widens | report per-object throughout; re-derive the obs split from the env cfg |
| Grasps at z ≈ 44–56 mm | precisely where DLS IK is documented to fail | FK-scored search from S1 onward (A2/A3) |
| Objects must go *into* a box, not onto a target xy | loaded-gripper clearance, not empty-gripper | measure loaded clearance in S1 (B5) |
| Box is randomly placed | the `eva_bc` expert needed a 12,953-row grasp table for a randomised basket | expect the place leg to need a goalset ranker, not a single nominal |
| Photoreal rendering cost | splats + cameras are not free | keep S1–S7 state-based and headless; vision only at S8 |
| 10 GB VRAM, shared | `eva_bc` budgeted 2048 envs at ~7 GB on a 12 GB card | one job at a time; check `nvidia-smi` first (E5) |

## Standing rules for this project

- ≥3 seeds per arm; champion on a held-out spawn seed; pooled ≥128 episodes. No single-run
  verdicts.
- Every wrapper gated on bit-exact reproduction before any training.
- Every claimed constraint appears in the termination or success predicate, or it does not
  exist.
- Pre-register each experiment; record verdicts in place; retractions get dated correction
  blocks, never silent edits.
- One GPU job at a time; long jobs as `systemd --user` units; `python -u`.
