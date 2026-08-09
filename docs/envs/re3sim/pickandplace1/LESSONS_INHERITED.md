# Lessons inherited — mistakes we are not allowed to repeat

Distilled from `eva_bc/README.md` "Lessons Learned", `eva_bc/docs/POSTMORTEM.md`,
`eva_bc/docs/slot/EXPERT_PORT.md`, `eva_bc/docs/slot/EXPERT_RESULTS.md` and
`eva_bc/clutter/docs/06_EXPERT_DESIGN.md`. **These cost real GPU-weeks.** Every one of them
is measured, and several are retractions of confident earlier claims.

---

## A. Expert / motion generation

**A1 — cuRobo is not available.** `python -c "import curobo"` fails in `env_isaaclab6`. The
`eva_bc` Stage-1 expert is cuRobo-based; both later ports (slot, clutter) had to replace it.

**A2 — differential IK diverges near the table, at exactly our height band.** Both repos say
so explicitly: *"DLS differential IK diverges from table-level configs — never trust raw DLS
near the table"* (`eva_bc/docs/PLAN.md:41`), and *"differential IK stalls at a z-floor
~0.045 m"* (`eva_rl/scripts/scripted_expert/generate_pick_place.py`). Our grasps live at
z ≈ 0.044–0.056. **DLS is the wrong primary instrument for this task**, unlike the slot task
where the insertion sits at z ≥ 0.090 and DLS was fine.

**A3 — the right instrument is FK-scored search read back from the sim.** `clutter`'s
conclusion: CEM over the 6 arm joints, scored by forward kinematics *evaluated in the sim*,
driving Cartesian waypoint chains. The property that matters is that **the achieved TCP error
is reported**, so the search cannot silently converge on an unreachable pose. Reference
implementations already exist: `eva_rl/scripts/analysis/grasp_geometry.py` and
`eva_rl/scripts/challenge/slot_insertion_probe.py`.

**A4 — planner-valid ≠ executable.** cuRobo returned plans whose end poses the PD-controlled
arm could not track; executed height of planner-valid poses spread ~16 mm. Detection needs an
*executed-state* check, not a planner success flag.

**A5 — position-only IK is not a sufficient specification for this arm.** The
finger-separation axis must be constrained explicitly or the wrist arrives holding the object
along the wrong dimension. Measured on the slot task: **0 % → 55–81 %** insert rate. The cost
term is `cost = pos + 0.25 * (1.0 - (sep @ Y).abs())`.

**A6 — finish every CEM/IK search BEFORE closing the gripper.** These searches call
`write_joint_state_to_sim`, which teleports the arm and re-opens the fingers hundreds of
times. Searching for a lift path *after* closing silently drops the object and reads as a
slip. Fixing the ordering alone took one control trial **0 % → 100 %**, and `eva_rl`'s handoff
says this cost more time than anything else in that effort.

**A7 — joint-space interpolation between two IK solutions does not keep the TCP straight.**
Solve a Cartesian waypoint chain with a tight search radius.

**A8 — CEM is not reproducible enough to generate demonstrations.** Use it for feasibility
probing and elbow-branch selection; get accuracy from a deterministic solver on top.

**A9 — retry loops must exclude failed candidates by identity, not list position.**

**A10 — run a GO/NO-GO feasibility spike before building the full expert runner**
(`eva_bc/expert/spike_plan_grasp.py` pattern). Do not write 800 lines against an infeasible
grasp.

---

## B. Success predicates and measurement

**B1 — success predicates are usually weaker than they read.** `mdp.is_inserted` bounded the
block's z only from *below*, so it was satisfied by a block resting on top of the walls or
still dangling in the gripper. One probe scored **93.8 % with a mean lateral error of
13.28 mm** — geometrically impossible inside the channel. **Write the strengthened predicate
first and report every number against it.**

**B2 — aggregate success rate hides real change.** Two policies matched to the decimal while
34 of 64 episodes flipped. Always report the failure taxonomy and the per-episode diff.

**B3 — a constraint that is never checked is not a constraint.** `clutter`'s
`distractors_disturbed` was computed on every step and wired only to a shaping reward; a
73.3 % expert was really 16.4 % once displacement was actually tested. Anything we claim the
task requires must appear in the termination or the success predicate.

**B4 — calibrate thresholds against the solver's own noise floor.** `DISTURB_TOL = 2 mm` was
adopted only after measuring that blocks drift 1 µm under a null action over a full episode.

**B5 — before believing physics contradicts geometry, check both describe the same
configuration.** A "contradiction" in the slot task was an empty gripper being compared with
one holding the block.

---

## C. BC training

**C1 — training-seed variance dominates single-run A/Bs.** Same data, same recipe spanned
**32.8 %–59.4 %**, and same-data different-seed pairs flipped 31–39 of 64 episodes — the same
flip count as the "treatment" comparison it invalidated. **Standing rule: ≥3 seeds per arm,
champion selected on a held-out spawn seed, pooled ≥128-episode numbers only. Single-run
comparisons are void.** Several confident verdicts ("DAgger nets zero", "offline recovery
data actively hurts") were later retracted as seed noise.

**C2 — chunk commitment is load-bearing.** Shortening the execution horizon at eval time,
with no retraining, collapsed success monotonically: **59.4 → 32.8 → 3.1 → 0 → 0 %** at
n_action_steps = 15/8/4/2/1. Never shorten the horizon; put RL on top of committed chunks.

**C3 — offline behaviour analyses must use each policy's own normalizer stats and
outcome-filtered states.** The original postmortem's "kill shot" finding was retracted after
forensics found it had normalized policy B's obs with policy A's stats. Offline probes on
expert states cannot rank policies at all.

**C4 — same-seed A/B pairing silently breaks at the first behavioural divergence** when
subsystems share an RNG stream.

---

## D. RL

**D1 — `rl_games` `clip_actions` is an action SCALE, not a clip.** `preprocess_actions`
clamps to [-1, 1] then **rescales to the action-space bounds**, so `clip_actions: 100.0`
multiplied every action by 100. Two full training runs and two wrong theories were spent
before reading the rl_games source. **It must be 1.0.** General form: every `env:` value in an
adapted RL config is load-bearing — read the consumer's source.

**D2 — additive action residuals on a flow base wash out symmetrically.** The healthy run
fixed 26 episodes and broke 26, exactly flat, with a **state-independent** learned residual —
PPO learned effort, not discrimination. Use **x0-steering** instead: it selects among the
base's own modes rather than perturbing its output. Measured 55.5 % → **91.4 %**.

**D3 — "zero-init residual" needs three things.** Zero the mu *weight*; know the mu *bias*
stays randomly initialised; and use a **small initial sigma**. σ ≈ 0.37 destroyed a base that
lives on mm-precision grasps (−46 points); σ ≈ 0.08 was healthy. Measure the frozen base's
noise tolerance *before* training.

**D4 — train reward can rise while success stays flat.** The flat residual run *beat* the base
on training reward with zero success change. Only held-out eval success counts.

**D5 — per-term reward channels from epoch 1 are the cheapest tripwire.** A dead run was
diagnosable at epoch 1 (`Episode_Reward/placed = 0.0` from the first log). Caveat for
chunk-window RL: episodic loggers only report on episode *completion*, so early epochs are
legitimately empty — know the cadence before reading zeros as pathology.

**D6 — fixed-action attribution beats theorizing.** When training reward looks wrong,
reproduce the exact training config and roll fixed action conditions (zero / bias / small
noise / large noise) with no RL. One 10-minute run pinpointed what two rounds of log analysis
had misdiagnosed.

---

## E. Engineering discipline

**E1 — verify every wrapper by bit-exact reproduction of the frozen base before any
training**, with the learned component zeroed. Cheap, decisive, and it *permanently*
exonerates the wrapper in every later debugging session. Make it enforceable with a
comparator that exits nonzero and aborts the chain.

**E2 — vectorize per-env controller state and gate the rewrite on bit-exactness.** Per-env
python deques were the scaling limit long before VRAM; the tensor-queue controller runs 2048
envs at 10–15 k steps/s in ~7 GB.

**E3 — information can be present in the obs but unused (salience failure).** The
grasped-vs-closed-on-air distinction was decodable from a *single frame*: a 5-dim probe
(finger pos/vel + last grip command) hit AUC 0.968 at **0 % FPR**, while the same probe given
all 41 dims mislabeled 53.5 %. The fix is re-surfacing the signal as an explicit validated
bit, not adding history (refuted: ≤ +0.01 AUC, transfers worse). The probe **needs the
commanded-action channel** — aperture alone gives 27.1 % FPR.

**E4 — pre-register experiments.** Design, beliefs and decision rules written down *before*
coding, verdicts recorded in place, retractions kept with dated correction blocks rather than
silent edits.

**E5 — one GPU job at a time.** 10 GB card. Use `python -u` or output buffers and you see
nothing. Long jobs go to `systemd --user`, never the foreground of a session.
