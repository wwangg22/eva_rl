# `Rebot-PreGrasp-v0` — non-prehensile reconfiguration before grasping

> ## ⚠ Status: premise disproven. This env needs a redesign before it is worth training.
>
> The task is built on the claim that the block, lying flat, is **too wide for the gripper to
> span**. That claim is false on this hardware, and the measurement is unambiguous. Everything
> below is kept because the MDP itself is sound and fully tested — only the mechanism that was
> supposed to make the start state ungraspable does not work. See
> [What went wrong](#what-went-wrong) for the numbers and the options.

**Skill under test:** extrinsic dexterity. The policy must *create its own preconditions*
before any grasp is possible.

**Source:** `challenge/pregrasp_env_cfg.py`, `challenge/mdp/pregrasp.py`
**Gym IDs:** `Rebot-PreGrasp-v0` (2048 envs), `Rebot-PreGrasp-Play-v0` (16 envs)
**Video:** `logs/videos/Rebot-PreGrasp-Play-v0_settle.mp4`
**Probe:** `scripts/challenge/pregrasp_probe.py` · **Test:** `scripts/test_pregrasp_env.py`

## The intended task

A block spawns lying flat against a back wall. Lying, its horizontal shadow is 100 mm
across at its narrowest; tipped up on edge it presents 60 mm. The policy was meant to have
to push it against the wall until it pivots up, and only then grasp it — a non-prehensile
action with no reward of its own, with the reach-and-close gradient flat until it has
already happened.

## What went wrong

Three measurements, in the order they landed:

**1. The gripper opens 89.1 mm, not 45 mm.** `_GRIPPER_OPEN = 0.045` is a *per-finger
prismatic joint value* and both fingers move: `separation = 2.000 × q`, exactly, and zero
when shut. The original design read 0.045 as a stroke and sized the block against it. That
alone moved the required block width from ~50 mm to >89 mm.

**2. The fingers can be forced open to ~120 mm.** The joint limits are 0.0500 m and
0.0715 m — asymmetric, summing to a 120 mm clear gap. The binary action can only *ask* for
89 mm, but an object wider than that just pushes the fingers apart on contact. So a 100 mm
block is not too wide at all:

| trial | presented width | finger gap after closing | lifted |
|---|---|---|---|
| lying flat | 100.0 mm | **101.4 mm** — splayed past the commanded opening, then gripped | **100 %** |
| up on edge | 60.0 mm | 60.4 mm — fingers exactly around it | 100 % |

The lying block is lifted every time. The start state is graspable.

**3. Nothing that fits the workspace can be too wide.** To be ungraspable a block would have
to exceed 120 mm in *every* horizontal direction. The usable TCP band is x ≈ 0.22–0.26 m
with a 44 mm floor (`CHALLENGE_SUITE.md` C9), so a 120 mm object spans the entire reachable
region and there is no room left to approach or push it. **Width cannot be the ungraspability
mechanism on this arm.**

Separately, the scripted wall-push tips the block in only 2–8 % of trials across push
heights of 46–62 mm, so even the intended route is weakly supported.

### Redesign options, in order of promise

1. **Use height, not width.** The 44 mm TCP floor is a hard, measured barrier: an object
   whose graspable feature sits below it cannot be reached at all. A block lying flat and
   only ~25 mm tall is genuinely ungraspable; stood on end it is 100 mm tall and 40 mm wide,
   which is comfortably graspable (proven — trial B above, 100 %). The open question is
   whether the gripper can *push* something that short, since the same floor limits contact.
2. **Occlusion.** Seat the block in a pocket whose walls stop the fingers entering beside it,
   so it must first be slid out into open space.
3. **Replace the env.** If neither mechanism survives measurement, drop extrinsic dexterity
   for this arm and pick a fourth skill the hardware supports.

Whichever is chosen, `pregrasp_probe.py` already implements the two-sided test it needs:
a grasp that must fail, a grasp that must succeed, and a scripted route between them.

## Scene (as currently built)

| entity | geometry | pose |
|---|---|---|
| block | cuboid **100 × 60 × 100 mm** authored standing, 0.08 kg, μ 0.8/0.6 | `(0.265, 0, 0.030)`, rot 90° about x (lying) |
| wall | static box `(0.04, 0.36, 0.14)` | inner face at x = **0.320** |
| table / plane / light | as all challenge envs | |

Lying, the block presents 100 × 100 mm and stands 60 mm tall; on edge it presents
100 × 60 mm and stands 100 mm tall.

## MDP

**Actions — 7-D**: 6 joint positions (`scale=0.5`, offset from `_START_POSE`) + binary gripper.

**Observations — 32-D**

| # | term | dim | slice |
|---|---|---|---|
| 1 | `joint_pos` | 8 | `0:8` |
| 2 | `joint_vel` | 8 | `8:16` |
| 3 | `block_pose` (root frame) | 7 | `16:23` |
| 4 | `block_extent` — narrowest width the block presents | 1 | `23` |
| 5 | `block_up` — world-z of the block's own long axis | 1 | `24` |
| 6 | `actions` | 7 | `25:32` |

Term 4 is `min_grasp_width`, the minimum width of the block's **horizontal shadow**,
minimised over closing directions. It has been wrong twice, in opposite directions, and both
cases are now pinned by the smoke test:

- projecting each *body axis* separately and taking the smallest reports ~0 mm for a block
  lying flat, because its vertical axis projects to nothing — i.e. "maximally graspable"
  exactly when the block is least so;
- normalising the degenerate "kink" direction of a vertical axis yields a zero vector, whose
  width evaluates to 0 mm — which made an axis-aligned upright block read as 0 mm wide.

**Rewards**

| term | weight | note |
|---|---|---|
| `reaching` | 1.5 | std 0.12 |
| `uprighting` | **20.0** | dense credit for tipping the block up; measured as how far the presented width has fallen from 100 mm toward 60 mm, so it is orientation-agnostic and pays for tipping about either horizontal axis |
| `lifting` | 10.0 | above 0.075 m and within 0.08 m of the TCP — the threshold must clear the 0.050 m the block's centre already sits at when merely stood up |
| `success` | 60.0 | reoriented **and** lifted **and** still in the gripper |
| `dropping_penalty` | −30.0 | |
| `action_rate` / `joint_vel` | −2e-2 / −5e-3 | |

**Terminations:** `time_out` (14 s), `block_dropped`. Deliberately **no** `block_toppled`
term — lying down is the start state here, not a failure. The smoke test checks that nothing
terminates on the first step, which is what such a term would cause.

**Success:** presented width below 80 mm, block above 0.075 m, TCP within 0.08 m.

## Validation status

`scripts/test_pregrasp_env.py` — **passes**, and prints the known-issue banner:

- 32-D observation concatenates and is finite; both scalar terms return `(N, 1)`
- block settles lying at z = 27 mm with its long axis horizontal
- presented width 100.00 mm lying / 60.00 mm on edge, to 0.5 mm
- uprighting reward spans 0.000 → 1.000 between the two poses
- success predicate fires 16/16 when reoriented, lifted and held, and is rejected by three
  negative controls (far from the gripper / still flat / never lifted)
- `block_dropped` fires below the table; nothing terminates at reset

`scripts/challenge/pregrasp_probe.py` — **the premise fails**, as above. The probe itself is
now trustworthy; getting it there took four corrections worth remembering:

1. retention must be a **real lift** — asking whether the block is within 80 mm of the TCP is
   satisfied by an untouched block lying on the table under the gripper, which scored a
   107 mm block in an 89 mm gripper at 100 %;
2. the verdict must come from the **finger separation after closing**, which cannot be faked;
3. **every CEM search must finish before the block is placed** — the search evaluates
   candidates by writing joint states into the sim, which teleports the arm and re-opens the
   fingers hundreds of times, so searching for the lift path *after* closing silently drops
   the block and the grasp reads as a slip;
4. grip **above** the object's centre of mass and above the 44 mm TCP floor, or it rotates
   out of the fingers during the lift.

With those fixed, the control trial — the precision-slot block at its proven geometry —
lifts **100 %** of the time, so the gripper itself is sound.
