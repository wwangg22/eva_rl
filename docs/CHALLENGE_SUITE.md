# reBot challenge suite — design and validation plan

The existing `Rebot-PickPlace-v0/v1` tests one skill: side-grasp a can and drop it into a
110 mm-wide open basket. The placement tolerance is **±50 mm** and the basket has no lid,
no neighbours and no ordering constraint. This document designs a suite of tasks that each
isolate a *different* manipulation skill, and records the measured hardware limits that
every design has to respect.

Nothing here is assumed. Every constraint below was measured on this asset in this sim.

---

## 1. Measured hardware envelope

Harnesses: `scripts/analysis/reachability_map.py`, `query_reachability.py`,
`query_socket_sites.py`, `grasp_envelope.py`. Raw data under `logs/analysis/`.

### C1 — There is no top-down grasp anywhere near the table

819,200 sampled joint configurations, TCP quantized to 2 cm voxels, 62,763 voxels with
≥5 samples. "Top-down capable" = some attainable pose has the finger axis within 26° of
straight down.

| TCP height band | voxels | top-down capable |
|---|---|---|
| z ∈ [−0.10, 0.00) | 4,903 | **0.00 %** |
| z ∈ [0.00, 0.10) | 5,853 | **0.00 %** |
| z ∈ [0.10, 0.25) | 9,121 | 0.10 % |
| z ∈ [0.25, 0.45) | 15,584 | 4.55 % |
| z ∈ [0.45, 0.90) | 24,218 | 46.09 % |

The steepest approach attainable anywhere in the table band (z ∈ [0.00, 0.10)) is
**42.3° off vertical**. Only 1.88 % of that band can even reach 45°-down.

> **Consequence.** Every classical benchmark task that assumes a top-down grasp or a
> top-down insert — RLBench `insert_onto_square_peg`, ManiSkill `PegInsertionSide`'s
> vertical variants, kitting into a shaped tray recess, AutoMate's "top-down assemblable"
> selection criterion — is **infeasible at table height on this arm**. Precision tasks
> here must be *horizontal*. This is the single most important design constraint, and it
> is what makes the suite arm-specific rather than a benchmark port.

### C2 — The *simulated* gripper is ~25× weaker than the real one

The RS-rebot-dev-arm is rated to hold about **2.5 kg**, and its USD carries the real
`drive:linear:physics:maxForce = 500`. But the same drive is authored with
**`stiffness = 100`** (N/m — the stage is `metersPerUnit = 1`, so this is not the
degree-vs-radian scaling trap the repo already documents for the revolute joints). A
position drive only produces force in proportion to its tracking error, and a finger
closing on a 34 mm object can only be in error by ~17 mm, so `maxForce` is never
approached:

    F ≈ 100 N/m × 0.017 m ≈ 1.7 N per finger

Measured sweep (34 mm object, μ = 1.0, 6 repeats per cell; held = still in the fingers
after being lifted off a support pedestal). "Squeeze" is the applied finger drive force
read back from the sim:

| finger stiffness | squeeze | 0.05 kg | 0.25 kg | 1.0 kg | 2.5 kg |
|---|---|---|---|---|---|
| **100 (as authored)** | 1.76 N | 6/6 | **0/6** | 0/6 | 0/6 |
| 500 | 8.81 N | 6/6 | 1/6 | 0/6 | 0/6 |
| 2000 | 43.2 N | 6/6 | **6/6** | 0/6 | 0/6 |
| 8000 | 158.0 N | 6/6 | 6/6 | 0/6 | 3/6 |
| 30000 | 0.01 N | 0/6 | 3/6 | 0/6 | 1/6 |

Measured squeeze tracks the prediction exactly (1.76 N at k = 100, 43 N at k = 2000),
which confirms the drive stiffness — not `maxForce`, not friction — is the binding
constraint.

**Conclusions.**

1. **As authored, the simulated gripper holds ~0.05–0.1 kg, not 2.5 kg.** This is an asset
   fidelity gap, not a property of the hardware, and it explains the repo's otherwise
   puzzling note that 0.35 kg YCB masses were "far too heavy for this small arm" and its
   decision to override every object to 0.04 kg. The existing validation suite would not
   have caught it: `VALIDATION.md`'s snap-to-limits test only checks free-space position
   tracking, which passes at any stiffness because there is nothing to push against.
2. **Raising finger stiffness to ~2000 N/m makes 0.25 kg reliable** and costs nothing else.
3. **2.5 kg is not yet certified.** k = 8000 reaches 3/6 at 2.5 kg but 0/6 at 1.0 kg — the
   non-monotonicity means the remaining failures are contact/solver dynamics in this
   deliberately harsh open-loop swing test, not a friction limit. k = 30000 destabilizes
   the drive outright (squeeze reads 0.01 N and results go incoherent).

**Design rule adopted:** tasks use 0.04–0.25 kg objects and the challenge envs raise finger
stiffness to 2000 N/m via an actuator override, keeping the arm joints at their validated
USD gains. Anything heavier needs the gripper drive properly characterized against the real
hardware first — worth doing, and out of scope here.

Harness: `scripts/analysis/gripper_stiffness_sweep.py`.

> Note on an earlier revision of this document: a first pass reported a "50 g payload
> ceiling". That number was depressed by a retention test that scored an object still in
> the fingers but displaced 55 mm as a drop. The table above uses a criterion that
> separates a genuine fall (dz ≈ −2.2 m) from a slip.

### C3 — Grasp width: the gripper opens **89 mm on command, ~120 mm if forced**

`_GRIPPER_OPEN = 0.045` is a *per-finger prismatic joint value*, not an opening, and **both**
fingers move. Measured (`scripts/analysis/gripper_stroke.py`):

| quantity | value |
|---|---|
| finger separation vs joint value | `sep = 2.000 * q`, exactly, and zero at `q = 0` |
| clear gap calibration | `gap = 1.0035 * (q_left + q_right) - 1.25 mm`, max residual **0.035 mm** |
| opening at the commanded `q = 0.045` | **89.1 mm** |
| joint limits (left, right) | 0.0500 m, 0.0715 m — asymmetric |
| **maximum forced-open gap** | **~120 mm** |

The calibration is fitted on 30 / 45 / 70 mm gauges obtained by *rotating* the same block,
not by scaling it — see C8.

The last row is the one with consequences. The binary gripper action can only *ask* for
89 mm, but an object wider than that simply pushes the fingers apart on contact, up to the
sum of their joint limits. So **the widest graspable object is ~120 mm, not 89 mm**, and
nothing that fits this arm's usable workspace (C9) can be too wide to grasp. That is what
disproved the pre-grasp task's premise.

> **The previous "26–42 mm sweet spot" figure is withdrawn.** It came from
> `grasp_envelope.py`, which varied object width by writing a USD `xformOp:scale`, and its
> widths only spanned 18–42 mm, so it never tested anything near the real limit. Its hold
> rates rose monotonically with width right up to the largest size tested, which is not a
> sweet spot with an upper bound — it is a curve that had not reached one. Treat that sweep
> as unvalidated until C8 is settled for the path it uses.

### C4 — Working envelope (kinematic)

Near-horizontal approach available at object height (z ∈ [0.01, 0.08)) out to r = 0.67 m
kinematically, but the repo's trained policies only ever grasped inside **r ≤ 0.32 m**,
and roll freedom degrades with radius (11/12 roll bins at r = 0.10–0.15, down to 5/12
beyond r = 0.30). Design target: **r ∈ [0.15, 0.32] m, |azimuth| ≤ 60°**.

This is a *kinematic* bound. C9 is the one that actually constrains fixture design.

### C5 — Where a horizontal socket can go

For an insertion the wrist must translate along the socket axis *without re-orienting*, so
some gripper heading must be attainable at both the bore mouth and the fully-inserted
pose. Checking 90 candidate sites at 50 mm insertion depth: **37 pass**.

- r = 0.20–0.28 m: passes at most heights.
- r = 0.32–0.36 m: almost never passes — the arm runs out of wrist freedom.
- z = 0.03–0.18 m: all workable; z = 0.08–0.18 is richest (up to 4 common headings).

**Chosen socket site: r ≈ 0.22 m, azimuth 0°, z ≈ 0.12 m.**

(These counts are a conservative lower bound: at ~13 samples per voxel the 30°-heading
bitmask under-reports. A site marked OK is definitely OK; a site marked `none` may still
be reachable.)

### C6 — Asset availability

Only four YCB props exist on the Isaac 6.0 asset server: `003_cracker_box`,
`004_sugar_box`, `005_tomato_soup_can`, `006_mustard_bottle`. Everything else 404s. All
new geometry in this suite is therefore **procedurally authored** — which is fine, because
a *bore* cannot be made from a convex primitive anyway and has to be assembled from boxes.

### C7 — Compute

11 GB RTX 2080 Ti. Factory-grade contact settings (192 solver position iterations,
`gpu_max_num_partitions=1`) fit in memory at 128–512 envs but are roughly an order of
magnitude more expensive per env-step than the current pick-place settings. Budget
**512–1024 envs** for contact-rich tasks, keeping the existing 4096 only for the tasks that
stay contact-light.

---

### C8 — A post-build `xformOp:scale` never reaches the PhysX collider

Scaling a rigid body's prim *after the scene is built* changes only what is drawn. Measured:
a block scaled `z x 0.5` and dropped rests at **32 mm**, not the 17.5 mm its visual size
implies (`scripts/analysis/gripper_stroke.py`, section 3).

Consequences: object size cannot be varied per environment at runtime, and any sweep that
varied a dimension this way measured one size repeatedly. Vary size by rebuilding the env
(one process per size), or by **rotating** the object to present a different face.

`grasp_envelope.py` writes its scale through `Sdf` *before* `sim.reset()`, which is a
different and possibly supported path — that specific case has not been re-tested, so its
width axis is flagged unproven rather than wrong.

### C9 — The TCP cannot go below ~44 mm above the table

`reachability_map.py` is pure kinematics: it writes joint states and reads link poses, so it
does not know the table exists. The gripper is a bulky body and bottoms out on the table long
before the TCP reaches it. Measured by commanding poses **through the env's own action
manager** and reading back the achieved TCP (`scripts/analysis/tcp_floor.py`):

| | |
|---|---|
| commands below the floor all land at | z = 0.039 – 0.055 m |
| **usable TCP floor** | **~44 mm above the table** |
| reliable band | x ≈ 0.22 – 0.26 m, z ≈ 0.045 – 0.10 m |
| x = 0.30 | unusable below z ≈ 0.10 m (14–48 mm tracking error) |

This is the single most restrictive constraint in the suite, and it is why the drawer
cabinet stands on a 45 mm plinth: its handle was originally at z = 26 mm, i.e. permanently
out of reach. Any fixture feature the gripper must touch has to sit above this floor.

### C10 — The tool centre point is 41.9 mm behind `gripper_end`, not 75 mm

With the fingers shut the two finger bodies' origins coincide, and that coincident point is
the grasp point. It sits at `(-0.0419, 0, 0)` in the `gripper_end` frame. The value
inherited from the lift task was `-0.075`, i.e. **33.1 mm too far forward**.

A constant TCP offset error is nearly invisible in a reach reward — the policy just learns a
shifted target — but it is fatal to any scripted grasp, because the fingers are commanded to
close 33 mm past the object and shut on air every time. It produced several flatly wrong
"this object cannot be grasped" measurements before it was caught. All four challenge envs
now bind their `ee_frame` to `mdp.TCP_OFFSET`.

## 2. Reference points from the literature

Numbers that calibrate "how tight is tight", from the NVIDIA assembly line of work:

| | clearance | success box | sensing | success |
|---|---|---|---|---|
| Factory (RSS'22) | 0.114 mm diametral (shipped asset) | 2.5 mm lateral × 1 mm vertical | proprioception + poses, **no F/T** | sim only |
| IndustReal (RSS'23) | 0.5–0.6 mm on 8/12/16 mm pegs | — | **no F/T** | 88.6 % sim / 83.3 % real insert |
| AutoMate (TOG'24) | 1.0 mm diametral | 3 mm height | proprioception (+point cloud for the generalist) | 81.5 % sim / 86.5 % real |
| FORGE (RA-L'25) | Factory assets | — | adds EE force from joint torques | 0.84 peg / 0.98 gear real |

### Tolerance is the axis, not object diversity

The cleanest evidence in the literature that *placement tolerance* — not object variety —
is what separates easy from hard: ManiSkill2's own ablation holds the task, the policy and
the objects fixed and changes only the clearance. `PegInsertionSide` goes from **0.01 to
0.74** when the clearance is widened 10×. Swapping objects moves almost nothing; tightening
the hole moves everything.

Where the existing task sits against published success criteria:

| source | position tol. | orientation tol. |
|---|---|---|
| Meta-World | 50 mm | none |
| **`Rebot-PickPlace-v0` (this repo)** | **±50 mm** | **none** |
| ManiSkill AssemblingKits | 20 mm | 4° |
| Ravens / Transporter | 10 mm | 15° |
| FurnitureBench | 7 mm | ~16° |
| Isaac Lab Stack | 40 mm xy / 5 mm height | — |
| Isaac Lab place-upright | — | 5.7° |
| ManiSkill `PegInsertionSide` | 15 mm axial | within hole radius (3 mm clearance) |
| Factory / FORGE / AutoMate | 2.5 mm lateral × 1 mm vertical | — |
| **`Rebot-PrecisionSlot-v0` (this suite)** | **1.5 mm lateral** | **6.9° (0.12 rad)** |

The current basket is looser than every benchmark in the table, which is a large part of
why 87.9 % was reachable at all.

Two further conclusions carried into the design:

1. **Force sensing is not required.** Factory and IndustReal both solve sub-mm insertion
   from pose observations alone. What substitutes for force is an **LSTM policy**, an
   **asymmetric critic** that sees privileged object poses, and **per-episode constant
   noise on the observed target pose** (Factory uses σ = 1 mm) which forces a search
   behaviour. This arm has no F/T sensor, so it follows Factory, not FORGE.
2. **Target 0.5–1.5 mm radial clearance, not 0.1 mm.** IndustReal's 0.5–0.6 mm on a
   research-grade Franka is the realistic ceiling; a low-cost 6-DOF arm should start
   looser and tighten as a difficulty axis.

Reward recipe adopted from Factory (`factory_utils.squashing_fn`,
`1/(exp(a·d) + b + exp(−a·d))`) at three length scales, plus a staged
engaged/success bonus. The manager-based port already exists in Isaac Lab at
`isaaclab_tasks/manager_based/manipulation/deploy/mdp/rewards.py` and uses 6-DoF
±xyz keypoints rather than Factory's collinear ones — the right choice here, since this
arm cannot lock roll and pitch the way Factory's action space does.

---

## 3. The suite

Each task isolates one skill the current pick-place does **not** exercise. Common thread:
none of them can be solved by the existing "reach, close, carry, open" primitive.

### A. `Rebot-PrecisionSlot-v0` — tight-tolerance horizontal insertion

**Skill.** Sub-centimetre alignment and a committed straight-line insertion. The current
task's placement tolerance is ±50 mm; this is ±1.5 mm, a ~30× tightening, and unlike a
basket the bore *rejects* a misaligned approach instead of funnelling it.

**Scene.** A fixture at r = 0.22, az = 0°, z = 0.12 (per C5) with a rectangular bore facing
the robot, built from four static boxes plus a chamfered lead-in. The peg is a procedural
30 × 30 × 70 mm box at 0.04 kg (per C2/C3) spawning on the table inside the graspable
envelope.

**Difficulty axis.** Radial clearance as a config parameter: 3.0 / 1.5 / 1.0 / 0.5 mm.

**Success.** Peg inserted past a depth threshold with its axis aligned within tolerance,
held for a settle window.

### B. `Rebot-PreGrasp-v0` — non-prehensile reconfiguration before grasping

**Skill.** The object starts in a pose that is *provably ungraspable*: flush against a back
wall, where the only grasp that would work is top-down — which C1 rules out. The policy
must first drag or push the object away from the wall (a non-prehensile action with no
reward of its own), and only then grasp and place it. Reaching directly for the object can
never succeed, so the shaped "reach the object" gradient that drives the current task is
actively misleading here.

**Why it is not just "a harder spawn".** The existing v0/v1 `lying_prob` still leaves the
can side-graspable. Here the initial state has zero grasp success by construction, and
that claim is validated rather than asserted (see V1 below).

### C. `Rebot-DrawerOrder-v0` — articulated object plus irreversible ordering

**Skill.** Interacting with an articulated joint (a prismatic drawer with a side-graspable
handle), plus strict precedence: the drawer must be opened before the object can go in.
Long-horizon with a hard gate rather than a smooth distance gradient.

**Scene.** A procedural cabinet at r ≈ 0.25 with one prismatic drawer, handle sized and
placed to be side-graspable per C3/C4.

### D. `Rebot-ClutterExtract-v0` — constrained retrieval from clutter

**Skill.** Collision-aware approach and gentle contact. The target sits in a tight row of
distractors spaced below the gripper's outer width; **moving** any distractor more than 2 mm,
or toppling it, terminates the episode. The policy must thread the gripper in precisely — the
"singulate by pushing neighbours apart first" route is available only if the pushed neighbour
ends within 2 mm of where it started, which in practice it does not.

**Scene.** Target plus four distractors in a row at object height, spacing a config axis.

⚠ **Constraint tightened 2026-08-03.** It was toppling alone (`up_z < 0.75`, ≈41° of tilt),
which a neighbour dragged across the table and set down upright passed cleanly. A scripted
expert measured at **73.3 %** under the old rule scores **16.4 %** under the new one; the
median old-rule "success" displaced a neighbour by 13.7 mm, and 22–25 % of episodes carried a
neighbour into the goal zone with the target. `Rebot-ClutterExtract-Lenient-v0` reproduces the
old behaviour so the old baselines stay re-runnable. See `docs/envs/clutter-extract.md`.

---

## 4. Validation ladder

No task is claimed to work until it clears every rung. The principle: **prove the task is
achievable without training a policy**, by exhibiting a scripted expert that solves it, and
prove the task is *non-trivial* by exhibiting a negative control that fails.

| rung | what it proves | how |
|---|---|---|
| **V1 Asset** | the geometry is what the config says | read the authored USD back with `UsdGeom.BBoxCache`; assert bore − peg = 2 × clearance, drawer travel, distractor spacing |
| **V2 Kinematics** | the arm can get where the task needs | look every task-critical TCP pose up in the measured reachability map (`query_socket_sites.py`); a site that fails is moved, not hoped over |
| **V3 Physics** | the scene behaves | objects settle without jitter or interpenetration; the drawer joint actually travels; distractors actually topple when nudged |
| **V4 Achievability** | **the task is solvable** | a scripted / IK expert completes it at a measured success rate. This is the load-bearing rung — it is what "validate the env is achievable" means |
| **V5 Negative control** | the task is *not* trivially solvable | the naive policy fails: direct grasp at the wall (B), placing before opening (C), barging through the row (D), off-axis push (A). Predicates must fire only where they should |
| **V6 Smoke test** | the MDP is wired correctly | obs dims and finiteness, reward finiteness, success predicate fires on teleport-to-goal and stays off nearby, terminations fire |
| **V7 Learnability** | there is a gradient to climb | short PPO run; success rate must rise measurably above the V5 baseline |

V4 and V5 together are the actual answer to "is this env achievable" — an upper bound
demonstrated by an expert, and a lower bound demonstrated by a naive baseline.

### Results so far — `Rebot-PrecisionSlot-*`

`scripts/test_precision_slot_env.py` (rungs V1, V3, V5, V6) **passes on all three
clearance variants**.

V1, geometry read back off the USD stage with `UsdGeom.BBoxCache`:

| task | authored gap | block | measured per-side clearance |
|---|---|---|---|
| `Rebot-PrecisionSlot-Loose-v0` | 36.000 mm | 30.000 mm | **3.000 mm** |
| `Rebot-PrecisionSlot-v0` | 33.000 mm | 30.000 mm | **1.500 mm** |
| `Rebot-PrecisionSlot-Tight-v0` | 31.000 mm | 30.000 mm | **0.500 mm** |

V5, the part that matters. The success predicate fires 16/16 at the home pose and stays
off for all four negative controls (laterally offset, over-yawed, barely entered, still on
the table). Separately, the *geometry itself* is shown to reject misalignment: a block
force-teleported into the slot at 0.6 rad of yaw and then allowed to settle relaxes to

| clearance | residual yaw | vs. the 0.12 rad success tolerance |
|---|---|---|
| 3.0 mm | 0.433 rad | still far outside |
| 1.5 mm | 0.154 rad | still outside |
| 0.5 mm | 0.000 rad | forced perfectly square |

That monotonic progression is the evidence that the clearance parameter is a real physical
difficulty axis and not just a number in a config — the tighter slot physically cannot
accept a yawed block, so a policy that arrives misaligned is rejected rather than
funnelled. It is the opposite of the basket, which funnels everything.

---

## 5. Implementation architecture

New package `source/reBot_RL/reBot_RL/tasks/manager_based/challenge/` with its **own**
`mdp/` subpackage.

Rationale: the existing `pick_place/mdp/common.py` hardcodes `OBJECT_NAMES` as a
module-level constant and `observations.objects_canonical` hardcodes a two-object
`torch.where` swap. Both are global to that package, so changing them to fit a new task
would change the environment behind the 87.9 % pick-place result and the whole
expert/BC/distillation pipeline downstream of it. A sibling package with its own mdp terms
costs a little duplication and risks nothing. `tasks/__init__.py` auto-discovers any new
directory with an `__init__.py`, so no central registry edit is needed.

Reused unchanged: `REBOT_ARM_CFG`, `_START_POSE`, the gripper constants, the
`FrameTransformerCfg` EE frame, the PhysX tuning, and the `ActionsCfg` (6 joint positions
+ binary gripper).
