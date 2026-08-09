# Step 1 — a good randomised workstation env

**Goal.** A reconstructed, photoreal, metrically-correct Isaac Lab environment
(`Rebot-Workstation-PickPlace1-v0`) in which the reBot arm sits at the user's real desk, a
stationary receptacle is placed at a random reachable spot, and the manipulable objects
spawn at random reachable poses with randomised orientation, mass, friction and size.

**Not in Step 1.** No reward shaping beyond a placeholder, no policy training, no expert.
Step 1 succeeds when the env builds, randomises sanely, survives a smoke test with negative
controls, and every object in it is *proven* graspable rather than assumed to be.

**This plan will change.** It is written against measured constraints that were taken on a
different table with different objects; Phase 1 exists specifically to re-measure them.

---

## 0. What we have

**Capture** — `~/Desktop/isaacLab/data/captures/2026-08-05/`

| file | |
|---|---|
| `scene.MOV` | the desk, marker on an iPad at the robot's mount spot, movable objects removed |
| `object_rubixcube.MOV`, `object_rolloftape.MOV`, `object_tapemeasure.MOV`, `object_box.MOV` | one close orbit each, no marker |
| `measurements.txt` | ruler + calipers |

**Measured objects** (from `measurements.txt`):

| object | height | longest | mass | intended role |
|---|---|---|---|---|
| rubixcube | 56 mm | 56 mm | 73 g | grasp target |
| rolloftape | 24 mm | 91 mm | 42 g | grasp target |
| tapemeasure | 36 mm | 71.5 mm | 184 g | grasp target |
| box | 93 mm | 218 mm | 95 g | **stationary receptacle**, randomly placed |

Marker 165 mm on an iPad 6.5 mm thick → in the marker frame the **table surface sits at
z = −0.0065 m**, and the robot's base plate mounts there. Robot faces the marker's **left**
edge.

**Arm.** `measurements.txt` says *reBot-B601-RS*. **Confirmed by the user (2026-08-05) to be
the same arm as `RS-rebot-dev-arm`** — `source/reBot_RL/data/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda`,
8 DOF, which is what every env in this repo uses.

## 0a. Decisions taken 2026-08-05

Answers to the four open questions, from the user. These are binding.

1. **B601 = RS-rebot-dev-arm.** Use the existing asset.
2. **The arm mounts exactly where the marker was**, on the desktop, offset down by the
   iPad's thickness. Marker frame origin is the marker's *top surface*, so the base plate
   sits at **z = −0.0065 m** and the desk surface is that same plane.
3. **All three objects must be grasp targets.** The "demote two of them to clutter" fallback
   in B1 is *rejected* — B1 has to be solved, not designed around.
4. **The box is an open container and objects must be placed *into* it.** B2 option (a):
   author an open box to the measured outside dimensions, reconstructed mesh as visual only.

---

## 1. Two blockers found before writing any code

Both come from this repo's own measured envelope (`docs/CHALLENGE_SUITE.md`), not from
guesswork. Both need a decision.

### ⚠ B1 — Two of the three grasp targets are below the TCP floor

**C9:** the gripper is a bulky body that bottoms out on the table; the TCP cannot get below
**~44 mm above the table surface**, measured through the action manager, not kinematics.

**C1:** there is **no top-down grasp** anywhere near table height — 0.00 % of the
z ∈ [0, 0.10) band is top-down capable, and the steepest approach available is 42.3° off
vertical. Every grasp here is a side grasp.

Against the captured objects:

| object | height | TCP floor at 44 mm | verdict |
|---|---|---|---|
| rubixcube | 56 mm | grasp band 44–56 mm | **marginal** — only the top 12 mm is reachable |
| tapemeasure | 36 mm | object top is 8 mm *below* the floor | **infeasible as-is** |
| rolloftape | 24 mm | object top is 20 mm *below* the floor | **infeasible as-is** |
| box (receptacle) | 93 mm | rim well above the floor | fine |

This is the same constraint that put the drawer cabinet on a 45 mm plinth.

**The user has ruled that all three must be graspable**, so this must be *solved*. Options,
in the order P1 should try them:

- **(a) Re-measure, first and regardless.** The 44 mm figure came from `tcp_floor.py` on the
  old flat-table env, and it is position-dependent (reliable band x ≈ 0.22–0.26 m; x = 0.30
  is unusable below z ≈ 0.10). Measure it on *this* table before designing around it.
- **(b) A real riser.** Put the flat objects on a book / tray / block that exists in the
  scene, so it reconstructs photoreally along with everything else. 45 mm of riser makes the
  tape measure's top land at 81 mm and the tape roll's at 69 mm — both comfortably above the
  floor. **Cost: this object has to be in the capture.** If it is not, a short extra capture
  is needed. This is the leading option.
- **(c) Edge grasp.** Near the desk edge the fingers can hang below the tabletop, so a flat
  object can be pinched from the side with the TCP below 44 mm. Real technique, but it
  confines those objects to a strip at the table edge and so weakens "random reachable
  location".
- **(d) Synthetic plinth.** Cheapest, and it is what the drawer cabinet does — but it puts a
  non-existent object into a photoreal scene, which defeats the point of the reconstruction.
- **(e) Re-mount the arm.** Not promising. The floor comes from the gripper body fouling the
  table near the arm's own base, so raising the base is as likely to hurt as help, and
  sinking it below the tabletop is impossible with the arm mounted *on* the desk.

**Recommendation: (a), then (b).** If the re-measurement confirms the floor, ask the user for
one more short capture of a book or tray to act as the riser rather than authoring a fake one.

### ⚠ B2 — The reconstructed box is a solid, not a container

`extract_object.py` adds a **synthetic flat base** and Poisson-closes the mesh
(`workstation/tools/extract_object.py:313`). That is correct for the cube and for anything
resting on a table — and wrong for a receptacle. A closed 218 × ? × 93 mm block has no
interior, so nothing can be placed *into* it.

**Decided (2026-08-05): option (a)** — author an open box from five thin cuboids at the
measured outside dimensions, and use the reconstructed mesh only as the *visual*. Metrically
faithful, physically correct, and the same trick `CHALLENGE_SUITE.md` C6 uses for the drawer
bore. Success is "object inside the box", not "on top of it".

Two things to settle while authoring it: the **wall thickness** (measure the real box) and
the **interior clear width** — an object must fit through the mouth while held in fingers
that are themselves ~16 mm thick per pad, which is what the slot task learned the hard way
(`eva_bc/docs/slot/EXPERT_RESULTS.md` §1a: a gripper *holding* an object clears a channel at
a completely different height than an empty one).

Also note the box is **218 mm long inside an r ∈ [0.15, 0.32] m reach annulus (C4)** — it
consumes a large fraction of the reachable workspace. Placement sampling has to guarantee it
does not swallow the region the objects spawn into.

---

## 2. Constraints every design decision here must respect

Carried from `docs/CHALLENGE_SUITE.md`; these are measured on this asset in this sim.

| id | constraint | consequence for this env |
|---|---|---|
| C1 | no top-down grasp below z ≈ 0.10 m | side grasps only; yaw randomisation must keep a side face presentable |
| C2 | authored finger stiffness 100 N/m ⇒ ~1.7 N squeeze, fails at 0.25 kg | **must** split the actuator into `arm` / `fingers` and override to 2000 / 40; the 184 g tape measure sits right at the failure cell |
| C3 | gripper opens 89 mm commanded, ~120 mm forced | cube 56 ✓, tape measure 71.5 ✓, tape roll 91 mm across its diameter ✗ but 24 mm across the rim ✓ |
| C4 | design target r ∈ [0.15, 0.32] m, \|azimuth\| ≤ 60° | the spawn annulus, and the box placement region |
| C8 | **a post-build `xformOp:scale` never reaches the PhysX collider** | size randomisation cannot be done at runtime — see §4 |
| C9 | TCP floor ~44 mm above the table | B1 above |
| C10 | TCP is `(-0.0419, 0, 0)` from `gripper_end`, not `-0.075` | bind `ee_frame` to `mdp.TCP_OFFSET` |

Repo conventions: quaternions are **(x, y, z, w)**; observation terms return **(N, 1)**;
data accessors use `.torch`; robot base and table top both at the origin, +z up, +x away
from the robot — which lines up cleanly with the marker frame the reconstruction emits.

---

## 3. Phases

### P0 — Reconstruction (Re3Sim side)

Runs entirely in `~/Desktop/isaacLab/Re3Sim`, output into **fresh** `data/scene2/` and
`data/object2/` so the existing working scene stays intact.

1. `video_to_frames.py --target 300 --aruco DICT_6X6_250` on all five videos. **Gate:** the
   ArUco report on `scene.MOV` — if the marker is visible in well under half the kept
   frames, reshoot rather than push a weak alignment downstream.
2. COLMAP sparse per capture. **Gate:** registered fraction and reprojection error against
   the last run's 293/300 @ 0.54 px.
3. `align_to_marker.py --marker-size 0.165` on the scene. **Gate:** alignment RMS vs the
   2.8 mm achieved last time.
4. 3DGS 30 k iters + OpenMVS dense, both as `systemd --user` units (never foreground).
5. `fit_table.py` → `table_bounds.json`. **Gate:** fitted surface z should land within ~1 mm
   of −0.0065 m; that is an independent check on the 165 mm marker measurement.
6. `crop_and_scale_gs.py` → `ply_to_nurec_usd.py` → `splats.usd`.
7. `extract_object.py` per object with `--target-height` from `measurements.txt`, then
   `mesh_to_rigid_usd.py`. **Gate:** reconstructed aspect ratio vs the measured one — the
   cube's previous run came out 64.5 × 63.9 × 50.4 against a true 57 mm, and the aspect
   check is what catches that.

**Deliverable:** `splats.usd`, `table_bounds.json`, four object USDs with per-object
measured mass and dimensions, and a short reconstruction report.

### P1 — Feasibility gate (before the env is written)

The repo's culture is to measure before designing, and B1 is exactly why.

1. Build a throwaway scene: reconstructed table collider + arm at the marker origin.
2. Re-run `scripts/analysis/tcp_floor.py` on **this** table. Does the 44 mm floor hold?
3. Per object, at the measured height and a sweep of yaws, ask: is there an attainable
   side-grasp pose? Reuse the pattern from `scripts/analysis/grasp_geometry.py`.
4. Sample the reach annulus for the box footprint — where can a 218 mm slab go without
   eating the spawn region?

**Deliverable:** a measured go/no-go table per object, and the B1/B2 decisions closed.
**This is the phase most likely to change the rest of the plan.**

### P2 — Scene skeleton

New package `source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/`, mirroring
`challenge/clutter_env_cfg.py` in layout (`*_env_cfg.py` + `mdp/`).

- splats as a visual-only `AssetBaseCfg`; analytic box collider for the tabletop
- arm at the marker origin, yaw from `robot_facing: left`
- objects as `RigidObjectCfg` with measured masses
- `arm` / `fingers` actuator split (C2), `ee_frame` bound to `TCP_OFFSET` (C10)

**Smoke:** env builds, arm does not fall through the table, all objects settle within
tolerance of the surface and stay put under a null action.

### P3 — Randomisation

**Object pose.** Rejection sampling in the reach annulus: draw (r, θ) in
r ∈ [0.15, 0.32], |θ| ≤ 60°, reject on overlap with another object, with the box footprint,
or outside the fitted tabletop. Yaw **uniform on [0, 2π)**; roll and pitch **fixed** — the
user's constraint, and it also keeps the synthetic underside hidden.

**The box.** Randomly placed but must not move: spawn as a rigid body with
`kinematic_enabled=True` and set its pose in the reset event. Kinematic is the honest way to
express "stationary" — mass tricks leak, and at 95 g this one would be shoved on first
contact.

**Domain randomisation.**

| property | mechanism | notes |
|---|---|---|
| mass | `randomize_rigid_body_mass` event | centred on the measured value; keep the tape measure below the C2 failure cell |
| friction / restitution | `randomize_rigid_body_material` event | runtime, standard |
| **size** | **build-time only** | **C8: runtime scale does not reach the collider.** Pre-build *k* scaled variants of each object USD in P0 and pick per-env at construction. Size then varies *across* envs but is fixed within one. |
| visual | lighting / material jitter, optional | `lift_vision/visual_randomization.py` has the pattern |

Size DR is the one request that cannot be honoured the obvious way — C8 is measured, with a
block scaled `z × 0.5` resting at 32 mm instead of 17.5 mm. The `Sdf`-before-`sim.reset()`
path is flagged unproven in C8 and is worth one experiment before falling back to variants.

**Test:** N resets, assert no interpenetration, everything inside the tabletop, everything
inside the reach annulus, and the box never moves under a null action.

### P4 — MDP skeleton + smoke test

Minimal but well-formed: observations (proprio + object poses relative to TCP), the
repo's standard action term, terminations (timeout, object dropped off table, box moved),
and a placeholder reward. `scripts/test_workstation_env.py` following
`scripts/test_clutter_env.py`, **with negative controls** — the repo expects them and every
other env doc reports them.

### P5 — Documentation

Fill in `docs/envs/re3sim/` to the standard the other four envs set: what the env is, exact
scene geometry, MDP, the measured constraints that forced the design, how to run it, and
validation evidence. Add the row to `docs/envs/README.md`.

---

## 4. Risks, ranked

| risk | likelihood | mitigation |
|---|---|---|
| B1 — most objects ungraspable at table height | **high** (measured on a similar setup) | P1 measures it first; fall back to cube-as-target |
| Size DR blocked by C8 | **high** (measured) | pre-built variants; one experiment on the `Sdf` path |
| Object reconstruction inaccurate | medium | caliper cross-check per object; `--target-height` |
| Box swallows the reach annulus | medium | P1 samples placements explicitly |
| Marker weak in the scene capture (iPad glare, oblique views at the mount spot) | medium | `--aruco` gate at P0.1; reshoot rather than push through |
| B601 ≠ RS-rebot-dev-arm | low | ask; only P2 onwards is affected |
| 10 GB VRAM contention with other jobs | low | `nvidia-smi` before each GPU stage; reconstruction stages run as systemd units |

## 5. Open questions

All four original questions were closed on 2026-08-05 (§0a). What remains open:

1. **Is there a riser in the scene capture?** If P1 confirms the 44 mm TCP floor, B1 needs a
   real object to stand the tape roll and tape measure on. If `scene.MOV` contains no
   suitable book/tray, a short extra capture is required.
2. **The real box's wall thickness and interior clear width** — needed to author the open
   box, and not in `measurements.txt`.
3. **Does the `Sdf`-before-`sim.reset()` scaling path work?** C8 flags it as unproven rather
   than wrong. One experiment decides whether size DR can be per-env instead of per-build.

None block P0. Reconstruction can start immediately and answers several of them.

## 6. Step 2 and beyond

Once the env exists the work moves to the expert → BC → RL ladder, planned separately in
[pickandplace1/](pickandplace1/). Read that before building the env, not after: several of
its constraints (the action interface, the `mdp` touchpoints the BC stack expects, the need
for a strengthened success predicate) are things the env has to provide, and retrofitting
them is more expensive than designing them in.
