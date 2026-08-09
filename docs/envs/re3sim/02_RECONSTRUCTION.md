# 02 — Reconstruction *is* environment setup

**Written 2026-08-06.** Supersedes the sequencing in `HANDOFF.md` §10, which had the Re³Sim
pipeline as a parallel track to be merged *after* BC training.

> *"I noticed in the plan, training 3dgs and basically the re3sim pipeline is AFTER training
> the BC policy. THIS IS COMPLETELY WRONG!! we should be doing this as part of the setting up
> pickandplace env! Setting up pickandplace env entails doing all of the re3sim pipeline NO
> CUTTING CORNERS!"*  — user, 2026-08-06

The correction is right, and not only on principle. **A policy trained against primitive
colliders is not a cheap first draft of the policy you want — it is a measurement of a
different task.** The grasp the expert finds depends on the collider it closes on: a 56 mm
analytic cube has perfectly flat faces and sharp edges, a reconstructed rubix cube has rounded
corners, a slightly convex face and a real centre of mass. Demonstrations generated against
the first are not demonstrations of the second, so every number measured before the swap has
to be re-measured after it. Doing reconstruction first is *cheaper*, not more virtuous.

So the order is now:

```
   capture ─► COLMAP ─► 3DGS  ────────────────► splats.usd  ─┐
                    └─► OpenMVS dense ─► fit_table ──────────┤
                                                             ├─► THE ENV
   object captures ─► COLMAP ─► OpenMVS dense                │
                          └─► extract_object ─► rigid USD ───┘
                                                             │
                                                  expert ─► demos ─► BC ─► eval
```

---

## 1. Two pipelines, and which one each asset uses

There are two distinct code paths in this repo and they are easy to confuse.

| | `Re3Sim/re3sim/scripts/reconstruct.py` | `Re3Sim/workstation/tools/*` |
|---|---|---|
| written for | the fork's own Franka scenes | this desk, by earlier sessions here |
| marker | 5×5 **ChArUco board** | one large **ArUco** marker |
| covers | COLMAP → 3DGS → OpenMVS | alignment, scale, table fit, crop, object extraction, USD authoring |
| status | patched (6 fixes, §5) | the validated last mile |

`reconstruct.py` gets you as far as a dense cloud and a trained splat. Everything that makes
those *metric and placeable* is in `workstation/tools/`. Re3Sim's own
`compute_transform_to_marker_aruco.py` is **not usable here** — it constructs
`CharucoBoard((5,5), 0.04, 0.03)` and this capture has a single 165 mm marker, so it detects
nothing.

### What gets 3DGS and what does not

**The scene gets splats. The objects do not, and that is Re³Sim's own design, not a shortcut.**
The method's split is appearance-vs-dynamics along a *static-vs-movable* line: the background
never moves and only has to look right, so it is carried as gaussians; every manipulable
object has to collide, be grasped and be re-posed each episode, so it is carried as a **mesh**.
Gaussians cannot do any of those three things. Training splats for the rubix cube would cost
~35 min of GPU and produce an asset the env has no way to use.

Concretely, that removes ~2.5 h of GPU from the critical path, which is why
`workstation/tools/recon_object.py` exists rather than reusing `reconstruct.py` for objects.

### What the objects do *not* get either

`ReconstructMesh` / `RefineMesh` / `TextureMesh` are skipped for objects too.
`extract_object.py` runs its own segmentation and screened-Poisson meshing **from the dense
point cloud**, because it first has to remove the table plane and the background. Meshing the
whole capture and then cutting the object out of it throws away the plane fit that makes the
segmentation work at all.

Object pipeline, in full:

```
images ─► COLMAP sparse ─► image_undistorter ─► InterfaceCOLMAP ─► DensifyPointCloud
       ─► extract_object.py --target-height ─► re-centre ─► mesh_to_rigid_usd.py
```

---

## 2. Measured results — scene capture (`data/scene2/`)

### 2.1 Marker detection

```
262/300 frames contain marker id=42 (DICT_6X6_250)
largest views: 410 px side, squareness 0.66
```

87 % of frames see the marker. The previous capture managed 75/95 (79 %) and produced a good
alignment, so this is comfortable.

### 2.2 Alignment and metric scale — **PASS, and better than the previous run**

`align_to_marker.py --marker-size 0.165`, solving marker position and scale **jointly** as one
linear least-squares problem in `(x_marker, 1/s)` (see the tool's docstring — it is not
"estimate pose then estimate scale"):

| | this run (`scene2`) | previous run | verdict |
|---|---|---|---|
| images used | 262 | 75 | |
| scale | 0.297717 m / COLMAP unit | — | |
| triangulation RMS | **15.20 mm** | 16.7 mm | **better** |
| orientation spread (median) | **0.66°** | 0.91° | **better** |
| orientation spread (max) | 125.93° | — | see below |

The 126° maximum is expected and harmless: a marker seen at a very oblique angle has an
almost-degenerate PnP and can flip. The *median* is what the joint solve is dominated by, and
0.66° over 262 views is tight.

**One number here is a measurement, not an inference, and that is an improvement over the
previous capture.** The old scene's marker was displayed on an iPad whose physical size was
unknown, so the marker side (186.9 mm) had to be *recovered* from screen geometry and iPad
panel DPI. This capture's `measurements.txt` gives `marker_black_square_mm: 165` from a ruler.
Everything metric scales linearly with that one number, so having it measured rather than
inferred removes the single largest source of systematic error in the previous scene.

### 2.3 Table fit — **provisional, one number needs the dense cloud**

Run against the **sparse** model, because MVS has not finished yet:

```
tabletop rectangle (marker frame)
  centre  : (+0.089, +0.141) m
  size    : 1.642 x 1.427 m
  yaw     : -0.01 deg
  surface : z = -0.0006 m
```

Two things to note, one reassuring and one open.

**Reassuring: the long edge is 1.642 m, matching the previous capture's fitted tabletop to
three decimals** (previous: 1.642 × 0.501 m). Two independent captures, two independent marker
sizes, two independent scale solves, and the same physical edge comes back the same. That is a
strong end-to-end check on the metric scale — far stronger than any internal residual.

**Open: the short edge reads 1.427 m against the previous run's 0.501 m.** A desk does not
change depth. The likely cause is that a *sparse* cloud within a ±40 mm band of the marker
plane picks up whatever else in the room happens to sit at desk height, and the rotated-rect
fit then stretches to enclose it. **This must be re-run on `mvs/scene_dense.ply`** — which is
what `finish_scene.sh` does automatically when the dense cloud exists. It is recorded here as
open rather than quietly used.

The fitted extent only feeds the **gaussian crop box**; the physics table is a flat collider at
z = 0 either way, so an over-wide fit costs visual clutter, not dynamics.

### 2.4 What is actually on the desk — checked, not assumed

`data/scene2/images/00150.jpg`, read directly rather than inferred from the file list:

* the ArUco marker is displayed on an **iPad lying flat on a light-wood desk**, near its
  front-left corner — consistent with the frame convention, and the reason the desk surface is
  one iPad below z = 0;
* **the manipulated objects are NOT on this desk.** The cardboard box sits on a *different*
  table behind it, and the cube, tape roll and tape measure are not in frame at all. This is
  the right way round and it was worth confirming: had they been captured in place, the splats
  would bake them into the background and every episode would render a ghost object beside the
  simulated one. Re³Sim captures the background empty for exactly this reason;
* the desk **does** carry a monitor, a bench PSU, a dev board and cable runs, several of which
  sit inside the arm's reach. `measurements.txt` says `needs_collision: none`, so they enter
  the env as **splat appearance with no collider**. That is the user's call and is recorded
  as such — it means the sim will let the arm pass through a PSU that the real arm would hit,
  which is a sim-to-real gap to close before any deployment, not a modelling error here.

### 2.5 The desk-height offset — **RESOLVED: use the calipers, 6.5 mm**

The env's frame convention (user decision #2) is: the arm mounts where the marker was, and the
desk surface is z = 0. The marker frame's origin is the marker's **top** surface — the iPad
screen — so the desk sits below it and

```
z_env = z_marker - surface_z
```

**Why it matters first:** 6 mm is 24 % of the cube's 25 mm grip height. Getting it wrong moves
the whole desk relative to the arm and invalidates the p01 grasp table, so it is worth three
independent looks rather than one.

| source | desk below the marker plane | |
|---|---|---|
| `measurements.txt` iPad thickness | **6.5 mm** | calipers, on the physical stack-up |
| plane fit over a ±40 mm band | **0.6 mm** | `fit_table.py`, sparse cloud |
| annulus 0.22–0.55 m, iPad excluded | **−4.9 mm** | `measure_desk_offset.py`, sparse cloud |
| previous capture's plane fit | 3.4 mm | for reference |

I wrote `measure_desk_offset.py` specifically to break the first tie, on the theory that the
plane fit is biased: the band it fits **contains the iPad's own top face, which *is* z = 0**,
and on a dark weakly-textured desk the reconstruction puts far more points on the textured
tablet than on the wood around it. Excluding the tablet by geometry — an annulus from 0.22 m
(clear of a 280 × 215 mm iPad) to 0.55 m — should then give the desk alone.

**It did not, and the reason is the useful part.** The annulus median came out at −4.9 mm with
an **interquartile range of 26.5 mm**. A desk surface does not have a 26 mm interquartile
range. Those points are not one surface: at 0.22–0.55 m from the marker this desk carries
cable runs, a strip of tape, a dev board and eventually its own edge — all visible in
`00150.jpg`. The control check is equally telling: points *under* the tablet, which sit on the
marker plane by construction, read −2.45 mm rather than 0.

**Conclusion: the sparse reconstruction cannot constrain this number, and the disagreement
between the two estimates is not evidence about the desk — it is evidence about what is
sitting on it.** The offset used is therefore the **caliper measurement, 6.5 mm**, which is a
direct reading of the actual physical stack-up rather than an estimator over a contaminated
point set. This also matches user decision #2, made before any of this was measured.

Recorded rather than silently chosen, because "the fit disagreed with the spec" is exactly the
kind of thing a later session would otherwise re-derive from scratch. The dense cloud will not
change the answer — it will have the same cables in it, just more densely.

---

## 3. Measured results — object captures

| capture | frames | pipeline state |
|---|---|---|
| `object2_rubixcube` | 300 | COLMAP mapping |
| `object2_box` | 300 | queued |
| `object2_rolloftape` | 300 | queued |
| `object2_tapemeasure` | 300 | queued |

Run sequentially by `workstation/tools/recon_all_objects.sh`, in order of consequence:
**rubixcube first** because it is the only grasp target and its collider is what the fingers
actually close on; then the box (visual + metrology); then the two clutter bodies.

### 3.1 The scale gate — and why the obvious one cannot work

Object captures have **no marker in frame**, so the reconstruction is metric only up to a
factor and `extract_object.py --target-height` takes that factor from the calipers.

**Scaling by a dimension you asserted makes that dimension unfalsifiable.** Whatever the
segmentation did to the object's base, the output is *exactly* the height you asked for. The
previous session's cube came out **50.4 mm against a true 57 mm** by this route and nothing in
the run said so.

`finish_object_asset.py` therefore holds the **second** measured dimension back and checks
against it. The failure mode is RANSAC plane removal eating the object's base when
`plane_thresh` is too generous, and eating the base shortens the raw *height* without
shortening the raw *footprint*. Scaling that short height up to the true one **inflates the
footprint**, so a longest-axis reading too large is a direct quantitative readout:

```
lost_base ≈ measured_height × (1 − measured_longest / reconstructed_longest)
```

The gate retries with `plane_thresh` at 3.0 / 2.0 / 1.25 / 0.75 × the cloud's own median point
spacing and keeps the best, failing loudly if none lands within 8 %.

### 3.1a ⭐ The gate I wrote could not fire — and it took a wrong asset to notice

The first cube asset built cleanly and the gate reported **+0.5 %, PASS**. The mesh measured

```
final extent: 28.0 x 22.9 x 56.3 mm
```

for an object that is 56 × 56 × 56 mm. **A quarter of the cube, passed as correct.**

The gate compared the reconstruction's *longest axis* against the measured longest. For a
cube the longest axis **is** its height — the same dimension used to set the scale — so the
scale factor makes that number right by construction no matter what the segmentation did. I
had reasoned carefully about holding a dimension back and then held back one that was not
independent.

**Generalisable:** a held-back check is only a check if it is *independent of the thing being
fitted*. "Different number" is not the same as "different degree of freedom".

Fixed: the spec now carries the **full measured bounding box** and the gate compares all axes
the sheet gives, sorted (orientation-free, since `extract_object` levels and centres a mesh
but does not choose which way round x and y come out).

| object | scale from | gate checks | mass |
|---|---|---|---|
| rubixcube | height 56.0 | 56.0 × 56.0 × 56.0 | 73 g |
| rolloftape | height 24.0 | 91.0 × 91.0 × 24.0 | 42 g |
| tapemeasure | height 36.0 | 71.5 × ? × 36.0 | 184 g |
| box | height 93.0 | 218.0 × ? × 93.0 | 95 g |

### 3.1b Why the segmentation went wrong: no ROI

The underlying failure was in locating the object at all. `extract_object.py` seeds on points
that rise clear of the support plane, and my seed height was a fraction of the 99.9th
percentile height **over the whole cloud**. Measured, for the cube capture:

```
48,566,234 points     the whole room
cube height           1.71 cloud units
99.9th percentile     4.29 cloud units      <- 2.5x taller than the cube, and not the cube
seed height (45 %)    1.97                  <- ABOVE the cube entirely
681 seed clusters; largest has 1988 points
```

So it seeded on background, took the largest cluster it found there, and expanded a footprint
around it. The resulting "cube" was a column of the room.

**Fix: locate the object from the camera geometry.** These captures are hand-held orbits of a
single object, so the cameras ring it and the centroid of their centres sits almost directly
above it. Projecting that centroid onto the support plane gives an ROI centre with no
appearance assumption at all — which matters here, because the rubix cube is deliberately
multi-coloured and `locate_object_by_color.py` has nothing to grip. The radius comes from the
median camera distance.

With the ROI in place the seed height is a fraction of the tallest thing *near the object*,
which is the object. The search then sweeps four seed fractions × four `plane_thresh`
multipliers and keeps the extraction whose measured axes agree best. The ROI crop is written
once and the whole sweep runs on it — sixteen passes over 48 million points to segment an
object described by about six thousand of them is not a search, it is a wait.

### 3.1c ⭐ THE REAL BUG: "up" was decided from the point distribution, not the cameras

**Everything in this section between roughly 02:00 and 08:30 on 2026-08-06 concluded that the
objects were absent from their own reconstructions. That conclusion was WRONG, and it was
wrong in the most expensive direction: it would have sent the user to re-shoot four videos of
objects that reconstruct perfectly well.** The corrected finding is below; the reasoning that
produced the wrong one is kept, because the failure is instructive and because a doc that
quietly deletes its mistakes cannot be trusted about its successes.

#### What actually happened

`extract_object.py` fits the support plane with RANSAC, which returns a normal of arbitrary
sign, and then picks the sign like this:

```python
h = points @ normal + d
if abs(percentile(h, 0.1)) > abs(percentile(h, 99.9)):
    normal, d, h = -normal, -d, -h          # "the object stands clearer of the table
                                            #  than any noise dips below it"
```

That reasoning is sound **when the object and sensor noise are the only things off-plane.**
These captures also contain the room. The far field is metres away on one side, so it owns the
larger tail, the test picks the wrong sign, and the normal ends up pointing **into the table**.
Every point above the surface then reads as *below* it and is discarded, leaving only the
flat, near-plane clutter that the clustering step dutifully reported.

Measured on the cube capture:

```
plane normal [0.107 0.006 0.994]  d -4.154
  percentile rule : p0.1 -0.772  p99.9 +4.263  -> flip? False
  CAMERA rule     : median camera height -4.273  -> flip? True      <-- they disagree
```

#### The fix, and why the cameras are the right authority

The cameras were above the table looking down at it. That is not an inference about the
scene's contents, it is how the capture was made, so the sign follows with no ambiguity:

```python
if median(camera_centres @ normal + d) < 0:
    normal, d = -normal, -d
```

`extract_object.py` now takes `--up-hint X,Y,Z`, and `finish_object_asset.py` derives it from
the COLMAP camera centres and passes it through. `check_object_present.py` uses the same rule.

#### What was there all along

An independent colour test found the cube immediately — a rubix cube is saturated
red/green/blue/yellow and newsprint is grey, which has nothing to do with planes or normals:

```
saturated (S>=180, V>=80): largest cluster 345,860 points
  bbox 1.176 x 0.903 x 0.988 units, max/min extent ratio 1.3      <- compact and near-cubic
  height above the CORRECTED plane: base -0.080, top +0.846       <- resting ON the table
  0.926 units tall; if that is the 56 mm cube then 1 unit = 60.5 mm
```

Re-running the presence check on all four captures with the corrected sign:

| capture | stands proud | expected | aspect | verdict |
|---|---|---|---|---|
| rubixcube | 0.821 units, 38,932 pts | ≥ 0.348 | 0.863 | ✅ **PRESENT** |
| box | 2.423 units, 1,416 pts | ≥ 0.598 | 3.689 | ✅ **PRESENT** |
| rolloftape | 0.485 units, 73,576 pts | ≥ 0.147 | 0.299 | ✅ **PRESENT** |
| tapemeasure | 0.682 units, 41,870 pts | ≥ 0.208 | 0.483 | ✅ **PRESENT** |

All four. The captures were fine. The pipeline was fine. One sign was wrong.

#### Three things this should have been caught by, and why each missed

1. **The presence checker inherited the same bug it was written to detect.** It was built from
   the same height-percentile idiom, so it confirmed the conclusion it was meant to test. A
   check that shares an assumption with the thing it checks is not independent.
2. **`rolloftape` passed even with the wrong sign**, which read as "the pipeline works, the
   other captures are bad" — the single most misleading possible outcome. It passed by luck:
   its cluster happened to clear the thresholds from the wrong side.
3. **Nobody looked at the point cloud.** Two hours of statistics about a cloud that could have
   been rendered. The colour test that settled it took four minutes to write.

**Generalisable:** when a pipeline reports that the *input* is bad, suspect the pipeline
first — that conclusion conveniently ends the investigation, and it is the one that costs the
user a re-shoot. Prefer a check that shares no machinery with the thing under test: colour
against geometry, cameras against points.

#### Per-object colour cues (user-supplied)

`extract_object.py`'s `segment_by_color` path is now used where colour is distinctive. It also
happens to sidestep the sign problem entirely, because it orients the normal *towards the
object it located* rather than by any distribution statistic:

| object | HSV window (OpenCV, H 0–179) | note |
|---|---|---|
| tapemeasure | `20,38,110,90` | yellow case |
| box | `8,25,60,80` | brown corrugated cardboard |
| rolloftape | — | geometry works |
| rubixcube | — | spans the whole hue circle; a single window would cut it up. Geometry + up-hint |


### 3.1f ⚠ The env's YAW was never set — found only by rendering the scene

The marker→env transform applied the height offset and **no rotation**. `measurements.txt`
says `robot_facing: left`, i.e. the arm's +x is a quarter turn from the marker's +x, and that
quarter turn was never encoded.

**Nothing automated could catch it.** The splats are visual-only — no predicate reads them —
so the smoke test, all eight `placed_mask` controls, the expert's success rate and the
policy's score are *bit-identical* whether the desk is under the arm or ninety degrees away
from it. It was found by rendering the scene and looking at it.

Measured, as the fraction of the object-spawn annulus (r 0.15–0.32 m, |az| < 65°, 2 cm cells)
that has reconstructed desk under it:

| yaw applied | workspace over real desk |
|---|---|
| **0° — what was shipped all night** | **14.0 %** |
| −90° | 66.9 % |
| **+180° — now** | **91.3 %** |
| +135° (argmax) | 97.1 % |

A top-down occupancy map of the desk-height gaussians in the marker frame shows why: the
marker sits at the desk's **near corner**, and in the env's +x direction the real desk ends
after about 10 cm. Every object was spawning over bare ground next to the table.

180° is used rather than the 135° argmax because it is the clean quarter turn that
`robot_facing: left` describes, and `fit_table` reports the desk square to the marker
(yaw −0.01°). **This number must match how the arm is physically bolted down** for the
sim-to-real transfer to hold, so it is a user decision, not a fitted parameter — the
reconstruction only says where the desk is, not where the arm will go.

Tooling: `workstation/tools/make_env_transform.py` composes both parts and is called by
`finish_scene_splats.sh`. `scripts/render_workstation.py` (in `eva_rl`) renders four
viewpoints for exactly this check; the *grazing* view is the one that exposes a height
mismatch and the *top-down* view the one that exposes a yaw error.

### 3.1g The dimension gate compared the wrong axes when one is unknown

Third gate bug, same family as §3.1a. `extents` is written descending with `None` where the
sheet is silent — but the scorer dropped the `None`s and truncated, which shifts every axis
past the gap:

```
tapemeasure best attempt [72.63, 68.77, 33.62] mm  vs measured [71.5, None, 36.0] mm
   compared 72.63 <-> 71.5   (+1.6 %)      correct
   compared 68.77 <-> 36.0   (+91.0 %)     WRONG -- that is the middle axis against the height
   GATE FAIL
```

The unknown axis is the **middle** one, not the smallest. A perfectly good mesh — +1.6 % on
the longest axis, −6.6 % on the height — was thrown away. Fixed by comparing **by rank**, so
position *i* of the sorted reconstruction lines up with position *i* of the measurement and
the unknown simply skips.

**Three gate bugs now, all of the same shape:** a check that looks rigorous, prints a
confident number, and is not actually testing what it claims. §3.1a compared a dimension to
itself; §3.1c-bis reported after it had already written; this one compared the wrong pair.
Each was caught only by looking at the *specific numbers* it produced rather than at its
verdict.

### 3.2 Origin convention — the trap in the swap

`extract_object.py` sits its mesh **base on z = 0**, because that is what "drop it on a table"
wants. Every constant in this env is calibrated for **centre-origin** primitives: `REST_Z` is
`size / 2`, and `placed_mask` reads `0.010 < z < 0.062` for a cube whose centre rests 34 mm up
inside the box.

Shipping base-origin assets would shift all of them by half an object — silently, since
nothing would error. `finish_object_asset.py` re-centres the mesh to its bounding-box centre
before authoring the USD, so **the swap changes appearance and collision shape and nothing
else.**

### 3.3 The box stays authored, and this is permanent

The box capture is reconstructed for its **appearance** and as a metrology check on
`BOX_OUTER_Y` (the one box dimension `measurements.txt` does not give). Its **physics stays
the authored open container**: `extract_object.py` adds a synthetic flat base and
Poisson-closes the mesh, so a reconstructed box is a *solid block* with no interior. This is
blocker B2 from the original plan, it is a property of watertight meshing rather than of this
capture, and no amount of re-running fixes it.

---

## 4. How the swap reaches the env

`workstation_env_cfg.py` now resolves each object through `_reconstructed_or(name, mass,
primitive)`:

* reconstructed USD present at `reBot_RL/data/workstation/objects/<name>.usda` → used;
* absent → the analytic primitive, **with a printed line saying so**;
* `REBOT_WORKSTATION_PRIMITIVES=1` → primitives forced, for a like-for-like comparison or to
  reproduce a pre-swap measurement.

Which path each object took is printed at load, so no run is ever ambiguous about what it was
measuring. Mass, friction and solver settings are re-asserted in the config rather than
inherited from the USD, so the two paths differ in **shape alone** and the domain
randomisation keeps one source of truth.

---

## 5. `reconstruct.py` needed SIX fixes, not four

Four were found in the previous session and are recorded in `HANDOFF.md` §7 (pycolmap
`FeatureExtractionOptions`; `SequentialPairingOptions`; the faiss-vs-flann vocab tree, worked
around with exhaustive matching; and the CUDA-less pycolmap wheel forcing CPU matching).

Two more surfaced today, and both would have fired *after* COLMAP, 3DGS and MVS had all
completed — the most expensive possible place to fail:

| # | symptom | cause | fix |
|---|---|---|---|
| 5 | `FileNotFoundError: 'colmap'` at `image_undistorter` | the Python side (`pycolmap`, `open3d`, `plyfile`, `pxr`) lives in **env_isaaclab6**; the `colmap` CLI and every OpenMVS binary live in **tools**. Neither env has both | append `~/miniconda3/envs/tools/bin` to `PATH` inside `reconstruct.py`. **Appended, never prepended** — prepending puts `tools/bin/python` ahead of env_isaaclab6's and the very first `import pycolmap` fails. PATH only: adding tools' `lib/` to `LD_LIBRARY_PATH` re-creates the CXXABI segfault |
| 6 | `ReconstructMesh` exits 1, writes nothing, pipeline marches on | `-p` is not an OpenMVS v2.1.0 flag; the binary takes `-i`/`-o`. The call was unchecked, so a missing mesh flowed into `RefineMesh` and the whole MVS stage was then marked complete | `-i`/`-o`, plus `check=True` on both mesh calls |

Also changed: the ChArUco alignment tail is now **skipped by default** with a printed pointer
to `workstation/tools/align_to_marker.py`. Previously it detected nothing, wrote no
`colmap_to_marker.npy`, and then an unconditional `shutil.copy` of that missing file killed the
run — after hours of work.

**Generalisable:** unchecked `subprocess.run` in a staged pipeline that records progress
converts a loud failure into a silent one. Every stage that a later stage depends on needs
`check=True`, or the progress marker is a lie.

---

## 6. Gates, and what each one would have caught

| gate | value | catches |
|---|---|---|
| marker frames ≥ ~50 % | 262/300 = 87 % ✅ | a capture where the marker is occluded |
| triangulation RMS vs 16.7 mm | 15.20 mm ✅ | a bad marker size, a bad scale solve |
| long table edge vs previous capture | 1.642 m vs 1.642 m ✅ | a scale error at the ~1 % level |
| desk surface, three independent looks | 6.5 mm adopted ✅ §2.5 | a frame offset that would shift every grasp |
| object longest axis vs calipers | pending | plane removal eating the object's base |
| reconstructed cube grasp rate vs 99 % | pending | a collider whose shape breaks the measured grasp |

The last one is the real acceptance test for the swap: `p01_grasp_feasibility.py` re-run
against the reconstructed cube should reproduce the primitive cube's 99 % at grip z = 25 mm.
If it does not, the mesh is wrong or the origin convention slipped, and demos generated
against it would be quietly worthless.

---

## ⭐ Bug found 2026-08-06 by filming the wrist camera: every object was a subdivision surface

`record_video.py` mounts the validated `WRIST_CAM_CFG` and films the policy from the gripper.
At close range the reconstructed cube renders as a **perforated lace shell** — you can see the
desk through it. Two separate causes, one of which is an authoring bug in the pipeline.

**Cause 1 — the meshes are genuinely not watertight.** `cube.usda` measured directly:

```
/Rubixcube/geometry: 2644 verts, 8000 faces, bbox 53.33 x 58.05 x 54.52 mm
edges 12256, boundary (hole-rim) edges 520 = 4.24 %
```

520 edges belong to exactly one face each, i.e. they are hole rims. Screened Poisson plus
decimation to 8 k faces left real gaps.

**Cause 2 — and this is the bug.** `mesh_to_rigid_usd.py` defines the `UsdGeom.Mesh` and
authors normals, but **never sets `subdivisionScheme`**. USD's fallback for that attribute is
**`catmullClark`**, so every object the pipeline has ever produced was authored as a
Catmull-Clark subdivision surface. The renderer is therefore asked to treat a decimated
triangle scan as a control cage and smooth it — which widens all 520 hole rims, and changes
the shape away from the extents the gate checked to ±10 %. Worse, per the `UsdGeom.Mesh`
spec the `normals` attribute is **ignored** whenever the scheme is not `none`, so the vertex
normals the tool carefully computes were never taking effect.

Fixed in `mesh_to_rigid_usd.py` (`mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)`), and
the three already-built assets were patched in place rather than re-meshed, since only this
one attribute changes:

```
cube.usda        /Rubixcube/geometry    catmullClark -> none
rolloftape.usda  /Rolloftape/geometry   catmullClark -> none
tapemeasure.usda /Tapemeasure/geometry  catmullClark -> none
```

**Scope of the impact.** Rendering only — the physics collider is a separate convex-hull
approximation and never saw the subdivision, so no expert or BC success number in
`03_EXPERT_AND_BC.md` is affected. It matters for the two things that depend on pixels:
vision distillation, and the real-world rollout the cube was reconstructed for in the first
place. Cause 1 (the holes) is NOT fixed by this and remains open — closing it needs either
hole-filling on the Poisson output or a denser capture.

**Lesson.** A geometry defect that three gates, a scale check and a physics smoke test all
passed was found by *pointing a camera at it*. The gates measured extents and mass; nothing
measured what the object looks like. Same lesson as the splat yaw (§5.5): render it.

### The real defect behind it: the objects were LATTICES, not shells

Rendering the mesh from six directions (filled triangles, coloured by the scan's own vertex
colours) shows what the numbers meant. The cube is recognisably a Rubix cube — correct 3×3
grid, correct sticker colours — **with the centre of every sticker punched out as a hole**.
Maximum triangle area is 20.9 mm² against a ~289 mm² sticker, so nothing spans them.

| | shipped | after the fix |
|---|---|---|
| distinct holes | **40** | **4** |
| boundary edges | 520 / 12,256 = **4.24 %** | **0.41 %** |
| convex hull / bbox | 0.674 | 0.634 |
| bbox vs a true 56 mm cube | 53.3 / 58.0 / 54.5 | 56.5 / 60.2 / 56.3 |

**Why the faces and not the grooves.** MVS matches pixels between frames. The black grooves are
high-contrast and match perfectly; a flat, glossy, uniformly-coloured sticker offers nothing to
match, and its specular highlight *moves* as the camera moves, so the same physical point looks
different in every frame. The reconstruction therefore recovered the cube's skeleton and none
of its faces — and `extract_object.py --density-quantile 0.06` then deleted exactly those
unmatched regions as low-confidence. `finish_object_asset.py` never passed the flag, so every
object ever built shipped at that default.

**The fixes.**

1. `--fill-holes` (default 30 mm) after decimation, not before — decimating a filled mesh
   re-opens it. Screened Poisson is watertight by construction, so this only restores what the
   trim removed. New vertices take the colour of the nearest original vertex, because
   `fill_holes` leaves them uncoloured and a black patch in the middle of a red sticker is
   worse than the hole was.
2. `--density-quantile` is now passed through, and is **per-object**. The cube wants 0.01; the
   others must stay at 0.06. MEASURED: at 0.01 the tape roll retains enough Poisson blob to
   inflate 91 mm → 127 mm and **fail its own gate**. One global value was wrong in both
   directions.
3. Material binding and `subdivisionScheme` — above.

### ⭐ The gate that let it through, and the one that replaces it

The scale gate measured **extents and mass**. *A hollow shell has exactly the same bounding box
as the solid it should be*, so the lattice passed at +4.8 % — a better score than the fixed
mesh gets. No amount of tightening that gate would ever have caught this.

`finish_object_asset.py` now also reports, and prints an explicit warning above 5 holes:

```
[rubixcube] TOPOLOGY  holes 4  boundary 0.41%  hull/bbox fill 0.634
```

* `holes` — distinct boundary loops. A closed surface has 0.
* `boundary` — fraction of edges belonging to exactly one face.
* `fill` — convex-hull volume / bbox volume. Solid box 1.00, sphere 0.52. Shape-dependent, so
  reported rather than thresholded.

**Lesson, and it is the third time this project has learned it.** Every gate so far measured a
*number the pipeline already believed*. The up-direction flip (§5.1), the splat yaw (§5.5) and
now the lattice were all found by rendering the thing and looking at it. Measure the artefact,
not the intent.

---

## ⭐ Are the gaussians healthy? Measured 2026-08-06

**Statistically, yes.** `/World/Splats/Gaussians`, a `ParticleField3DGaussianSplat`:

| metric | value | reading |
|---|---|---|
| count / SH degree | 356,768 / 3 | full view-dependent colour, not flat albedo |
| median scale | 1.13 mm (p90 8.5 mm) | millimetre-scale detail on a desk |
| oversized (> 50 mm) | 0.27 % | almost no blobby smear |
| opacity | median 0.51, 38.3 % > 0.9, 9.4 % < 0.05 | not a ghost field |
| spatial | 42.9 % at table ±20 mm, 39.6 % above, 9.3 % below | 82.5 % where the scene is |
| in arm workspace | 84,710 (23.7 %) | the reachable region is well populated |

**Visually, it depends entirely on the viewpoint, and that is the finding.**

| view | verdict |
|---|---|
| `top_down` | good — desk reads cleanly, wood grain and objects legible |
| `over_shoulder` | acceptable, mild speckle |
| `close_cube` | desk acceptable |
| `grazing` (near eye-level) | **poor** — a curtain of stringy needles below the desk edge |

The capture orbited *above* the desk. Views near that trajectory are interpolation and look
fine; a grazing view is extrapolation and shows the floaters 3DGS parked where no training
camera could see them. **This is the same root cause as the hollow objects: the reconstruction
knows only what the camera saw.**

### The prune, and why it does NOT ship

`prune_splats.py` was written to strip the 33,227 gaussians (9.3 %) below the table, on the
reasoning that a desk underside the capture never observed cannot contain anything real. It
does clean up the grazing view. It also makes the *good* views worse:

| prune | kept | dark speckle on the desk, top-down | grazing floaters |
|---|---|---|---|
| **none (shipped)** | 100 % | **3.25 %** | heavy |
| below −60 mm | 96.7 % | 4.30 % | reduced |
| below −20 mm | 90.7 % | 8.00 % | mostly gone |

**Gaussians are volumetric, not surface samples.** One centred 30 mm below the desk with an
8 mm scale still contributes to how the surface looks *from above*, so "below the table" is
not the same as "not part of the table" — which is exactly what the −20 mm threshold assumed.

The decision follows from which viewpoint the system actually uses: `WRIST_CAM_CFG` sits
171 mm from the TCP on a `tilt_x_m30` mount and looks steeply **down**, i.e. in the same
regime as `top_down`, where the unpruned field is best. The grazing view is a diagnostic angle
no camera in this system occupies. **Shipping unpruned.** The tool and the variants stay
available behind `RE3SIM_SPLATS` for anyone who needs the other end of the dial.

### Two measurement mistakes made while establishing this, both instructive

1. **A speckle metric that measured the wrong thing.** Counting high-local-variance pixels
   showed speckle *rising* after pruning. It rose because removing the below-table hair
   uncovered the sharp-edged grid floor behind it — the score was reading an occluder leaving,
   not artefacts arriving. Isolating the desk by hue and counting dark pixels *inside it* is
   the measurement that actually answers the question.
2. **Attributing the desk speckle to the opacity cut.** The first prune bundled `--below` with
   `--min-opacity`, the close-up got speckly, and the opacity cut looked guilty — plausible,
   since low-opacity gaussians are the blend layer between opaque ones. Re-running with
   `--below` alone showed the same speckle: the opacity cut contributes 0.55 of the 4.75
   points and the depth cut contributes the rest. **Two changes in one experiment cannot be
   attributed.**

### Also fixed: `prune_splats.py` and the strided SH array

`radiance:sphericalHarmonicsCoefficients` is **16 entries per gaussian** at SH degree 3
(5,708,288 = 356,768 × 16). Filtering the one-per-gaussian arrays and skipping this one leaves
every surviving gaussian wearing another gaussian's view-dependent colour — and that renders
as *plausible wrong colour*, never as an error. Any per-gaussian filter must handle stride.
(The `Quatf` orientations also cannot survive a numpy round-trip; that one at least raises.)

---

## The object3 re-capture (2026-08-07): better data, two new failure modes

The user re-shot all three objects after the lattice finding. **The captures are materially
better** — but they broke the pipeline in two new places, both caused by the same choice that
made them better.

| object | registered | dense cloud |
|---|---|---|
| rubixcube | 363/400 (COLMAP split into 4 sub-models; `sparse/0` is the good one) | 2.3 GB (was 1.3) |
| tapemeasure | **399/400** | 2.1 GB |
| rolloftape | **399/400** | 2.1 GB |

399/400 is about as good as photogrammetry gets. The reason is the **richly textured newspaper
background** the objects were shot on: MVS needs features to solve camera poses, and printed
flyers give it thousands. The old capture's plain surroundings are why it tracked worse.

### ⭐ Failure 1: metric scale, from a support plane 12.8 mm too low

Every tapemeasure attempt — ~30 combinations of seed height, plane threshold and footprint
expansion — returned **52-53 × 53-54 × 26.7 mm** against a measured 71.5 × ? × 36.

The tell is that they all *agreed*, and that the aspect ratio was right:

```
reconstructed 52.8 / 26.7 = 1.98        TRUE 71.5 / 36.0 = 1.99
```

**The shape was perfect; only the scale was wrong**, by a factor of 0.738. Inverting that:
a 36 mm object read as 48.8 mm, i.e. the fitted support plane sat **~12.8 mm below the surface
the object actually rests on** — the thickness of the stack of flyers. `--target-height` scales
by *height above the fitted plane*, so it is only ever as good as the plane.

**Fix:** `finish_object_asset.py --scale-by longest` passes `--target-longest` instead. The
longest axis is measured on the object itself and never involves the plane. Result on the
first attempt, no search needed:

| tapemeasure | height-scaled | **longest-scaled** |
|---|---|---|
| extent | 52.8 × 53.9 × 26.7 mm | **69.5 × 70.3 × 34.1 mm** |
| gate | +25.7 %, failed after ~30 attempts | **+5.3 % PASS, attempt 1** |

Default remains `height` so nothing about the object2 assets changes.

### ⭐ Failure 2: the cube cannot be segmented out of a colourful background

The cube never got a usable ROI. Five approaches, all defeated by the flyers:

| approach | result |
|---|---|
| plane + largest cluster (the pipeline default) | a **6.2 mm slab** of flyer |
| `locate_object_by_color.py`, full hue circle | masks covered **20 % of image area**; centre landed 7.5 units off |
| connected saturated sparse clusters | percolate through the flyers; no stable cluster at any link radius |
| explicit ROI at the located centre, R = 0.75 / 0.95 / 1.2 | raw extent **exactly 2R** every time — a *tilted* plane cuts the flat floor into a wedge whose height grows with radius (z/2R constant at 0.47) |
| 3D and 2D hue-**diversity** (six hues in 56 mm) | **located it** — two independent methods agreeing to 0.4 units — but the crop is still 30 % saturated with all 9 hue bins present at every radius |

The last row is the honest conclusion: **the flyers are multi-hue at every scale**, so no colour
statistic separates a 56 mm cube from them. The reconstruction is fine; the *segmentation*
has nothing to grip.

**Decision: keep the object2 cube.** It is already rebuilt with all four pipeline fixes (40
holes → 2, boundary 4.24 % → 0.70 %, material bound, subdivision `none`) and gates at +7.3 %.
The grasp only ever touches its convex-hull collider.

### The tapemeasure is replaced, and hole COUNT is a bad metric

| | object2 | **object3 (shipped)** |
|---|---|---|
| longest | 73.3 vs 71.5 = +2.5 % | **70.3 vs 71.5 = −1.7 %** |
| height | 32.7 vs 36.0 = −9.1 % | **34.1 vs 36.0 = −5.3 %** |
| hull/bbox fill | 0.541 | **0.580** |
| holes | 14 | 26 |

object3 has *more* holes and is *obviously* the better mesh: the render shows object2 is a
**ring**, with one enormous void through its entire centre, while object3 is a solid body with
small nicks. Count says 14 < 26; reality is the opposite. **Use `hull/bbox fill` and the
render; treat hole count as a flag, not a score.**

### For the next capture

The textured background is worth keeping — it is why registration hit 399/400. Add a **plain,
matte, contrasting mat directly under the object**. Textured surroundings for the *cameras*,
clean separation for the *segmentation*. The two needs are different and this capture served
only the first.

### Also fixed here: the §5.4 rank bug, again, in the new tool

`inspect_object_mesh.py` dropped the unknown (`None`) axis before zipping measured against
reconstructed, so the tapemeasure's 69.2 mm **second** axis was scored against its 36 mm
**height** — reporting **+92 %** on a mesh that is really within 3 %. Exactly the bug already
catalogued in §5.4, reintroduced in a new tool the same day it was written. A `None` means
"no check at this rank" and must hold its position, not close the gap.

### rolloftape: object3 fails on height, object2 kept

With `--scale-by longest` the **diameter is near-perfect** — 90.9 × 90.2 mm against a true
91 × 91 — but the height reads **16.7 mm against 24 mm** and the gate fails at +30 %.
Tightening `plane_thresh` through the whole search only recovers to 17.7 mm. This is the
"plane ate the base" failure the module docstring already describes: RANSAC cutting ~7 mm up
into the roll. Note it is the *opposite* direction to the tapemeasure's plane error on the
same capture, which is why one global plane policy cannot serve every object here.

**Kept object2**, which gates at +4.1 %.

### ⭐ FINAL ASSET STATE after the object3 round

| asset | source | worst axis | holes | boundary | fill |
|---|---|---|---|---|---|
| `cube.usda` | **object2** | +7.3 % | **2** | 0.70 % | 0.657 |
| `rolloftape.usda` | **object2** | **+4.1 %** | 6 | 0.63 % | 0.550 |
| `tapemeasure.usda` | **object3** | **+5.3 %** | 26 | 1.24 % | **0.580** |

Two of three keep the older capture and one takes the newer — decided per object on measured
evidence, not on which capture is newer. The object2-derived versions of all three are
preserved in `data/workstation/objects_object2_backup/`.

**The object3 captures are not wasted.** They are better data (399/400 registered, 2.1-2.3 GB
dense) and both new failures are in *segmentation and scaling*, not in the reconstruction. A
plain mat under the object at capture time would very likely make all three usable, since it
fixes the plane fit and the ROI in one change.

**Verified in the env** (`renders/final_objects/`): all three objects render as closed solids
on the desk. They are lumpy and low-fidelity at close range — *solid* was the bar that was
failing, and photoreal they are not. The remaining fidelity ceiling is the capture, not the
pipeline.

---

## ⭐ Filming the EXPERT (2026-08-07): a third bug, and an open one

`collect_demos.py --record-video DIR` films env 0 from the wrist camera and a workstation
camera **through the executor itself**, not a re-implementation — a recorder that rebuilds the
control loop films a slightly different manoeuvre and the difference is invisible in the
output. Files land as `expert_wrist.mp4` / `expert_station.mp4`, labelled with whether env 0
actually succeeded so the film is never read as a success rate.

### FIXED: the desk is absent from every MULTI-ENV render

The first films showed the arm, the box and all three objects floating on the bare ground
plane. `workstation_env_cfg.py` spawns the gaussians once at `/World/Splats`, reasoning that
it is "a backdrop that only env 0 is standing in anyway" — but **Isaac Lab's grid cloner
centres the env grid on the world origin**, so env 0 coincides with it only when
`num_envs == 1`. MEASURED at 8 envs: env 0 sits at `[2.5, -2.5, 0]`, i.e. the desk was **3.5 m
away** from the env being filmed.

Every render produced before this used `render_workstation.py`, which hardcodes `num_envs=1`.
**The bug was structurally invisible to the only tool that had ever looked for it**, and would
have surfaced later as a vision-distillation student trained on photoreal pixels with no
background at all. The recorder now translates `/World/Splats` onto the filmed env's origin
(appearance-only; nothing in the MDP reads it).

Two smaller ones fixed alongside: `isaacsim.core.utils` does not exist in this build (use
`omni.usd.get_context().get_stage()`), and `AddTranslateOp` on a prim that already carries a
transform stack **appends a second op that composes** rather than replacing — so re-use an
existing translate op if there is one, or the desk lands at twice the offset.

### OPEN: the objects render washed out, and it is neither the mesh nor exposure

| | saturation |
|---|---|
| cube mesh vertex colours | median **172**/255, p90 224, **75 %** of verts vivid (S>110) |
| rendered pixels at wrist range | median **67**/255, **6.2 %** vivid |

A 2.6x collapse. Two candidate explanations were tested and both are **excluded**:

* **Not overexposure.** Rendered value mean 185/255 with **0.0 % of pixels clipped**. Blown
  highlights would show as clipping; there is none.
* **Not the mesh.** The authored `displayColor` is vivid, and every reconstruction averaging
  step (MVS, Poisson, decimation, hole-fill) leaves it at median 172.
* **Probably not a double-gamma error either.** The obvious candidate — `displayColor`
  authored sRGB, `UsdPreviewSurface` treating it as linear, renderer converting again — was
  tested by pre-linearising the colours (which lifts mesh saturation 172 -> 232). Rendered
  median moved only **69 -> 71**, with vivid pixels 2.6 % -> 11.3 %. A real double-gamma error
  would have recovered most of the 2.6x. **Reverted; not shipped.**

Remaining suspect is the material path itself: RTX may not honour a
`UsdPrimvarReader_float3` -> `UsdPreviewSurface.diffuseColor` connection at full strength and
may be falling back to a partial displayColor contribution. Next step is to test an OmniPBR /
MDL material, or bake the vertex colours to a texture, and re-measure with the same
mesh-vs-rendered saturation comparison — which is the measurement that localised this and
should be the acceptance test for any fix.

**Why this matters beyond looks:** a Rubix cube that renders pastel in sim will not match a
vivid one in reality. This is exactly the appearance gap that breaks vision transfer, and it
is invisible to every geometric gate in this pipeline.
