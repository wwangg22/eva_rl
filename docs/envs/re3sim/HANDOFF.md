# HANDOFF — re3sim photoreal workstation
*Written 2026-08-11, end of the expert-takeover + visual-DR session. Supersedes
`HANDOFF_2026-08-09_directive.md` (kept, with the other archives). Written for a fresh
context: assume nothing from the session that produced it.*

Read alongside: **[05_VISUAL_DR.md](05_VISUAL_DR.md)** (the DR module, every gate, every
catch), **[07_CUROBO_EXPERT.md](07_CUROBO_EXPERT.md)** (the ported expert's diagnosis
ledger). `06_VISION_POLICY.md` was **purged 2026-08-11 on Big Will's instruction** — it
documented a third party's student rounds built on the broken pre-takeover expert and is
unrelated to this campaign (recoverable from git history at f356007 if ever needed).

---

## 0. Directives in force

1. **Big Will (2026-08-11): data collection is GREEN-LIT**, with start-pose randomisation
   included. Start-pose jitter is already the collection default —
   `collect_vision.sh` sets `RE3SIM_ARM_START_JITTER=0.15` unless overridden (the swept
   value: 96.1 % → 90.6 % on the ArmKin expert, measured pre-spawn-band-change; 0.30
   costs too much at 75.0 %). Nothing extra to wire; do NOT collect at 0.30.
2. Everything is committed locally and **ready to push, but NOT pushed** — that decision
   is Big Will's. reBot_RL `master` is 3 ahead (DR module; spawn floor + expert doc;
   this handoff), reBot_ACT `main` is 4 ahead (paths; DR wiring; the cuRobo expert).
   `reBot_ACT/docs/HANDOFF.md` has Big Will's OWN uncommitted edit — never commit or
   revert it.

## 1. State of the world (what exists and its measured number)

| thing | where | state |
|---|---|---|
| **ArmKin expert** (production demo collector) | `reBot_ACT/re3sim/expert/{workstation_expert,collect_demos}.py` | **95.3 % (122/128) @128 envs seed 11 on the NEW 0.225 band** (§4 step 0 PASSED 2026-08-11; was 96.9 % on the old band — band change benign for this expert) |
| **cuRobo expert** (the run_expert_v1 port, this session) | `reBot_ACT/re3sim/expert/run_expert_ws.py` | ~75 % over 128 eps (76.6/73.4 on two 64-ep runs); ledger + remaining levers in 07_CUROBO_EXPERT.md; videos `expert_ws_ep{3,4,5}_ok_*.mp4` delivered to Big Will |
| **-VisionDR tasks** (visual domain randomisation) | `reBot_RL .../re3sim/{mdp/visual_dr.py, workstation_vision_dr_env_cfg.py}`, ids `Rebot-Workstation-PickPlace1-VisionDR{,-Play}-v0` | ALL gates PASS (§2b); appearance-only proven bit-exact |
| **sensor-level augmentation** | `reBot_ACT/act/augment_vision.py`, `--augment` on `train_flow_vision.py` | strength-monotone, identity at 0, 10 ms/img |
| **robustness matrix** | `reBot_ACT/re3sim/act/eval_vdr_matrix.sh` | per-axis rows on -VisionDR-Play, ckpt-keyed resume |
| env spawn band (grasp target) | `mdp/events.py reset_objects` | cube annulus **0.225–0.28 m** (this session; was 0.20, orig 0.15). ⚠ obsoletes every pre-change success number until re-measured |
| datasets on disk | `reBot_ACT/re3sim/expert/data/` + `data/` | **vision_base_s21: 367 eps @ 95.6 %, 33 GB** (2026-08-11, jitter 0.15); DR seeds collecting. (vision_r1/vision_d1 from the purged effort are NOT on disk — stale references only.) Old pick-place: `data/exp08_{vision,dagger}` KEPT (the ~80 % student's data + the EXP09 perception-head shards); `data/exp09_awr{,_eval}` DELETED 2026-08-11 per Big Will (AWR closed non-compounding; ~12 GB freed; it1 champion head kept in `runs/exp09/`) |

## 2. What this session did, in order

### 2a. Expert takeover (Big Will handed it over mid-session)

Upstream was building the cuRobo pick-and-place expert on the workstation and hit a wall.
Ported `run_expert_v1.py` → `run_expert_ws.py` (machinery imported, not copied) and found
three root causes — each with the measurement that proved it (details: 07 doc):

1. **Flush planner desk** (collision slab top at z = 0.0) fails EVERY `plan_grasp` with
   "Goalset planning returned None" — almost certainly upstream's blocker. The pure-cuRobo
   `probe_ws_plan.py` isolated it in one 30-row sweep; 2 mm recess (the pick-place
   spike's own convention) fixes all of it.
2. **`root_quat_w` is XYZW in this Isaac Lab build, not wxyz.** Read wxyz, every cube
   yaw = π exactly → the grasp-alignment axis was one constant wrong direction. One line;
   83.3 % → 91.7 % on the same seed. (The env's own `reset_objects` and the og expert's
   `can_axis` both already encode xyzw — check there before trusting any quat unpack.)
3. **Grasp-table candidate existence ≠ executability below r ≈ 0.225** (the table's floor
   is 0.221 and it refuses lateral shifts; the 0.035 matching tolerance still *returns*
   rows down to ~0.19 which then air-close). All 7 inner-band episodes of a 64-ep run
   failed. Fixed env-side: spawn floor 0.225.

Also: og's pre-close mis-execution gate restored **XY-only** (executed FK pocket vs cube
\> 22 mm → abort + exclude row). A z-gate on the FK pocket was tried and **retracted**
(−14 pts): the og's z signal was finger BODY heights from the sim, not FK — different
measurement, and the FK-pocket z reads 0.07–0.09 on perfectly healthy rows.

### 2b. Visual DR — built, and triple-confirmed before any collection

The design in one line: the demo experts are state-based, so visual DR is **free** —
camera pose/roll/focal jitter (+ rare bumped-rig), wrist-mount jitter, per-env key/fill
lights + dome shifts, desk-swap (splats ↔ tinted slab), floor cards + ground tint +
backdrop + visual-only distractors, all per-env per-reset, env-var knobs (`RE3SIM_VDR_*`,
master `RE3SIM_VDR`, sweep knob `RE3SIM_VDR_SCALE`).

Gates, all scripted and re-runnable (05 doc §4d):
* **dynamics bit-exact** vs -Vision-v0 (200-step scripted sweep, max |Δobs| = 0.0) — no
  collider leaks; re-run `probe_vdr_invariance.py` ×2 + `compare_vdr_traj.py` after ANY
  DR edit;
* camera path bit-exact vs `set_world_poses_from_view` at zero DR (`probe_vdr_camera.py`);
* DR draws deterministic in the run seed; varied across resets;
* 32-env end-to-end shard smoke: DR visibly in recorded frames
  (`renders/vdr_shard_frames.png`), `inspect_shards` leak-check clean, expert taxonomy
  unchanged, RAM peak measured.

An independent high-effort code review then found (and I fixed) among others: **the
env-origin double-offset** — child-prim `xformOp:translate` is env-LOCAL, adding the
origin shipped every per-env light/distractor into a NEIGHBOUR'S cell (grids looked
plausible because 2×origin lands on another cell of a symmetric grid); augmentation
severity/gates not scaling with strength; matrix env-var leaks and a stale-ckpt resume
hazard; `record_video.py`'s unguarded camera re-aim. Full list + fixes in 05 §4e.

### 2c. Machine limits (both learned the hard way — the freezes Big Will saw)

* ⭐ **ONE Isaac instance at a time.** Two Kit processes exhausted 31 GB RAM (2.3 GB free,
  0 swap → hard freeze/reboot).
* ⭐ **128-env shard collection alone does not fit either**: the collector buffers
  ~14.7 MB/step × 1018–1118 steps ≈ 16 GB/batch on top of Isaac's 12–14 GB. Hence:
  **64 envs × 6 batches**, and every collection runs under
  `systemd-run --user --scope -p MemoryMax=26G` (worst case a killed process, never a
  frozen box). Measured: 32 envs peak 16.9 GB total system.
* Disk: shards are **~90 MB/episode** (measured again 2026-08-11: baseline 367 eps =
  33 GB). After the AWR deletion, 120 GB free vs ~99 GB for the 3 DR seeds — fits with
  ~20 GB headroom. Drop DR seed 33 first if that erodes; every slice regenerates
  exactly from its seed.
* Don't pipe long background runs through `tail`/`grep` — output buffers until exit and
  you fly blind. Redirect raw.

## 3. What worked / what didn't (the transferable lessons)

**Worked:**
* *Render-and-look beats every automated gate for appearance code.* Five real DR bugs
  (grid-floor cue, floating objects on desk-hide, distractor overhang, blown-white
  backdrops, palette) were caught ONLY by eyeballing stills grids. Nothing in the MDP
  reads pixels; no assert can catch these.
* *Bit-exactness probes as standing gates.* The zero-DR camera probe and the dynamics
  invariance pair caught nothing today precisely because they were run after every edit —
  they are the reason the DR module can be trusted at all. Convention-trap country
  (wxyz/xyzw, opengl/ros/world) demands empirical equality, not docstring faith.
* *Instrument before iterating.* The pocket-vs-cube print solved the yaw-parity bug in
  one read. Cheap targeted probes beat blind retraining loops.
* *Small-N A/B on the same seed* (12 eps) to validate a fix, THEN 64-ep honest numbers.
  But n=64 has ±5.5 pt CI — do not read 73.4 vs 76.6 as a difference.
* *Import, don't copy* (grasp table, table_candidates, carry rungs, dataset loaders) —
  every ported constant that was re-derived instead (table z, TCP conventions) was a bug
  source; everything imported stayed correct.

**Didn't work / retracted:**
* The FK-pocket **z-gate** (−14 pts, retracted; wrong signal).
* **6 m floor cards** (neighbour overlap + z-stagger = wrong card visible, mid-episode
  recolours) → 1.9 m cards + global ground tint.
* **Hiding the splat desk** alone (objects float on an invisible collider) → swap to a
  tinted slab instead: "a different table", not a hole in the world.
* My first two smoke-run camera checks were on a **stale snapshot of the code** (the
  review agent's findings were against pre-fix files twice) — always re-verify a finding
  against the current tree before acting on it.
* Two background Isaac jobs in parallel (froze the machine, twice).

## 4. ⭐ The collection runbook (GREEN-LIT — execute top to bottom)

```bash
conda activate env_isaaclab6 && cd ~/Desktop/isaacLab/reBot/reBot_ACT

# 0. RE-VERIFY the ArmKin expert on the NEW spawn band (0.225 floor changed the task!)
#    Gate: >= 90 %. ✅ PASSED 2026-08-11: 95.3 % (122/128), taxonomy clean
#    (0 plan-failed, 6 never-got-there, 0 lifted-but-lost). Deterministic; re-run
#    only if the env or expert code moves again. ~15 min.
systemd-run --user --scope -p MemoryMax=26G -- \
  python -u re3sim/expert/collect_demos.py --headless --num_envs 128 --batches 1 \
    --seed 11 --out /tmp/verify_band225_s11.hdf5
#    Also re-run the two DR gate probes if reBot_RL moved (05 §4d — cheap).

# 1. baseline (nominal appearance; start-pose jitter 0.15 is the script's default)
systemd-run --user --scope -p MemoryMax=26G -- \
  bash re3sim/expert/collect_vision.sh re3sim/expert/data/vision_base_s21 64 6 21

# 2. DR set — seeds 31/32/33 (baseline doubles as the ~25 % nominal mix fraction)
for SEED in 31 32 33; do
  TASK=Rebot-Workstation-PickPlace1-VisionDR-v0 \
  systemd-run --user --scope -p MemoryMax=26G -- \
    bash re3sim/expert/collect_vision.sh re3sim/expert/data/vision_vdr_s$SEED 64 6 $SEED
done
# gate per batch: expert success ~= step-0's number (visual DR is state-invisible;
# a drop means a physics leak — STOP and run the invariance probes)

# 3. train baseline + DR policy; eval BOTH on nominal and DR; then the matrix
python -u re3sim/act/train_flow_vision.py --data re3sim/expert/data/vision_base_s21 \
    --out re3sim/runs/vbc_base --steps 100000 --seed 1 --num-workers 8
python -u re3sim/act/train_flow_vision.py --augment 1.0 --num-workers 8 \
    --data re3sim/expert/data/vision_base_s21 re3sim/expert/data/vision_vdr_s31 \
           re3sim/expert/data/vision_vdr_s32 re3sim/expert/data/vision_vdr_s33 \
    --out re3sim/runs/vbc_vdr --steps 100000 --seed 1
bash re3sim/act/eval_vdr_matrix.sh re3sim/runs/vbc_vdr/ckpt_final.pt re3sim/runs/vdr_matrix
```

Serial, always — one Isaac job at a time (§2c). Everything is deterministic in its seed.

## 5. Future plan (subject to change — Big Will decides)

*(Per Big Will 2026-08-11: the third party's student results (1.6 %/0 %) were built on
the broken pre-takeover expert and are UNRELATED — do not treat them as evidence about
this pipeline. Our student is trained fresh, from this campaign's data only.)*

1. **Collect** (runbook above): step-0 gate → baseline s21 → DR seeds 31/32/33.
2. **Train the pair**: nominal-only student vs (nominal + 3×DR + `--augment 1.0`) student
   (§4 step 3). Eval BOTH on the nominal task and on -VisionDR-Play — the 2×2 is the
   first real evidence of what DR buys.
3. If the student underperforms, **localise before scaling** — handoff-style probes
   (take over mid-episode from a replayed expert prefix) and per-camera ablations are
   cheap; blind DAgger/data-scaling rounds are not. Watch specifically: wrist-camera
   image quality during the descent, cube pixel footprint in the workspace cam
   (~7 px at 160×120), and action representation (absolute joint targets vs delta) —
   candidate levers if grasp precision is the weak phase. A camera-resolution change
   re-prices the RAM/disk budgets in §2c (94 MB/ep scales with pixels; re-measure with
   one smoke batch first).
4. **DAgger under DR** once the base student shows signal, then re-eval.
5. Run the **robustness matrix** (`eval_vdr_matrix.sh`), feed the weakest axis back into
   the next DR round.
6. cuRobo-expert hardening if it is to become a collector (07 doc ledger: obstacle-blocked
   plan refusals ~6 %, close-shoves ~4 %, place-on-rim ~3 %); the ArmKin expert remains
   the production collector meanwhile.
7. Real-robot bring-up: the robustness matrix's per-axis success-vs-magnitude curves are
   the deploy-readiness evidence Big Will asked the whole campaign for.

## 6. Traps, refreshed (append-only; the old lists still hold)

1. All of HANDOFF_2026-08-09_directive.md §4 (planner population = env count; never
   sub-32; conda env_isaaclab6; render-product pixel ceilings; splats need per-env or
   placement; stepped pre-roll before any render; `env_.*` resolved prim paths...).
2. `root_quat_w` is **XYZW** here (§2a.2). `pose[:, 3:7]` buffers likewise.
3. cuRobo collision tables must sit ~2 mm BELOW the physics surface (§2a.1).
4. Env-child `xformOp:translate` is env-LOCAL — never add `env_origins` (§2b).
5. One Isaac instance; 64-env collection cap; MemoryMax wrapper; ~94 MB/episode (§2c).
6. The spawn-band change invalidated prior expert numbers — ArmKin re-measured 95.3 %
   on the new band (§4 step 0, 2026-08-11); the cuRobo expert's ~75 % is still an
   old-band number.
7. The doc numbering skips 06: that slot held the purged third-party vision-policy doc
   (see header note); the cuRobo expert doc is **07** and stays 07 (two earlier commit
   messages say 06 — follow the file).
8. `-VisionDR` + `collect_demos`/`record_video`: the env owns the station-cam pose
   (guarded by `aim_station_cam` presence). If that event is ever renamed, three guards
   fail OPEN to the fixed aim — grep "aim_station_cam" first.
