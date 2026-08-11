# 06 — The cuRobo expert on the workstation (the pick-and-place expert, ported)

*2026-08-11. Big Will's directive: "implement an expert that is exactly the same as the
pick and place expert but for the workstation task", taking over the in-flight upstream
attempt (reBot_RL cb7148b prepared the ground: spawn band vs the grasp table, URDF FK
cross-check, `table_candidates` 200/200 / `plan_grasp` 10/10 feasibility spikes).*

**Code:** `reBot_ACT/re3sim/expert/run_expert_ws.py` (+ `probe_ws_plan.py`, the pure-cuRobo
failure-isolation probe). The planning machinery is IMPORTED from the pick-and-place
expert — `spike_plan_grasp.table_candidates`, the shared 12 953-row grasp table, the
carry-config fallback rungs — not copied. Task-specific: one grasp target (the 56 mm
cube, face-axis aligned goalsets with an unaligned fallback), the yawed 93 mm-wall box as
the receptacle (collision-modelled live), `mdp.placed_mask` as truth, place heights
bracketing the verified ArmKin `CARRY_Z`.

## Measured state

| run | result |
|---|---|
| first smoke (2 eps) | 0 % — every plan "Goalset planning returned None" (the upstream symptom) |
| + recessed planner table | 50 % (2/4) |
| + align fallback + instrumentation (12 eps, s302) | 83.3 % |
| + quaternion-parity fix (same 12 eps) | **91.7 %** |
| 64 eps, s303 | 76.6 % — two failure buckets isolated (below) |
| + spawn floor 0.225 + XY misexec gate (64 eps) | **73.4 %** (z-gate variant 62.5 %, retracted) |

Current headline: **~75 % over 128 episodes** (73.4/76.6 on two 64-ep runs; n=64 CI ≈ ±5.5 pts).
Videos: `re3sim/expert/expert_ws_ep{3,4,5}_ok_{station,wrist}.mp4` (+ ep0 failure).

## The three root causes, in the order they fell

1. **A flush desk slab kills every plan.** The collision table's top face at exactly z = 0
   collides with the grasp poses' finger spheres — "Goalset planning returned None", 6/6.
   `probe_ws_plan.py` isolated it in one sweep (30/30 rows: flush FAIL, 2 mm-recessed OK,
   cube/clutter/box obstacles all innocent). The pick-place spike had already learned this
   (its table sits at z = −0.027); the port re-learned it. **This was almost certainly the
   upstream blocker.**
2. **`root_quat_w` is XYZW in this build, not WXYZ.** Unpacked wxyz, every cube read
   yaw = π exactly, so the face-alignment axis was one constant wrong direction. The
   instrumented 12-ep run showed every air-close at "yaw=3.14" and every success at a real
   yaw; the one-line fix took the same seed 83.3 % → 91.7 %.
3. **Candidate existence ≠ executability at the grasp-table's inner edge.** All 7 whole-
   episode grasp failures of the 64-ep run sat at r < 0.225 — below the table's 0.221
   floor, where the 0.035 m matching tolerance still *returns* rows but their lateral
   shift mis-executes (the table's own "never translate laterally" rule, paid at close
   time). Fixed where it belongs: `reset_objects` target annulus 0.20 → **0.225**.

Plus the og's pre-close mis-execution gate, restored XY-only: FK pocket vs cube > 22 mm →
abort + exclude the row. (A z-gate on the FK pocket was tried and RETRACTED — it read
0.07–0.09 on healthy rows and cost 14 points; the og's z signal was finger BODY heights,
a different measurement.)

## Remaining failure ledger (~25 %, next levers)

* **plan-refusals at valid radii** (~3-4 eps/64, "Goalset planning returned None" ×3):
  obstacle-blocked configurations — box or clutter adjacent to the cube. Lever: obstacle-
  aware candidate ordering, or a clutter-clearance re-draw like the ArmKin expert's.
* **close shoves** (~2-3 eps/64): the close pushes the cube out (disp 5–13 cm) and retries
  chase it. Lever: the og's close-displacement retry logic prices this; a softer descent
  settle may too.
* **place-on-rim** (~1-2 eps/64): released cube settles on the box wall or just outside
  (`placed=False` with delivery "succeeding"). Lever: drop height / release-centering.
* one-off: `attach_from_scene` sphere-fit overflow (9 > 8 slots) degrades one transport.

## How to run

```bash
conda activate env_isaaclab6 && cd ~/Desktop/isaacLab/reBot/reBot_ACT
python -u re3sim/expert/run_expert_ws.py --episodes 64 --seed 303      # evaluate
python -u re3sim/expert/run_expert_ws.py --episodes 6 --video-all      # film for review
python -u re3sim/expert/run_expert_ws.py --episodes 32 --record-h5 ws_demos.h5  # demos
python -u re3sim/expert/probe_ws_plan.py                               # planner probe
```

Same CLI, labelling scheme (`segments`/`outcomes`/`train_mask`), perturb and diversify
modes as `expert/run_expert_v1.py`. Note the **ArmKin expert**
(`re3sim/expert/{workstation_expert,collect_demos}.py`, 95.3 % at 128 envs) remains the
production demo collector; this cuRobo expert is its architectural twin from the
pick-and-place lineage, per the directive.
