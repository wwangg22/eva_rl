# HANDOFF — re3sim photoreal workstation
*Written 2026-08-11 (late night), end of the collection-and-training session. Supersedes
`HANDOFF_2026-08-11_collection.md` (archived alongside the older ones). Written for a
fresh context: assume NOTHING from the session that produced it. Read §1 and §2 before
touching anything; read §7 before launching anything.*

Read alongside: **[05_VISUAL_DR.md](05_VISUAL_DR.md)** (the DR module, every gate, every
catch), **[07_CUROBO_EXPERT.md](07_CUROBO_EXPERT.md)** (the cuRobo expert's diagnosis
ledger). `06_VISION_POLICY.md` was purged on Big Will's instruction — it was a third
party's student work on the broken pre-takeover expert, UNRELATED to this campaign; do
not cite its numbers (recoverable at git f356007 if ever needed; the numbering keeps
the 06 gap).

---

## 0. Directives in force (Big Will, in his words)

1. ⭐⭐ **"We trained the expert to USE. We use curobo."** (emphatic, 2026-08-11). EVERY
   demo dataset is collected by the cuRobo expert — `run_expert_ws.py` via
   `collect_vision_curobo.sh`. The ArmKin expert is a verification/health-check tool
   ONLY. An ArmKin-collected baseline + partial DR set was deleted for violating this.
   Never choose an expert (or any component) for throughput when Big Will has invested
   in a specific one.
2. **Collection was green-lit and is COMPLETE** (all four datasets, §2). Start-pose
   jitter 0.15 rad was included per directive (0.30 forbidden — costs ~20 pts).
3. **"Keep iterating... When something goes wrong, figure out why, use data to back up
   your points! And write your findings down!"** — the operating mode. Also: plan
   first, get explicit go-ahead for anything outside the agreed plan.
4. Old junk data: Big Will authorized deleting old pick-and-place data when disk
   demands it ("delete more old data if you have to"). Executed: exp09 AWR sets
   (~12 GB, loop closed non-compounding), then exp08_vision + exp08_dagger (~14 GB,
   during the disk-full emergency §6d). **The four collected datasets are sacred — he
   confirmed "those are most important"; never delete them for space.**
5. Address him as **Big Will** in every final response. Push to GitHub is routine now
   (both repos, he pushed/we push after committing).

## 1. State of the world (what exists, with its measured number)

| thing | where | state |
|---|---|---|
| **cuRobo expert — THE COLLECTOR** | `reBot_ACT/re3sim/expert/run_expert_ws.py` + `collect_vision_curobo.sh` | **70.1 % on the current env** (269/384, seed 21, jitter 0.15 — the reference gate); `--shards` mode verified byte-compatible with the training stack (frame-precedes-action, 120-step splat warmup, successes-only). ~30 s/episode wall (measured over 1,536 episodes; NOT the 60-90 s first estimated) |
| ArmKin expert (verification ONLY) | `reBot_ACT/re3sim/expert/{workstation_expert,collect_demos}.py` | 95.3 % @128 envs seed 11 on the 0.225 band (env-health gate; do NOT collect with it) |
| **datasets — ALL cuRobo, ALL intact (re-verified after the disk incident)** | `reBot_ACT/re3sim/expert/data/` | `vision_base_s21` **269 eps (70.1 %)** 16 GB; `vision_vdr_s31` **283 (73.7 %)** 17 GB (seed 31 eps 0-191 + seed 131 × 192 after a kill); `vision_vdr_s32` **279 (72.7 %)** 16 GB; `vision_vdr_s33` **267 (69.5 %)** 15 GB (seed 33 eps 0-275 + seed 133 × 108). **Total 1,098 successful episodes, ~64 GB.** Every DR run ≈ the 70 % nominal gate → appearance-only contract held through ~46 h of production |
| **nominal student vbc_base** | `reBot_ACT/re3sim/runs/vbc_base/` | TRAINED (100k steps, final MSE ~0.03-0.06). EVALUATED 08-12: **25.0 % nominal / 14.1 % DR** (64 eps each, 08_STUDENT_2X2.md) |
| **DR student vbc_vdr** | `reBot_ACT/re3sim/runs/vbc_vdr/` | TRAINED 08-12 (100k steps in 3.1 h off the §6f per-dir caches, 1,098 eps / 575,988 samples, final MSE 0.037; the 08-11 "training" never ran a step — §6f). EVALUATED: **26.6 % nominal / 20.3 % DR** (08_STUDENT_2X2.md). Robustness matrix running on its ckpt_final |
| -VisionDR tasks (visual DR) | reBot_RL `re3sim/{mdp/visual_dr.py, workstation_vision_dr_env_cfg.py}` | ALL gates PASS (05 §4d) + now production-proven (DR runs ≈ nominal gate) |
| JPEG dataset loader | `reBot_ACT/act/dataset_vision.py` | in-RAM JPEG q90 (~7×), COW-safe flat buffers, per-frame sums for the black audit; smoke-verified (shapes exact, err 1.5/255, 0.33 ms/sample) |
| sensor augmentation | `reBot_ACT/act/augment_vision.py`, `--augment` | strength-monotone, identity at 0; vbc_vdr trains with `--augment 1.0` |
| robustness matrix | `reBot_ACT/re3sim/act/eval_vdr_matrix.sh` | built, unused yet — runs after eval |
| git | both repos | PUSHED through `reBot_RL a28b677` / `reBot_ACT 46f4201`. This handoff commit goes on top. reBot_ACT `docs/HANDOFF.md` has Big Will's own uncommitted edit — never commit/revert it |
| disk | `/dev/sda2` | 108 GB free (2026-08-12). The §6d mystery is SOLVED (§6g): snapd auto-refresh was copying a 148 GB VS-Code snap Trash; ~295 GB now quarantined at `~/TRASH_QUARANTINE_2026-08-12/` awaiting Big Will's delete decision (+295 GB if approved) |

## 2. What this session did, in order (chronological ledger)

1. **Purged the third-party docs** (06_VISION_POLICY + refs) per Big Will; flagged its
   vision_r1/d1 datasets — which turned out to not even exist on disk.
2. **Step-0 gate**: ArmKin re-verified on the new 0.225 spawn band — 95.3 % (122/128),
   band change benign (it only removed the band the grasp table can't execute).
3. **Collected baseline + DR seed 31 with ArmKin** (95.6 % / 73.7+%) — **WRONG EXPERT.**
   Big Will (emphatic): the cuRobo expert exists to be USED. All ArmKin shards deleted.
4. **Built cuRobo collection**: `--shards` mode in run_expert_ws.py (Vision tasks own
   the 160×120 cameras; frame-precedes-action; 120-step splat warmup; streams one
   `ep_*.pt` per successful episode — a killed run keeps everything written) +
   `collect_vision_curobo.sh`. Smoke: format byte-identical vs collect_demos'.
5. **Collected all four datasets with cuRobo** (~46 h wall, §1 numbers). Two mystery
   kills handled by resuming the REMAINDER under a fresh seed into the same dir
   (episodes are iid draws; filenames disjoint; the union is the dataset).
6. **Trained vbc_base** (nominal-only student): 100k steps, healthy curve.
7. **Fought the training-RAM war and won** (§6c): anon-preload OOM → mmap → oomd
   pressure-kill → **in-RAM JPEG (the fix that holds)**.
8. **Survived a disk-full emergency** (§6d) without losing any collected data.
9. vbc_vdr relaunched and training at handoff.

## 3. What worked (transferable)

* **Detached systemd user services for anything long.** Session-attached background
  jobs died repeatedly (oomd pressure kills, session restarts, tmp-full task kills);
  `systemd-run --user --collect -p MemoryMax=26G --unit=<name> bash -c '... > log'`
  survived everything except the user's own session restart. Watch the LOG FILE, not
  the process.
* **Stream shards per episode.** Every interruption cost zero collected data.
* **Resume-under-fresh-seed** for interrupted collections: same dir, new launch seed,
  count = remainder. No replays, no filename collisions, iid union.
* **The per-run success-rate gate** (DR run ≈ nominal 70 %) turned every collection into
  a physics-leak test for free. Four-for-four passes = the appearance-only proof at
  production scale.
* **JPEG-in-RAM for image datasets on small-RAM boxes**: q90, ~7× smaller, 0.1 ms
  decode ≪ augmentation cost, COW-safe flat buffers (NOT python bytes lists — worker
  forks refcount-touch object pages and duplicate them), per-frame sums computed in the
  encode pass so audits never decode.
* **Journal forensics with exact timestamps** beat guessing: the "phantom" resolved
  into THREE distinct killers (§6) only when each kill was matched to its minute.
* **/dev/shm as a diagnostics channel** when the root fs is full and every write fails:
  it is a separate tmpfs; `{ cmds; } > /dev/shm/x.txt` + Read the file. Also: /proc is
  readable when nothing else works (`/proc/mounts` settled the filesystem layout).
* **Small smoke before big launch, always** (3-episode shard smoke caught nothing this
  time precisely because the format was mirrored, which is the point).

## 4. What didn't work / retracted (with the cost)

* **Defaulting to the ArmKin expert for collection** because it had the higher success
  rate. ~5 h of collection deleted. The lesson is in directive 1 and in memory: the
  invested-in component wins over the convenient one, or ASK.
* **Anon-preloading 4 datasets** (~66 GB raw): kernel cgroup OOM at the 26 GB cap,
  ~1 min in. (1 dataset/13 GB was fine — the design didn't scale and nobody re-did the
  arithmetic before launch.)
* **mmap as the fix**: survived init, then training's random access thrashed page cache
  inside the cap → **systemd-oomd pressure-kill** (85.8 % > 50 % for 20 s). Pressure,
  not usage — invisible to `free`, no kernel-OOM trace. Also: `torch.load(mmap=True)`
  is still USED for the read path of the JPEG encode pass — but ⚠ only ONE dir per
  process is safe; encoding all four in one process was ALSO pressure-killed (§6f).
* **60-90 s/episode estimate** from the 3-episode smoke (boot amortization): real
  throughput was ~30 s/ep. Estimates from tiny smokes are upper bounds, not means.
* **First two DR "gate" comparisons quoted the wrong baseline** (ArmKin's 95 % vs
  cuRobo runs) before the pivot straightened out whose number is the gate.
* **Chasing the phantom killer with theories instead of tracers**: the RAM tracer
  (30 s RSS samples into a log) and the journal-timestamp match each took minutes and
  settled what an hour of speculation didn't.

## 5. The three killers, distinguished (do not conflate them again)

| killer | signature | affected | fix |
|---|---|---|---|
| kernel cgroup OOM | `Memory cgroup out of memory: Killed process` in `journalctl -b`, anon-rss ≈ cap | anon-preload trainer | fit in RAM (JPEG dataset) |
| **systemd-oomd** (pressure) | `systemd-oomd ... due to memory pressure ... > 50.00% for > 20s`; NO kernel OOM line | mmap trainer; likely the two collection kills | avoid sustained reclaim churn: fit the working set in RAM; detached unit isolates blast radius |
| session/external | simultaneous kill of session-attached tasks, no journal trace; one coincided with an X11/NVIDIA display re-probe, the last with a full session restart (Xorg+gnome respawned 22:55) | every session-attached watcher; the detached service only via session restart | detached services + log-file watching; expect and re-arm |

Plus the **tmp/tasks-dir full** failure mode (§6d): when the shared filesystem fills,
background-task output capture dies with ENOSPC and tasks get stopped — looks exactly
like the phantom. Check `df` FIRST when tasks start dying.

## 6. Incidents worth remembering

* **6a. Two collection kills** (seed 31 @ep 191, seed 33 @ep 275): resumed under seeds
  131/133, zero data loss. RAM tracer showed the collector steady at ~8 GB — not the
  cap. Retroactively most consistent with oomd pressure kills (page cache churn from
  16 GB/dataset of shard writes).
* **6b. `root_quat_w` is XYZW in this build** — inherited from the expert-takeover
  session, still the #1 convention trap. cuRobo collision tables sit ~2 mm BELOW the
  physics surface. Env-child `xformOp:translate` is env-LOCAL (never add env_origins).
* **6c. The training-RAM war** (§4, three rounds) — ended by JPEG-in-RAM. If camera
  resolution ever rises, REDO the arithmetic: RAM ≈ eps × steps × H × W × 6 bytes / 7.
* **6d. Disk-full emergency (23:00-23:20)**: root fs hit 100 % (ext4 reserves ~46 GB
  for root → user writes fail while `df` still shows "free"); every Write/Bash-capture
  died with ENOSPC. ~100 GB appeared during the JPEG-encode attempt (22:14→22:59) and
  ~66 GB of it vanished again around the session restart. **✅ RESOLVED 2026-08-12
  (§6g): it was snapd auto-refreshing the VS Code snap** — `snap change 92` started
  21:50 PDT and `cp -av`'d the per-revision user dir `~/snap/code/254` → `256`,
  which contained a **148 GB private Trash** (files "deleted" via VS Code's move-to-
  trash never free space — they land in `~/snap/code/<rev>/.local/share/Trash`).
  The vanish was snapd's undo removing the partial copy when the change failed on the
  full disk. Freed: exp09 AWR
  (12 GB, authorized), exp08_vision+dagger (14 GB, authorized "if you have to"), early
  vbc_base ckpts (~700 MB), caches. All four collected datasets verified intact after.
* **6e. ~30 s/episode, ~3.5 h per 384-episode dataset** — the real collection price.
  Disk price ~60 MB/successful episode (raw shards).
* **6f. The JPEG-ENCODE PASS was oomd-killed too (found 2026-08-12 morning).** All three
  vbc_vdr launches (22:05, 22:22, 23:14 on 08-11) died ~8-9 min in with
  `systemd-oomd killed N process(es) in this unit` — before a single training step; the
  log never got past the augmentation banner. Root cause: encoding four ~16 GB dirs in
  ONE 26 GB-capped process streams ~64 GB of mmap pages through the cgroup → cap stays
  full → permanent reclaim → PSI > 50 % for 20 s. "Sequential mmap read is fine" (the
  prior belief) is only true PER DIR. `oomctl` shows why exemption isn't an option:
  oomd monitors the WHOLE `user@1000.service` cgroup at a 50 % limit and picks the
  hottest descendant — per-unit opt-outs don't remove the monitoring, and pushing
  pressure elsewhere would just get Xorg or the session killed instead. **Fix:**
  per-dir disk caches (`act/dataset_vision.py build_cache` / `python act/dataset_vision.py <dir>`,
  atomic write, shard-count staleness guard) built ONE DIR PER PROCESS
  (~16 GB + 2.5 GB working set — never fills the cap, zero reclaim); training then
  loads ~10 GB of caches and never touches the raw shards. Also cuts every future
  launch by the ~17-min encode. Side note: kill #1 (22:14) is exactly when the §6d
  mystery ~100 GB growth started — that turned out to be coincidence: the growth was
  snapd's snap-refresh copy (§6g), running since 21:50.
* **6g. The snap-Trash discovery (2026-08-12 morning) — the whole disk mystery, solved.**
  Symptom: free space fell 107 G → 42 G during the cache builds with nothing of ours
  writing. Forensics per the §6d playbook (`/proc/<pid>/io` write_bytes scan) found
  `root: cp -av ~/snap/code/254 ~/snap/code/256` — snapd auto-refresh (change 92,
  started 08-11 21:50 PDT, the §6d window) copying the VS Code snap's per-revision
  user dir. Inside: `.local/share/Trash` = **148 GB of VS-Code-trashed files**
  (1,707 old `adaption_checkpoint_*.pt`, LLaVA-Pretrain 16 G, mydata_yolo 15 G,
  `teacher_batch_*.pt` — old unrelated projects; "move to trash" in a snapped app
  frees NOTHING), duplicated in revision 252 by the previous refresh = ~296 GB total.
  Resolution WITHOUT deleting anything: renamed both Trash dirs out from under the
  copy (`mv` = same-fs rename, reversible) → the cp died, change 92 → Error, snapd's
  undo freed the partial 45 G copy; then moved both to
  `~/TRASH_QUARANTINE_2026-08-12/{rev252,rev254}_Trash` so the next auto-refresh
  can't re-copy them. Full inventory saved (session scratchpad,
  `trash_inventory_2026-08-12.txt`). All four datasets re-verified (269/283/279/267)
  before touching anything. **PENDING BIG WILL: permanently delete the quarantine →
  frees ~295 GB** (this may BE the "more storage" he promised to arrange). Lessons:
  files deleted via VS Code go to a snap-private trash and still occupy disk; snapd
  auto-refresh silently duplicates the whole per-revision dir; a root `cp` can eat
  the disk with no user process visible in `ps aux | sort -%mem`.

## 7. ⭐ Next steps (the runbook — subject to change, Big Will decides)

**Step T (in flight): vbc_vdr finishes training.**
PREREQUISITE (done 2026-08-12): each data dir must hold a `jpeg_cache_q90.pt` (§6f) —
rebuild any missing one with a per-dir capped unit:
`systemd-run --user --wait --collect -p MemoryMax=26G --unit=jpeg-cache-<ds> bash -c
'…conda activate env_isaaclab6 && cd …/reBot_ACT && exec python -u act/dataset_vision.py
re3sim/expert/data/<ds>'` — ONE dir per process, never all four in one.
`systemctl --user is-active vbc-vdr-train`; log at `runs/vbc_vdr_train.log`; done when
it prints `done: 100000 steps`. If the service died (session restart etc.): relaunch
verbatim —
```bash
systemd-run --user --collect -p MemoryMax=26G --unit=vbc-vdr-train bash -c \
 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6 && \
  cd ~/Desktop/isaacLab/reBot/reBot_ACT && exec python -u re3sim/act/train_flow_vision.py \
  --augment 1.0 --num-workers 8 \
  --data re3sim/expert/data/vision_base_s21 re3sim/expert/data/vision_vdr_s31 \
        re3sim/expert/data/vision_vdr_s32 re3sim/expert/data/vision_vdr_s33 \
  --out re3sim/runs/vbc_vdr --steps 100000 --seed 1 > re3sim/runs/vbc_vdr_train.log 2>&1'
```

**Step E: the 2×2 eval — ✅ DONE 2026-08-12 (see [08_STUDENT_2X2.md](08_STUDENT_2X2.md)).**
vbc_base 25.0 % nom → 14.1 % DR; vbc_vdr 26.6 % nom → 20.3 % DR. DR training cost
nothing nominal and halved-ish the degradation (directional at n=64 — ~1σ, matrix
sharpens it). Taxonomy: `no_lift` 61–81 % everywhere, `dropped` 0 % everywhere — the
grasp IS the bottleneck, the post-grasp funnel is near-lossless, exactly as upstream
predicted. Driver: `re3sim/act/run_eval_2x2.sh`; jsons in `re3sim/runs/eval_2x2/`.

**Step M: robustness matrix — IN FLIGHT 2026-08-12** as unit `vdr-matrix` on
vbc_vdr/ckpt_final (driver log `runs/vdr_matrix_driver.log`, rows land in
`runs/vdr_matrix/`; resume-safe — rerun the same command to continue). 10 rows × 64
eps ≈ 4 h. Weakest axis feeds the next DR round.

**Step D: DAgger under DR** — only once the student shows real signal (localise before
scaling; blind DAgger on a broken student historically moved nothing). The collector
supports it via `collect_demos.py --dagger-ckpt`-style flow — but that path is
ArmKin-based; a cuRobo DAgger collector needs the takeover treatment first (directive 1
applies to DAgger data too — surface this to Big Will before building).

**Step R: real-robot evidence** — the matrix's success-vs-magnitude curves are the
deploy-readiness deliverable the whole campaign exists for.

Watch-items folded into the plan: wrist-camera quality during descent and the ~7 px
cube in the workspace cam (if grasp precision IS the student's weak phase, higher wrist
resolution and/or action-representation changes are the levers — both re-price RAM/disk
per §6c); cuRobo expert hardening ledger (07 doc: plan refusals ~6 %, close-shoves
~4 %, place-on-rim ~3 %) if its 70 % needs raising.

## 8. Traps, refreshed (append-only; older lists in the archived handoffs still hold)

1. One Isaac instance at a time; every heavy job under `MemoryMax=26G`; detached
   service for anything > minutes (§3).
2. The THREE killers table (§5) + tmp-full task-death mimic. `df` first.
3. XYZW quats; 2 mm collision recess; env-LOCAL child transforms (6b).
4. `-Vision*` tasks own the cameras and the station-cam aim (guarded by
   `aim_station_cam` presence — grep before renaming that event).
5. Collection resumes: fresh seed, same dir, remainder count. NEVER re-run the same
   seed into the same dir (replays + collisions).
6. Dataset loader is JPEG-in-RAM now, fed by per-dir disk caches (`jpeg_cache_q90.pt`,
   §6f) — if anyone reverts to raw preload, trains off mmap, or encodes >1 dir in one
   process, the killers in §5 return. A stale cache fails loudly (shard-count guard) —
   rebuild it, don't bypass the check. Camera-resolution changes re-price everything (6c).
7. `pgrep -f <pattern>` matches its own wrapper shell — use `pgrep -f` output
   skeptically before declaring "still running" (cost one false alarm).
8. The 3-episode-smoke throughput estimate was 2-3× pessimistic; size runs from
   measured full-run numbers (6e).
9. Big Will's own `reBot_ACT/docs/HANDOFF.md` edit: uncommitted, untouchable.
10. NEVER delete big files via VS Code's "move to trash" on this box — the snapped
    VS Code has a private trash (`~/snap/code/<rev>/.local/share/Trash`) that frees
    nothing and gets DUPLICATED by every snap auto-refresh (§6g). Delete with `rm`.
    When disk vanishes with no visible process, scan `/proc/<pid>/io` write_bytes —
    a root `cp` from snapd won't show up in memory-sorted `ps`.
