#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gate A smoke test for the vision pick-place env (EXP08 section 4).

Checks, on a moving robot (deterministic sinusoidal joint actions):

a. obs shapes/dtypes for both groups; student group passes the no-privileged-info audit
   (no term func from PRIVILEGED_OBS_FUNC_NAMES; proprio dims 8+8+7).
b. per-camera frame FRESHNESS over the whole run: consecutive-frame mean-abs-diff per
   camera must keep changing (the step-0 twin camera froze after ~35 steps).
c. wrist depth-min ~2-5 cm every frame (gripper housing => the render tracks the link).
d. FPS (env-steps/s) for collection budget planning.
e. same seed + same actions => the privileged 41-D trajectory must match the camera-free
   env (states, NOT pixels). Run this script once per task with --dump-states, then
   compare offline with --compare (no sim launch).

Usage (conda env_isaaclab6):
    # vision env, full camera checks + state dump
    python scripts/test_pick_place_vision_env.py --num-envs 4 --steps 120 \
        --dump-states /tmp/vision_states.pt
    # camera-free reference
    python scripts/test_pick_place_vision_env.py --task Rebot-PickPlace-Play-v1 \
        --num-envs 4 --steps 120 --dump-states /tmp/state_states.pt
    # offline comparison
    python scripts/test_pick_place_vision_env.py --compare /tmp/vision_states.pt /tmp/state_states.pt
"""

import argparse
import math
import sys
import time

# stale-frame threshold: consecutive-frame mean abs diff (uint8 units) below this counts
# as a frozen frame. A frozen buffer repeats the SAME tensor (diff ~0.0); a live but
# slow-changing view (distant workspace cam at 160x90) measured 0.44 min — so the
# discriminator is near-zero, not "small".
FRESH_THRESH = 0.05
WARMUP_STEPS = 20  # render warmup: blank/settling frames, excluded from freshness stats


def compare_dumps(path_a: str, path_b: str) -> int:
    import torch

    a, b = torch.load(path_a), torch.load(path_b)
    if a["obs"].shape != b["obs"].shape:
        print(f"FAIL: shape mismatch {tuple(a['obs'].shape)} vs {tuple(b['obs'].shape)}")
        return 1
    diff = (a["obs"] - b["obs"]).abs()
    print(f"policy-obs trajectories: shape {tuple(a['obs'].shape)}")
    print(f"max abs diff:  {diff.max().item():.3e}")
    print(f"mean abs diff: {diff.mean().item():.3e}")
    per_step = diff.amax(dim=tuple(range(1, diff.ndim)))
    first_bad = (per_step > 1e-5).nonzero()
    if len(first_bad):
        print(f"first step with diff > 1e-5: {first_bad[0].item()} of {len(per_step)}")
    verdict = "PASS" if diff.max().item() < 1e-4 else "FAIL"
    print(f"Gate A-e state match: {verdict}")
    return 0 if verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Rebot-PickPlace-Vision-Play-v1")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump-states", default=None, help="save the 41-D policy obs trajectory to this .pt")
    parser.add_argument("--aa-mode", default=None, choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"],
                        help="override sim.render.antialiasing_mode")
    parser.add_argument("--spp", type=int, default=None,
                        help="override sim.render.samples_per_pixel (direct-lighting spp; "
                             "per-frame, load-independent noise reduction)")
    parser.add_argument("--static", action="store_true",
                        help="zero actions (static scene): report per-pixel temporal noise stats "
                             "instead of failing freshness (frames SHOULD be near-identical)")
    parser.add_argument("--env-spacing", type=float, default=None,
                        help="override scene.env_spacing (match the vision env's 6.0 when dumping the "
                             "camera-free reference, so world-coordinate fp noise doesn't confound check e)")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two dumps and exit (no sim)")

    if "--compare" in sys.argv:
        args = parser.parse_args()
        return compare_dumps(*args.compare)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import reBot_RL.tasks  # noqa: F401  (registers the envs)
    from reBot_RL.tasks.manager_based.pick_place_vision.pick_place_vision_env_cfg import (
        PRIVILEGED_OBS_FUNC_NAMES,
        STUDENT_PROPRIO_DIM,
    )

    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if args.env_spacing is not None:
        env_cfg.scene.env_spacing = args.env_spacing
    if args.spp is not None:
        env_cfg.sim.render.samples_per_pixel = args.spp
    if args.aa_mode is not None:
        env_cfg.sim.render.antialiasing_mode = args.aa_mode

    has_cameras = hasattr(env_cfg.scene, "wrist_cam") and env_cfg.scene.wrist_cam is not None
    failures: list[str] = []

    # ---- check a (part 1): cfg-level no-privileged-info audit, before any sim ----
    if hasattr(env_cfg.observations, "student"):
        student_cfg = env_cfg.observations.student
        proprio_terms = []
        for name in student_cfg.__dict__:
            term = getattr(student_cfg, name)
            func = getattr(term, "func", None)
            if func is None:
                continue
            func_name = getattr(func, "__name__", str(func))
            if func_name in PRIVILEGED_OBS_FUNC_NAMES:
                failures.append(f"AUDIT: privileged term '{name}' (func {func_name}) in student group")
            if func_name not in ("image", "image_features"):
                proprio_terms.append((name, func_name))
        print(f"[audit] student terms: {[n for n, _ in proprio_terms]} + cameras; "
              f"no privileged funcs: {'PASS' if not failures else 'FAIL'}")
        # depth channel on the wrist for check c (diagnostic only, not a student obs)
        env_cfg.scene.wrist_cam.data_types = ["rgb", "distance_to_image_plane"]

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs

    obs_dict, _ = env.reset()

    # ---- check a (part 2): live shapes/dtypes ----
    obs_pol = obs_dict["policy"]
    assert obs_pol.shape == (n, 41), f"policy obs {tuple(obs_pol.shape)} != ({n}, 41)"
    print(f"[shapes] policy: {tuple(obs_pol.shape)} {obs_pol.dtype}")
    if "student" in obs_dict:
        stu = obs_dict["student"]
        pdim = sum(stu[k].shape[-1] for k in ("joint_pos", "joint_vel", "actions"))
        if pdim != STUDENT_PROPRIO_DIM:
            failures.append(f"student proprio dim {pdim} != {STUDENT_PROPRIO_DIM}")
        for key in ("wrist_rgb", "workspace_rgb"):
            img = stu[key]
            print(f"[shapes] student/{key}: {tuple(img.shape)} {img.dtype} "
                  f"range [{img.min().item()}, {img.max().item()}]")
            if img.shape[1:3] != (90, 160) or img.shape[-1] != 3:
                failures.append(f"{key} shape {tuple(img.shape)} != (N, 90, 160, 3)")
            if img.dtype != torch.uint8:
                failures.append(f"{key} dtype {img.dtype} != uint8 (normalize must stay False)")
        print(f"[shapes] student proprio dims: {pdim} (expect {STUDENT_PROPRIO_DIM})")

    # ---- drive: deterministic sinusoid, arm joints moving, gripper slow square wave ----
    def action_at(t: int) -> torch.Tensor:
        a = torch.zeros(n, 7, device=u.device)
        if args.static:
            return a
        for j in range(6):
            a[:, j] = 0.4 * math.sin(2 * math.pi * t / 60.0 + j * 0.7)
        a[:, 6] = 1.0 if (t // 40) % 2 == 0 else -1.0
        return a

    prev_frames: dict[str, torch.Tensor] = {}
    fresh_log: dict[str, list[float]] = {"wrist_rgb": [], "workspace_rgb": []}
    static_frames: dict[str, list[torch.Tensor]] = {"wrist_rgb": [], "workspace_rgb": []}
    depth_mins: list[float] = []
    states: list[torch.Tensor] = []

    t0 = time.time()
    for t in range(args.steps):
        obs_dict, _, _, _, _ = env.step(action_at(t))
        states.append(obs_dict["policy"].detach().cpu().clone())
        if "student" in obs_dict:
            stu = obs_dict["student"]
            for key in fresh_log:
                frame = stu[key].float()
                if key in prev_frames:
                    fresh_log[key].append((frame - prev_frames[key]).abs().mean().item())
                prev_frames[key] = frame
                if args.static and t >= WARMUP_STEPS:
                    static_frames[key].append(frame[0].cpu())
            depth = u.scene.sensors["wrist_cam"].data.output["distance_to_image_plane"]
            finite = depth[torch.isfinite(depth)]
            if finite.numel():
                depth_mins.append(finite.min().item())
    elapsed = time.time() - t0
    fps = args.steps * n / elapsed
    print(f"[fps] {args.steps} steps x {n} envs in {elapsed:.1f} s = {fps:.0f} env-steps/s "
          f"({args.steps / elapsed:.1f} policy Hz)")

    # ---- check b: freshness (skipped in --static: frames SHOULD be near-identical) ----
    for key, diffs in fresh_log.items():
        if not diffs:
            continue
        post = diffs[WARMUP_STEPS:]
        stale = [i + WARMUP_STEPS + 1 for i, d in enumerate(post) if d < FRESH_THRESH]
        print(f"[freshness] {key}: consecutive-diff min {min(post):.2f} / median "
              f"{sorted(post)[len(post) // 2]:.2f} / max {max(post):.2f} "
              f"(after {WARMUP_STEPS}-step warmup); stale steps: {stale[:10] if stale else 'none'}")
        if stale and not args.static:
            failures.append(f"{key}: {len(stale)} stale frames (first at step {stale[0]})")

    # ---- static-scene temporal noise anatomy ----
    for key, frames in static_frames.items():
        if not frames:
            continue
        stack = torch.stack(frames)  # (T, H, W, 3), env 0, static scene
        tstd = stack.std(dim=0)  # per-pixel temporal std
        tmean_drift = (stack.mean(dim=(1, 2, 3)) - stack.mean()).abs().max()
        frac_noisy = (tstd > 5.0).float().mean()
        print(f"[static-noise] {key}: per-pixel temporal std mean {tstd.mean():.2f} / "
              f"p95 {tstd.flatten().kthvalue(int(0.95 * tstd.numel())).values:.2f} / "
              f"max {tstd.max():.1f}; frac pixels std>5: {frac_noisy:.2%}; "
              f"global-mean drift {tmean_drift:.2f}")

    # ---- check c: wrist depth housing ----
    if depth_mins:
        lo, hi = min(depth_mins), max(depth_mins)
        print(f"[depth] wrist image-min depth over run: [{lo:.3f}, {hi:.3f}] m (expect ~0.02-0.05)")
        if not (0.005 <= lo <= 0.10):
            failures.append(f"wrist min depth {lo:.3f} outside housing range: render may not track the link")

    # ---- check e dump ----
    if args.dump_states:
        torch.save({"obs": torch.stack(states), "task": args.task, "seed": args.seed}, args.dump_states)
        print(f"[dump] policy-obs trajectory ({len(states)} steps) -> {args.dump_states}")

    env.close()
    print("=" * 60)
    if failures:
        print("GATE A CHECKS FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE A in-process checks PASSED (state-match check e runs offline via --compare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
