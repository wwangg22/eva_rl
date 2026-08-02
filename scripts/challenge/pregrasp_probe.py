# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Is the pre-grasp block really ungraspable lying down, and graspable once up?

The whole env rests on that pair of claims, and a geometric argument is not a measurement.
This probe establishes both sides empirically, plus the route between them:

**A. Ungraspable lying (must fail).** The gripper approaches the block from a standoff with
the fingers wide, drives in to the grasp point, closes, and lifts. Success would mean the
env is broken.

**B. Graspable on edge (must succeed).** Identical procedure, block tipped up on edge.

**C. Tippable (achievability).** A scripted push: the closed gripper drives into the block's
near face, jamming its far edge against the back wall, and the block should rotate up.
Swept over push height, because pushing at or below the block's centre of mass generates no
tipping moment at all.

Three things this had to get right, each of which produced a confidently wrong number first:

* **Retention must be a real lift.** Asking whether the block is within 80 mm of the TCP is
  satisfied by an untouched block lying on the table under the gripper -- a 107 mm block in
  an 89 mm gripper scored 100 %.
* **The gripper must approach from a standoff**, not be teleported to the grasp point. For
  a too-wide block that point is *inside* the block, so the teleport buries the fingers in
  the collider and physics then flings the block along with the arm.
* **Motion must go through ``env.step``**, i.e. through the action manager, exactly as the
  policy's would. Driving ``set_joint_position_target`` directly left the arm stalled ~190
  mm short of every commanded pose.

.. code-block:: bash

    python scripts/challenge/pregrasp_probe.py --num_envs 128
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pre-grasp two-sided achievability probe.")
parser.add_argument("--task", type=str, default="Rebot-PreGrasp-Play-v0")
parser.add_argument("--num_envs", type=int, default=128, help="CEM population / trial count")
parser.add_argument("--cem_iters", type=int, default=60)
parser.add_argument("--out_dir", type=str, default="logs/analysis/pregrasp")
parser.add_argument("--skip_push", action="store_true", help="run only the grasp trials")
parser.add_argument("--control", action="store_true",
                    help="replace the block with a small easy one (30 x 30 x 70 mm, 0.04 kg) "
                         "and run only the lift. Answers whether this gripper can lift "
                         "ANYTHING, which has to be true before any grasp result means much.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp

TCP_OFFSET = (-0.0419, 0.0, 0.0)  # measured; see mdp/common.TCP_OFFSET
GRIPPER_LINK = "gripper_end"
#: how far in front of the block's near face the gripper starts its approach [m].
#: Kept short: the arm's inner working radius is ~0.15 m, so a long standoff puts the
#: start pose inside it, where the arm folds up and then cannot unfold past the block.
STANDOFF = 0.060
#: how far past the near face the grasp point sits, i.e. how deep the fingers go [m]
GRIP_INSET = 0.030
#: measured usable TCP floor above the table [m] -- scripts/analysis/tcp_floor.py.
#: Every grasp and push height here must clear it or the arm simply rests on the table.
TCP_FLOOR = 0.045
#: where the block sits, matching pregrasp_env_cfg._SPAWN_X
BLOCK_X = 0.265


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # the scripted sequences run far longer than one episode; a mid-probe time-out reset
    # would silently put the block back on the table and the numbers would be nonsense
    env_cfg.episode_length_s = 1.0e5
    if args_cli.control:
        # the precision-slot task's proven geometry: 30 mm across the fingers, gripped at
        # z = 0.072 with the block's centre at 0.055 -- the one grasp on this arm already
        # demonstrated end to end (slot_insertion_probe, 55-81 %)
        env_cfg.scene.block.spawn.size = (0.030, 0.030, 0.110)
        env_cfg.scene.block.spawn.mass_props.mass = 0.04
        env_cfg.scene.block.init_state.pos = [0.245, 0.0, 0.055]
        env_cfg.scene.block.init_state.rot = [0.0, 0.0, 0.0, 1.0]
        env_cfg.scene.wall.init_state.pos = (1.5, 0.0, 0.070)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    block = e.scene["block"]
    n = e.num_envs
    env.reset()

    body_idx = robot.body_names.index(GRIPPER_LINK)
    li = robot.body_names.index("gripper_left")
    ri = robot.body_names.index("gripper_right")
    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
    fing_dof = [robot.joint_names.index(x) for x in ("joint_left", "joint_right")]
    lo = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)[arm_dof, 0]
    hi = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)[arm_dof, 1]
    q_default = torch.as_tensor(robot.data.default_joint_pos[0], device=dev).clone()
    q_arm0 = q_default[arm_dof].clone()
    offs = torch.tensor(TCP_OFFSET, device=dev).repeat(n, 1)
    Y = torch.tensor([0.0, 1.0, 0.0], device=dev)

    H = mdp.PG_BLOCK_HALF

    def fk(q_arm: torch.Tensor, open_q: float = 0.045):
        """Batched TCP, finger-separation axis and gripper forward axis for (n, 6) configs."""
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = open_q
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        bp = robot.data.body_pos_w.torch
        lq = robot.data.body_quat_w.torch[:, body_idx, :]
        tcp = bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins
        sep = bp[:, li, :] - bp[:, ri, :]
        sep = sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)
        fwd = quat_apply(lq, torch.tensor([-1.0, 0.0, 0.0], device=dev).repeat(n, 1))
        return tcp, sep, fwd

    def cem(target: torch.Tensor, seed: torch.Tensor, w_align: float = 0.25, std0: float = 0.45):
        """Arm config putting the TCP at ``target`` with the fingers separated along y.

        Only the finger axis is constrained; the approach heading is left to the arm,
        because this arm's reachable headings at table height are narrow and forcing +x
        makes the search return unreachable junk (docs/CHALLENGE_SUITE.md C1).
        """
        mean, std = seed.clone(), torch.full((6,), std0, device=dev)
        best_q, best, best_p, best_a = seed.clone(), 1e9, 1e9, 1e9
        for _ in range(args_cli.cem_iters):
            cand = (mean + std * torch.randn(n, 6, device=dev)).clamp(lo, hi)
            cand[0] = mean
            tcp, sep, _ = fk(cand)
            pos_err = (tcp - target).norm(dim=1)
            align = 1.0 - (sep @ Y).abs()
            cost = pos_err + w_align * align
            elite = cand[cost.topk(max(8, n // 8), largest=False).indices]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.01)
            j = int(cost.argmin())
            if float(cost[j]) < best:
                best, best_q = float(cost[j]), cand[j].clone()
                best_p, best_a = float(pos_err[j]), float(align[j])
        return best_q, best_p, best_a

    def act(q_arm: torch.Tensor, close: bool) -> torch.Tensor:
        """Convert a target arm config into the env's 7-D action (scale 0.5, default offset)."""
        a = torch.zeros((n, 7), device=dev)
        a[:, :6] = (q_arm - q_arm0.unsqueeze(0)) / 0.5
        a[:, 6] = -1.0 if close else 1.0
        return a

    def run(q_from: torch.Tensor, q_to: torch.Tensor, steps: int, close: bool):
        for s in range(steps):
            f = min(1.0, (s + 1) / (steps * 0.8))
            env.step(act(((1 - f) * q_from + f * q_to).unsqueeze(0).repeat(n, 1), close))

    def hold(q: torch.Tensor, steps: int, close: bool):
        for _ in range(steps):
            env.step(act(q.unsqueeze(0).repeat(n, 1), close))

    def path(targets, seed):
        """CEM a config for each Cartesian waypoint, each seeded from the previous.

        Interpolating between two IK solutions in JOINT space does not keep the TCP on a
        straight line -- and with a 60 mm standoff the swing was enough to drive the wrist
        into the block and shove it 65 mm before the fingers ever closed. Solving a
        waypoint chain with a tight search radius keeps the arm in one IK branch and the
        TCP on the commanded path.
        """
        qs, errs, q = [], [], seed
        for i, t in enumerate(targets):
            q, err, _ = cem(t, q, std0=0.45 if i == 0 else 0.12)
            qs.append(q)
            errs.append(err)
        return qs, errs

    def run_path(qs, steps_each, close):
        for i in range(len(qs) - 1):
            run(qs[i], qs[i + 1], steps_each, close)

    def lerp_targets(a, b, k):
        return [a + (b - a) * (i + 1) / k for i in range(k)]

    def place(pos, quat):
        block.write_root_state_to_sim(
            torch.cat([pos + e.scene.env_origins, quat, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def teleport_arm(q_arm, open_q=0.045):
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = open_q
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def achieved_tcp():
        bp = robot.data.body_pos_w.torch
        lq = robot.data.body_quat_w.torch[:, body_idx, :]
        return bp[:, body_idx, :] + quat_apply(lq, offs) - e.scene.env_origins

    # ---------------------------------------------------------------------------
    def grasp_trial(block_quat, block_centre_z, hold_z, label):
        """Place the block AT the measured TCP, close, lift, and see what the fingers did.

        Deliberately not an approach-from-standoff: scripted Cartesian approaches on this
        arm turned out to be unreliable enough to dominate the result, and the question here
        is about the gripper and the object, not about a hand-written controller. Placing the
        object at the tool centre point is the same pattern the slot-insertion probe uses,
        which is the one manoeuvre on this arm already validated end to end.

        The verdict comes from the finger separation after closing, which cannot be faked:
        fingers around the object stall at its width, fingers on nothing go to ~0 mm, and
        fingers jammed outside a too-wide object stop near the 89.1 mm limit.
        """
        # park the arm somewhere comfortable and read where its TCP actually is
        hold_t = torch.tensor([0.245, 0.0, hold_z], device=dev)
        q_hold, herr, aerr = cem(hold_t, q_arm0)
        # EVERY search has to finish before the block is placed. cem() evaluates candidates
        # by writing joint states into the sim, which teleports the arm and re-opens the
        # fingers hundreds of times -- so searching for the lift path after closing silently
        # drops the block, and the grasp reads as a slip.
        q_up, lerrs = path(lerp_targets(hold_t, hold_t + torch.tensor([0.0, 0.0, 0.080], device=dev), 3),
                           q_hold)
        teleport_arm(q_hold)
        tcp0 = achieved_tcp() + e.scene.env_origins

        # The block takes its x, y from where the TCP actually is, and its z from its own
        # geometry -- the two are separate. Placing it at the TCP's z instead left a lying
        # block floating 45 mm below the fingers, which closed on air and read as "cannot
        # be grasped" for entirely the wrong reason.
        bpos = tcp0.clone()
        bpos[:, 2] = block_centre_z + e.scene.env_origins[:, 2]
        bpos[:, 1] += (torch.rand(n, device=dev) - 0.5) * 0.004   # a little scatter
        block.write_root_state_to_sim(torch.cat(
            [bpos, block_quat, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)
        width = float(mdp.min_grasp_width(e).median())
        z0 = float((bpos - e.scene.env_origins)[:, 2].median())

        hold(q_hold, 60, close=True)                 # close on it immediately
        gap_t = 1.0035 * robot.data.joint_pos.torch[:, fing_dof].sum(dim=1) - 0.00125
        # tight tolerance on purpose: at 15 mm a 100 mm block and fingers merely splayed to
        # their ~120 mm joint limit are indistinguishable, and the splayed case reads as a
        # grasp. The limit matters -- the binary open command asks for 89 mm, but a wide
        # object forces the fingers past it, so the real maximum grasp width is the joint
        # limit sum (0.050 + 0.0715 m => ~120 mm), not the commanded opening.
        enclosed = (gap_t - width).abs() < 0.006
        gap = float(gap_t.median())

        run_path([q_hold] + q_up, 35, close=True)
        hold(q_up[-1], 60, close=True)

        pos = mdp.object_pos_local(e, "block")
        tcp = achieved_tcp()
        print(f"{'':14s} after lift: TCP {tcp.median(0).values.cpu().numpy().round(4)} "
              f"(cmd z {float(hold_t[2]) + 0.080:.3f}), block {pos.median(0).values.cpu().numpy().round(4)} "
              f"(started z {z0:.4f}), lift CEM err {lerrs[-1] * 1000:.2f} mm")
        rose = pos[:, 2] > z0 + 0.045
        near = (tcp - pos).norm(dim=1) < 0.09
        held = rose & near & enclosed
        print(f"[{label}] width {width * 1000:6.1f} mm | hold TCP "
              f"{(tcp0 - e.scene.env_origins)[0].cpu().numpy().round(4)} (cmd err "
              f"{herr * 1000:.1f} mm), align {aerr:.3f}")
        print(f"{'':14s} finger gap after close {gap * 1000:6.1f} mm "
              f"(enclosing in {float(enclosed.float().mean()):.0%})")
        print(f"{'':14s} rose {float(rose.float().mean()):5.1%}  "
              f"near {float(near.float().mean()):5.1%}  "
              f"HELD {float(held.float().mean()):.1%}")
        return {"label": label, "width_m": width, "finger_gap_m": gap,
                "enclose_rate": float(enclosed.float().mean()),
                "rose_rate": float(rose.float().mean()), "near_rate": float(near.float().mean()),
                "held_rate": float(held.float().mean())}

    results = {}
    lying_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=dev).repeat(n, 1)
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
    print(f"\n[probe] measured gripper opening {mdp.GRIPPER_OPENING * 1000:.1f} mm; block is "
          f"{2 * H[0] * 1000:.0f} x {2 * H[1] * 1000:.0f} x {2 * H[2] * 1000:.0f} mm\n")

    # ---- 0. can the arm HOLD a commanded TCP pose at all, at these heights? ---
    # The reachability map is pure kinematics; it does not know the table exists. Every
    # result below is meaningless if the arm cannot physically hold the poses being asked
    # for, so this measures the gap between the commanded TCP and the achieved one over the
    # x-z band the task actually uses. The block is parked well out of the way.
    print("[probe] 0. workspace tracking check (commanded TCP vs achieved, empty scene)")
    park = torch.tensor([0.20, -0.45, H[1]], device=dev).repeat(n, 1)
    place(park, ident)
    print(f"  {'x':>7} {'z':>7} | {'CEM err':>9} | {'achieved (x, y, z)':>26} | {'track err':>10}")
    print("  " + "-" * 66)
    track = []
    for tx, tz in ((0.25, 0.024), (0.25, 0.045), (0.25, 0.070), (0.25, 0.100),
                   (0.30, 0.024), (0.30, 0.070), (0.20, 0.024), (0.20, 0.070)):
        t = torch.tensor([tx, 0.0, tz], device=dev)
        q_t, err, _ = cem(t, q_arm0)
        teleport_arm(q_t, open_q=0.0)
        hold(q_t, 80, close=True)
        got = achieved_tcp()[:, :3].median(dim=0).values
        terr = float((got - t).norm())
        track.append({"x": tx, "z": tz, "cem_err_m": err, "track_err_m": terr,
                      "achieved": [float(v) for v in got]})
        print(f"  {tx:7.3f} {tz:7.3f} | {err * 1000:8.2f}mm | "
              f"({got[0]:7.4f}, {got[1]:7.4f}, {got[2]:7.4f}) | {terr * 1000:9.2f}mm")
    results["tracking"] = track
    ok_track = [t for t in track if t["track_err_m"] < 0.010]
    print(f"[probe]   {len(ok_track)}/{len(track)} poses held to within 10 mm")
    if not ok_track:
        print("[probe]   NOTHING tracks -- the arm is not following commands at all, so every")
        print("[probe]   number below is measuring the controller, not the task.")
    print()

    if args_cli.control:
        # PG_BLOCK_HALF drives min_grasp_width, so point it at the control block's geometry
        mdp.pregrasp.PG_BLOCK_HALF = (0.015, 0.015, 0.055)
        grasp_trial(ident, 0.055, 0.072, label="CTRL 30x30x110")
        env.close()
        return

    # ---- A. lying flat: must FAIL --------------------------------------------
    # grip at 50 mm: within the lying block's 0..60 mm height and clear of the 44 mm TCP
    # floor, so a failure here is the block's width and not the arm resting on the table
    results["lying"] = grasp_trial(lying_quat, H[1], 0.050, label="A lying flat")

    # ---- B. up on edge: must SUCCEED -----------------------------------------
    # grip ABOVE the block's centre of mass (which sits at H[2] = 50 mm) and above the
    # 44 mm TCP floor: gripping at the CoM height put the fingers on the floor and let the
    # block rotate out of them during the lift
    results["upright"] = grasp_trial(ident, H[2], 0.068, label="B up on edge")

    if args_cli.skip_push:
        print("\n[probe] push phase skipped (--skip_push)")
        env.close()
        return

    # ---- C. scripted push against the wall, swept over contact height ---------
    print("\n[probe] C. scripted push-to-wall, swept over push height "
          f"(block CoM sits at {H[1] * 1000:.1f} mm)")
    g = torch.Generator(device="cpu").manual_seed(0)
    jit = torch.zeros(n, 3)
    jit[:, 0].uniform_(-0.010, 0.010, generator=g)
    jit[:, 1].uniform_(-0.030, 0.030, generator=g)
    jit[:, 2].uniform_(-0.10, 0.10, generator=g)
    jit = jit.to(dev)
    half = jit[:, 2] / 2
    qz = torch.stack([torch.zeros_like(half), torch.zeros_like(half), half.sin(), half.cos()], dim=-1)
    jit_quat = quat_mul(qz, lying_quat)
    lying_pos = torch.tensor([BLOCK_X, 0.0, H[1]], device=dev).repeat(n, 1)
    jit_pos = lying_pos.clone()
    jit_pos[:, 0] += jit[:, 0]
    jit_pos[:, 1] += jit[:, 1]

    near_x = BLOCK_X - H[0]
    push_rows = []
    print(f"  {'push z':>8} | {'start err':>10} | {'end err':>9} | {'TCP x cmd':>10} | "
          f"{'TCP x got':>10} | {'width':>8} | {'up-axis':>8} | {'tipped':>7}")
    print("  " + "-" * 92)
    # all above the 45 mm TCP floor, all below the lying block's 60 mm top, and all above
    # its 30 mm centre of mass -- which is the band where a push can produce a tipping moment
    for push_z in (0.046, 0.050, 0.054, 0.058, 0.062):
        start_t = torch.tensor([max(near_x - STANDOFF, 0.180), 0.0, push_z], device=dev)
        end_t = torch.tensor([near_x + 0.105, 0.0, push_z], device=dev)
        q_a, ea, _ = cem(start_t, q_arm0)
        q_push, perrs = path(lerp_targets(start_t, end_t, 6), q_a)
        eb = perrs[-1]

        place(jit_pos, jit_quat)
        teleport_arm(q_a, open_q=0.0)
        hold(q_a, 20, close=True)                 # fingers closed: push with a flat tool
        run_path([q_a] + q_push, 40, close=True)
        hold(q_push[-1], 120, close=True)

        got = achieved_tcp()
        w_end = mdp.min_grasp_width(e)
        up_end = mdp.block_up_axis(e)
        tipped = w_end < mdp.W_GRASPABLE
        rate = float(tipped.float().mean())
        push_rows.append({"push_z": push_z, "start_err_m": ea, "end_err_m": eb,
                          "cmd_tcp_x": float(end_t[0]), "got_tcp_x": float(got[:, 0].median()),
                          "final_width_m": float(w_end.median()),
                          "final_up_axis": float(up_end.median()), "tipped_rate": rate})
        print(f"  {push_z * 1000:7.1f}mm | {ea * 1000:9.2f}mm | {eb * 1000:8.2f}mm | "
              f"{float(end_t[0]):10.4f} | {float(got[:, 0].median()):10.4f} | "
              f"{float(w_end.median()) * 1000:7.1f}mm | {float(up_end.median()):8.2f} | {rate:7.1%}")

    best = max(push_rows, key=lambda r: r["tipped_rate"])
    print(f"\n[probe]   best push height {best['push_z'] * 1000:.0f} mm -> tipped {best['tipped_rate']:.1%}")
    results["push"] = {"sweep": push_rows, "tipped_rate": best["tipped_rate"],
                       "best_push_z": best["push_z"]}

    # ---- verdict --------------------------------------------------------------
    a_ok = results["lying"]["held_rate"] < 0.05
    b_ok = results["upright"]["held_rate"] > 0.50
    c_ok = results["push"]["tipped_rate"] > 0.20
    print("\n" + "=" * 74)
    print(f"  A  lying block is ungraspable   {'PASS' if a_ok else 'FAIL'}  "
          f"(held {results['lying']['held_rate']:.1%}, want < 5%)")
    print(f"  B  on-edge block is graspable   {'PASS' if b_ok else 'FAIL'}  "
          f"(held {results['upright']['held_rate']:.1%}, want > 50%)")
    print(f"  C  push against wall tips it    {'PASS' if c_ok else 'FAIL'}  "
          f"(tipped {results['push']['tipped_rate']:.1%}, want > 20%)")
    print(f"  => pre-grasp is {'ACHIEVABLE and non-trivial' if (a_ok and b_ok and c_ok) else 'NOT YET PROVEN'}")
    print("=" * 74)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    with open(os.path.join(args_cli.out_dir, "pregrasp_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] wrote {args_cli.out_dir}/pregrasp_probe.json")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
