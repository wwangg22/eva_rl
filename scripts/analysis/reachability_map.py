# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Empirical reachability / approach-direction map for the RS-rebot-dev-arm.

Answers the questions any new task design has to clear before it is worth building:

* which (x, y, z) can the TCP actually reach at all?
* at each reachable cell, which *approach directions* are available -- i.e. can the
  fingers point down (top-down insert), sideways (side grasp), or forward?
* how much wrist roll is available about a given approach axis (regrasp / reorient)?

Method: sample joint configurations uniformly inside the USD joint limits, write them
to the articulation, run ``sim.forward()`` (kinematics only, no dynamics), and read the
``gripper_end`` link pose. The TCP is ``gripper_end`` + local offset (-0.075, 0, 0),
matching the ``FrameTransformerCfg`` used by every task env. The finger *approach axis*
is ``gripper_end`` local -X (fingers extend 0.089 m along it).

Pure kinematics -- no gravity, no contact -- so this is an *upper bound* on the
workspace. A cell marked unreachable here is definitively unreachable.

.. code-block:: bash

    python scripts/analysis/reachability_map.py --headless --num_envs 2048 --batches 300
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reachability / approach-direction map for the reBot arm.")
parser.add_argument("--num_envs", type=int, default=2048, help="configurations evaluated per batch")
parser.add_argument("--batches", type=int, default=300, help="number of sampling batches")
parser.add_argument("--cell", type=float, default=0.02, help="voxel edge length [m]")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default="logs/analysis/reachability")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from reBot_RL.assets import REBOT_ARM_CFG

# gripper_end -> TCP offset, identical to the FrameTransformerCfg in the task envs
TCP_OFFSET = (-0.075, 0.0, 0.0)
# fingers extend along gripper_end local -X
APPROACH_LOCAL = (-1.0, 0.0, 0.0)
GRIPPER_LINK = "gripper_end"


@configclass
class _ArmOnlySceneCfg(InteractiveSceneCfg):
    """Bare arm, no table/objects -- this is a kinematic query, not a physics scene."""

    robot = REBOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _quantize(pos: np.ndarray, cell: float) -> np.ndarray:
    return np.floor(pos / cell).astype(np.int64)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    scene = InteractiveScene(_ArmOnlySceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()

    robot = scene["robot"]
    device = robot.device
    body_idx = robot.body_names.index(GRIPPER_LINK)
    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]

    limits = torch.as_tensor(robot.data.joint_pos_limits[0], device=device).clone()  # (dof, 2)
    lo = limits[arm_dof, 0]
    hi = limits[arm_dof, 1]

    print("[map] joint limits (rad):")
    for name, l, h in zip([f"joint{i}" for i in range(1, 7)], lo.tolist(), hi.tolist()):
        print(f"       {name}: [{l:+.4f}, {h:+.4f}]  ({np.degrees(l):+7.2f} deg .. {np.degrees(h):+7.2f} deg)")

    env_origins = scene.env_origins  # (N, 3)
    tcp_offset = torch.tensor(TCP_OFFSET, device=device).repeat(args_cli.num_envs, 1)
    approach_local = torch.tensor(APPROACH_LOCAL, device=device).repeat(args_cli.num_envs, 1)

    cell = args_cli.cell
    # per-sample records, aggregated per voxel once at the end (vectorized -- a
    # per-sample python loop over ~1e6 samples is minutes of pure interpreter time)
    rec_key: list[np.ndarray] = []
    rec_az: list[np.ndarray] = []
    rec_dbin: list[np.ndarray] = []
    rec_rbin: list[np.ndarray] = []
    rec_hbin: list[np.ndarray] = []

    joint_state = torch.zeros((args_cli.num_envs, robot.num_joints), device=device)
    joint_vel = torch.zeros_like(joint_state)

    total = 0
    for b in range(args_cli.batches):
        q = lo + (hi - lo) * torch.rand((args_cli.num_envs, 6), device=device)
        joint_state[:, arm_dof] = q
        robot.write_joint_state_to_sim(joint_state, joint_vel)
        sim.forward()
        robot.update(0.0)

        link_pos = torch.as_tensor(robot.data.body_pos_w[:, body_idx, :], device=device)
        link_quat = torch.as_tensor(robot.data.body_quat_w[:, body_idx, :], device=device)

        tcp_w = link_pos + quat_apply(link_quat, tcp_offset)
        tcp_local = tcp_w - env_origins  # robot base is at the env origin
        approach_w = quat_apply(link_quat, approach_local)
        # the gripper's "roll" reference: local +Y of gripper_end (finger opening axis)
        open_axis_w = quat_apply(link_quat, torch.tensor([0.0, 1.0, 0.0], device=device).repeat(args_cli.num_envs, 1))

        p = tcp_local.cpu().numpy()
        a = approach_w.cpu().numpy()
        o = open_axis_w.cpu().numpy()

        keys = _quantize(p, cell)
        # coarse approach direction bin: sign-ish octant of the unit approach vector
        dbin = np.clip(np.round(a * 2.0).astype(np.int64), -2, 2)
        # roll bin: angle of the finger-opening axis projected perpendicular to approach
        proj = o - (np.sum(o * a, axis=1, keepdims=True)) * a
        n = np.linalg.norm(proj, axis=1, keepdims=True)
        proj = np.divide(proj, np.maximum(n, 1e-9))
        rollang = np.arctan2(proj[:, 2], proj[:, 1])
        rbin = np.floor((rollang + np.pi) / (np.pi / 6)).astype(np.int64)  # 30 deg bins

        # horizontal approach heading, 30-deg bins, only for near-horizontal approaches
        # (|approach_z| < 0.35). This is what decides whether a *side*-facing socket at a
        # given spot can actually be entered along its own axis.
        head = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
        hbin = np.floor((head + 180.0) / 30.0).astype(np.int64)
        hbin[np.abs(a[:, 2]) >= 0.35] = -1  # not a horizontal approach
        rec_hbin.append(hbin)

        rec_key.append(keys)
        rec_az.append(a[:, 2].astype(np.float32))
        # pack the 3 direction-bin components (each in [-2, 2]) into one int
        rec_dbin.append((dbin[:, 0] + 2) * 25 + (dbin[:, 1] + 2) * 5 + (dbin[:, 2] + 2))
        rec_rbin.append(rbin)
        total += p.shape[0]

        if (b + 1) % 25 == 0:
            print(f"[map] batch {b + 1}/{args_cli.batches}  samples={total}")

    os.makedirs(args_cli.out_dir, exist_ok=True)

    all_keys = np.concatenate(rec_key, axis=0)
    all_az = np.concatenate(rec_az, axis=0)
    all_dbin = np.concatenate(rec_dbin, axis=0)
    all_rbin = np.concatenate(rec_rbin, axis=0)
    all_hbin = np.concatenate(rec_hbin, axis=0)

    ks, inv, cnt = np.unique(all_keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)

    mn = np.full(len(ks), np.inf, dtype=np.float32)
    mx = np.full(len(ks), -np.inf, dtype=np.float32)
    np.minimum.at(mn, inv, all_az)
    np.maximum.at(mx, inv, all_az)

    def _n_unique_per_voxel(vals: np.ndarray) -> np.ndarray:
        """Number of distinct ``vals`` observed in each voxel."""
        pairs = np.unique(np.stack([inv, vals.astype(np.int64)], axis=1), axis=0)
        out = np.zeros(len(ks), dtype=np.int32)
        u, c = np.unique(pairs[:, 0], return_counts=True)
        out[u] = c
        return out

    ndir = _n_unique_per_voxel(all_dbin)
    nroll = _n_unique_per_voxel(all_rbin)
    centers = (ks.astype(np.float64) + 0.5) * cell

    # per-voxel bitmask of attainable horizontal approach headings (12 x 30-deg bins)
    head_mask = np.zeros(len(ks), dtype=np.int32)
    ok = all_hbin >= 0
    np.bitwise_or.at(head_mask, inv[ok], (1 << all_hbin[ok]).astype(np.int32))

    npz = os.path.join(args_cli.out_dir, "reachability.npz")
    np.savez_compressed(
        npz,
        cell=cell,
        voxel_idx=ks,
        centers=centers,
        counts=cnt,
        min_approach_z=mn,
        max_approach_z=mx,
        n_dir_bins=ndir,
        n_roll_bins=nroll,
        horiz_heading_mask=head_mask,
        total_samples=total,
    )

    # ---- summary ------------------------------------------------------------
    r = np.linalg.norm(centers[:, :2], axis=1)
    z = centers[:, 2]
    # "reliable" cells: enough samples that we are not looking at a single lucky config
    rel = cnt >= 5

    def pct(mask):
        return 100.0 * mask.sum() / max(rel.sum(), 1)

    summary = {
        "total_samples": int(total),
        "n_voxels": int(len(ks)),
        "n_voxels_reliable": int(rel.sum()),
        "cell_m": cell,
        "max_reach_xy_m": float(r.max()),
        "max_reach_z_m": float(z.max()),
        "min_reach_z_m": float(z.min()),
    }

    # table-plane bands (table top is z=0 in the task envs, robot base at origin)
    bands = {}
    for zlo, zhi, label in [
        (0.00, 0.06, "at_table_0_6cm"),
        (0.06, 0.12, "low_6_12cm"),
        (0.12, 0.20, "mid_12_20cm"),
        (0.20, 0.35, "high_20_35cm"),
    ]:
        m = rel & (z >= zlo) & (z < zhi)
        if m.sum() == 0:
            bands[label] = {"n_voxels": 0}
            continue
        # top-down capability: approach axis pointing down means approach_z <= -0.9
        topdown = m & (mn <= -0.90)
        angled = m & (mn <= -0.50)
        side = m & (np.abs(mn) < 0.25) | (m & (np.abs(mx) < 0.25))
        bands[label] = {
            "n_voxels": int(m.sum()),
            "r_xy_min_m": float(r[m].min()),
            "r_xy_max_m": float(r[m].max()),
            "frac_topdown_capable_pct": float(100.0 * topdown.sum() / m.sum()),
            "frac_45deg_down_capable_pct": float(100.0 * angled.sum() / m.sum()),
            "frac_side_capable_pct": float(100.0 * side.sum() / m.sum()),
            "median_roll_bins_of_12": float(np.median(nroll[m])),
            "median_dir_bins": float(np.median(ndir[m])),
        }
    summary["bands"] = bands

    # radial profile at table level -- where can a fixture physically go?
    prof = []
    for rlo in np.arange(0.10, 0.50, 0.05):
        m = rel & (r >= rlo) & (r < rlo + 0.05)
        if m.sum() == 0:
            prof.append({"r_lo": round(float(rlo), 3), "n_voxels": 0})
            continue
        prof.append({
            "r_lo": round(float(rlo), 3),
            "n_voxels": int(m.sum()),
            "z_min_m": float(z[m].min()),
            "z_max_m": float(z[m].max()),
            "best_downwardness": float(mn[m].min()),
            "median_roll_bins_of_12": float(np.median(nroll[m])),
        })
    summary["radial_profile"] = prof

    with open(os.path.join(args_cli.out_dir, "reachability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== REACHABILITY SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\n[map] wrote {npz}")


if __name__ == "__main__":
    main()
    simulation_app.close()
