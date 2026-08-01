# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate pick-and-place demonstrations with a HYBRID expert: RL grasps, script places.

Pure kinematic scripting failed on the grasp (differential IK stalls at a z-floor
~0.045 m; joint-table replays jam on contact), but the RL policy from run 13 is a proven
reach+grasp+carry primitive (holds the can 95% of steps) that, with canonical
observations, automatically targets the nearest unplaced object -- its only flaw is that
it never releases. The scripted carry-place stages are proven too. So each episode
alternates: the RL policy reaches and grasps; once the object is held and lifted,
a scripted position-IK sequence carries it over the basket, lowers, releases and
retreats; control returns to the RL policy for the next object.

Actions are recorded in the environment's NATIVE action space, so the (obs, action)
pairs train a BC policy that runs unmodified in the same env and can be RL-fine-tuned.
Only successful episodes (both objects placed, nothing dropped) are saved.

``--grasp_mode kinematic`` replaces the RL grasp with a pure-kinematic state machine:
PREK/SLIDEK interpolate joints through a sim-verified grasp-table config (angle-aware
NN lookup: radial entry, corridor kept clear of the other can, lying cans grasped
across the barrel), FINEK does live position-IK onto the grasp pocket, CLOSEK grips,
then the usual LIFT..UP place stages run.

.. code-block:: bash

    python scripts/scripted_expert/generate_pick_place.py --headless \\
        --num_envs 64 --episodes 512

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate hybrid-expert pick-and-place demonstrations.")
parser.add_argument("--task", type=str, default="Rebot-PickPlace-v0")
parser.add_argument(
    "--rl_checkpoint",
    type=str,
    default="logs/rl_games/rebot_pick_place/2026-07-30_17-03-34/nn/rebot_pick_place.pth",
    help="rl_games checkpoint used as the grasp primitive (run-13 best).",
)
parser.add_argument(
    "--grasp_mode",
    type=str,
    default="rl",
    choices=["rl", "kinematic"],
    help="Grasp primitive: the RL policy, or a grasp-table + joint-interpolation state machine.",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=512, help="Target number of SUCCESSFUL episodes to save.")
parser.add_argument("--max_rounds", type=int, default=40, help="Safety cap on collection rounds.")
parser.add_argument("--steps", type=int, default=600, help="Steps per episode (50 Hz).")
parser.add_argument("--out_dir", type=str, default=None, help="Default: data/pick_place_demos/<timestamp>/")
parser.add_argument("--episodes_per_shard", type=int, default=128)
parser.add_argument("--video", action="store_true", help="Record a workspace video of env 0 (first round).")
parser.add_argument(
    "--grasp_test",
    action="store_true",
    help="Fast tuning mode: 275 steps, report first-can grasp/place per 16-env sweep block, save nothing.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.grasp_test:
    args_cli.steps = 275
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
from datetime import datetime

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.pick_place import mdp

# TCP sits 0.075 m along gripper_end local -X (see the lift env's ee_frame offset)
_TCP_OFFSET = 0.075
_ARM_SCALE = 0.5  # JointPositionActionCfg scale
_OPEN, _CLOSE = 1.0, -1.0  # BinaryJointPositionAction: action < 0 -> close

# scripted place stages (position-only IK): (name, tol [m], timeout [s]); tol<=0 = fixed
_STAGES = [
    ("LIFT", 0.03, 1.2),  # straight up with the held object
    ("MOVE", 0.020, 2.0),  # over the basket center
    ("LOWER", 0.015, 1.5),  # just above the rim
    ("OPEN", -1.0, 0.5),  # release
    ("UP", 0.030, 1.0),  # retreat clear of the basket
    ("HOME", 0.045, 2.5),  # back to the RL policy's familiar start region, then hand over
]
_S = {name: i for i, (name, _, _) in enumerate(_STAGES)}
_DONE_STAGE = len(_STAGES)
_RL = -1  # stage value meaning "RL policy in control"

_CMD_STEP = 0.006  # max commanded-position step per control tick (0.3 m/s at 50 Hz)
_CARRY_Z = 0.12
_LOWER_Z = 0.075
_HELD_DIST = 0.07
_HELD_LIFT_Z = 0.055  # object height that triggers the handoff to the scripted place
_HELD_STEPS = 5  # consecutive held+lifted steps before handoff

# -- kinematic grasp mode (--grasp_mode kinematic). Stage codes are negative so they never
# collide with the scripted place stages; _RL keeps meaning "needs a grasp" and triggers a
# (re)start of the kinematic machine instead of the policy.
_K_PRE, _K_SLIDE, _K_FINE, _K_CLOSE, _K_LIFT = -2, -3, -4, -5, -6
# joint-space place stages: the whole carry runs on interpolated joints through
# FK-verified waypoints (data/pick_place_demos/carry_waypoints.pt, see probe_carry_fk.py).
# Differential IK is NOT used: from the table's limit-hugging grasp configs (and from the
# vertical default posture) DLS cannot track and the arm diverges over the base.
_K_CARRY, _K_LOW, _K_OPEN, _K_UP, _K_YAW, _K_CHECK = -7, -8, -9, -10, -11, -12
# CHECKK-FAIL recovery: lower the can back to the table (grip closed), THEN open and
# back off. Opening at the micro-lift height tip-and-rolled the can into its neighbor.
_K_DOWN, _K_BACK = -13, -14
# Lateral alignment before the slide-in: table quantization leaves ~2 cm lateral error
# and sliding in with it PLOWS the can flat (run 7: 26 SLIDEK knock events). A live-IK
# align stage made it WORSE (run 8: DLS from the pre config diverges and sweeps the can,
# 20 self-tips) -- instead the azimuth error is corrected analytically ONCE at pre_done:
# joint1 yaws the pocket azimuth EXACTLY (FK-probe fact), so dj1 = az(can) - az(pocket).
_K_WAYPOINTS = "data/pick_place_demos/carry_waypoints.pt"
_K_TABLE = "data/pick_place_demos/grasp_table.pt"
_K_AIM_DZ = 0.004  # aim this far above the object center [m]
_K_POCKET = 0.027  # grasp pocket sits this far behind the TCP along the finger axis [m]
_K_BACKOFF = 0.05  # pre-grasp pocket backoff along the finger axis [m]
_K_ZW = 3.0  # z weight in the grasp-table NN cost
_K_TOPK = 32  # nearest table samples re-ranked by approach angle
_K_CLEAR = 0.04  # xy clearance the approach corridor should keep from the other can [m]
_K_LYING = 0.7  # |uprightness| below this = can knocked over
_K_LIE_Z = 0.012  # lying-can center height (cylinder radius) [m]
_K_DQ = 0.02  # max commanded joint step per control tick [rad]
_K_JTOL = 0.10  # joint-stage advance tolerance [rad]
_K_JTIMEOUT = 2.0  # joint-stage timeout [s]
_K_FINE_T = 0.6  # FINEK duration [s]
_K_FINE_TOL = 0.006  # FINEK early-exit TCP error [m]
_K_CLOSE_T = 0.6  # CLOSEK duration [s]
_K_CHECK_T = 0.5  # CHECKK (micro-lift grip verification) timeout [s]
_K_CHECK_D2 = 0.08  # CHECKK joint2 delta (~2 cm pocket lift)
_K_DOWN_T = 0.6  # DOWNK (reverse the micro-lift before reopening) timeout [s]
_K_BACK_T = 1.0  # BACKK (open + back off to the pre config) timeout [s]
_K_ALIGN_MAX = 0.3  # clamp on the one-shot joint1 azimuth correction [rad]
_K_SLIDE_SLOW = 0.12  # SLIDEK joint err below this -> creep at dq 0.01 (must exceed the
# _K_JTOL 0.10 stage-exit tolerance or the creep zone never engages before FINEK)
_K_TIP = 0.95  # mid-slide uprightness abort threshold (upright-selected cans only)
_K_DROOP = 0.04  # tcp_z - can_z beyond which the grip is judged slipping [m]
_K_LIFT_T = 1.5  # LIFTK (joint2 lift off the table) timeout [s]
_K_YAW_T = 2.0  # YAWK (joint1-only swing onto the basket azimuth) timeout [s]
_K_CARRY_T = 2.5  # CARRYK (swing to over-basket waypoint) timeout [s]
_K_LOW_T = 1.5  # LOWK (descend to release waypoint) timeout [s]
_K_UP_T = 1.5  # UPK (retreat back to over-basket waypoint) timeout [s]
_K_PLACE_TOL = 0.05  # joint tolerance for the carry/lower waypoints [rad] (~1 cm at TCP)

# --grasp_test sweep: 16-env blocks, block b overrides these knobs. live_off updates the
# carry-offset compensation DURING the carry (the hang offset rotates with the wrist,
# a frozen measurement goes stale by ~2-3 cm); lower_z is the release height; open_t
# holds the opened gripper still before the UP retreat.
_SWEEP = [
    {"live_off": False, "lower_z": 0.075, "open_t": 0.5},
    {"live_off": True, "lower_z": 0.075, "open_t": 0.5},
    {"live_off": True, "lower_z": 0.065, "open_t": 0.5},
    {"live_off": True, "lower_z": 0.075, "open_t": 1.5},
]
_BLOCK = 16


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None  # fixed-length rounds, manual resets
    # demos always start with both objects on the table
    env_cfg.events.prestage_object_a = None
    env_cfg.events.prestage_object_b = None
    if args_cli.video:
        from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG

        env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=640, height=360)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    kin = args_cli.grasp_mode == "kinematic"
    agent, resume_path = None, None
    if not kin:
        resume_path = os.path.abspath(args_cli.rl_checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

    uenv = env.unwrapped
    n, device = uenv.num_envs, uenv.device
    robot = uenv.scene["robot"]
    ee_frame = uenv.scene["ee_frame"]

    arm_joint_ids = robot.find_joints(["joint[1-6]"])[0]
    ee_body_id = robot.find_bodies("gripper_end")[0][0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id
    jacobi_joint_ids = [j + robot.num_base_dofs for j in arm_joint_ids]
    q_default = robot.data.default_joint_pos.torch[:, arm_joint_ids]

    ik_pos = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        num_envs=n,
        device=device,
    )
    basket_c = torch.tensor(mdp.BASKET_CENTER, device=device)

    if kin:
        tab = torch.load(_K_TABLE, map_location=device)
        tab_q, tab_pocket, tab_fwd = tab["q"], tab["pocket"], tab["fwd"]
        wp = torch.load(_K_WAYPOINTS, map_location=device)
        q_over_w, q_lower_w = wp["q_over"].to(device), wp["q_lower"].to(device)  # (2,6) radius bands
        d2_lift = float(wp["d2_lift"])
        r_split = float(wp["r_split"])
        try:
            jlims = robot.data.joint_pos_limits.torch[0]
        except AttributeError:
            jlims = robot.data.joint_limits.torch[0]
        j2_lo = float(jlims[arm_joint_ids[1], 0]) + 0.02
        j2_hi = float(jlims[arm_joint_ids[1], 1]) - 0.02
        for b in range(q_over_w.shape[0]):
            print(
                f"[kin] CARRY=joint-space band {b} (r_split {r_split}) d2_lift {d2_lift:+.2f}"
                f" q_over {[round(float(x), 3) for x in q_over_w[b]]}"
                f" pocket_over {[round(float(x), 3) for x in wp['pocket_over'][b]]}"
                f" q_lower {[round(float(x), 3) for x in q_lower_w[b]]}"
                f" pocket_lower {[round(float(x), 3) for x in wp['pocket_lower'][b]]}"
            )

    # per-env knobs (uniform defaults; --grasp_test sweeps them per 16-env block)
    blk = (torch.arange(n, device=device) // _BLOCK).clamp(max=len(_SWEEP) - 1)
    live_off_t = torch.ones(n, dtype=torch.bool, device=device)
    lower_z_t = torch.full((n,), _LOWER_Z, device=device)
    open_t_t = torch.full((n,), _STAGES[_S["OPEN"]][2], device=device)
    if kin and args_cli.grasp_test:
        for b, cfg in enumerate(_SWEEP):
            m = blk == b
            live_off_t[m] = cfg["live_off"]
            lower_z_t[m] = cfg["lower_z"]
            open_t_t[m] = cfg["open_t"]

    writer = None
    out_dir = args_cli.out_dir or os.path.join("data", "pick_place_demos", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    if args_cli.video:
        import imageio.v2 as imageio

        writer = imageio.get_writer(os.path.join(out_dir, "expert_env0.mp4"), fps=50, macro_block_size=1)

    def object_pos_local(name):
        return uenv.scene[name].data.root_pos_w.torch - uenv.scene.env_origins

    def tcp_pos_local():
        return ee_frame.data.target_pos_w.torch[..., 0, :] - uenv.scene.env_origins

    def held_unplaced_mask():
        """(n,) True where some unplaced object is held and lifted."""
        tcp = tcp_pos_local()
        placed = mdp.placed_mask(uenv)
        out = torch.zeros(n, dtype=torch.bool, device=device)
        for i, name in enumerate(mdp.OBJECT_NAMES):
            pos = object_pos_local(name)
            held = torch.linalg.norm(pos - tcp, dim=1) < _HELD_DIST
            out |= held & (pos[:, 2] > _HELD_LIFT_Z) & ~placed[:, i]
        return out

    # -- collection state
    saved, attempted, shard_idx = [], 0, 0
    total_success = 0

    def flush(force=False):
        nonlocal saved, shard_idx
        while len(saved) >= args_cli.episodes_per_shard or (force and saved):
            chunk, saved = saved[: args_cli.episodes_per_shard], saved[args_cli.episodes_per_shard :]
            path = os.path.join(out_dir, f"shard_{shard_idx:05d}.pt")
            torch.save(chunk, path)
            print(f"[save] {len(chunk)} episodes -> {path}")
            shard_idx += 1

    rounds = 0
    while total_success < args_cli.episodes and rounds < args_cli.max_rounds:
        rounds += 1
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        if agent is not None:
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
        ik_pos.reset()

        stage = torch.full((n,), _RL, dtype=torch.long, device=device)
        stage_t = torch.zeros(n, device=device)
        tgt_pos = torch.zeros(n, 3, device=device)
        grip = torch.full((n,), _CLOSE, device=device)
        held_ctr = torch.zeros(n, dtype=torch.long, device=device)
        dropped = torch.zeros(n, dtype=torch.bool, device=device)
        done_step = torch.full((n,), args_cli.steps, dtype=torch.long, device=device)
        cmd_pos = tcp_pos_local().clone()
        q_cmd = q_default.clone()  # kinematic joint-space command (interpolated)
        q_ktgt = q_default.clone()  # current joint-stage target (q_pre or q_grasp)
        kin_q_grasp = q_default.clone()
        kin_q_pre = q_default.clone()  # pre-grasp config, reused as the BACKK retreat target
        kin_slide_retry = torch.zeros(n, dtype=torch.long, device=device)  # mid-slide tip aborts
        kin_obj = torch.zeros(n, dtype=torch.long, device=device)
        kin_band = torch.zeros(n, dtype=torch.long, device=device)  # carry-waypoint radius band
        kin_lying = torch.zeros(n, dtype=torch.bool, device=device)  # grasp branch used
        # xy of (held can - TCP), measured during LIFT: the place stages then put the CAN
        # over the basket center, not the TCP (the pocket hangs ~2.5 cm behind the TCP).
        # Stays zero in rl mode.
        carry_off = torch.zeros(n, 2, device=device)
        # carry divergence diagnostic: max per-step TCP displacement during LIFTK/MOVE/LOWER
        carry_jump = torch.zeros(n, device=device)
        tcp_prev = tcp_pos_local().clone()

        obs_buf = torch.zeros(args_cli.steps, n, obs.shape[1], device=device)
        act_buf = torch.zeros(args_cli.steps, n, 7, device=device)
        first_grasp = torch.full((n,), -1, dtype=torch.long, device=device)
        first_place = torch.full((n,), -1, dtype=torch.long, device=device)

        def enter_stage(env_ids, s_idx):
            if len(env_ids) == 0:
                return
            stage[env_ids] = s_idx
            stage_t[env_ids] = 0.0
            name = _STAGES[s_idx][0] if 0 <= s_idx < _DONE_STAGE else "OTHER"
            if name == "LIFT":
                # take over from the RL policy at the current pose
                tcp = tcp_pos_local()[env_ids]
                cmd_pos[env_ids] = tcp
                tgt_pos[env_ids, 0] = tcp[:, 0]
                tgt_pos[env_ids, 1] = tcp[:, 1]
                tgt_pos[env_ids, 2] = _CARRY_Z
                grip[env_ids] = _CLOSE
            elif name in ("MOVE", "LOWER", "OPEN"):
                tgt_pos[env_ids, 0] = basket_c[0] - carry_off[env_ids, 0]
                tgt_pos[env_ids, 1] = basket_c[1] - carry_off[env_ids, 1]
                tgt_pos[env_ids, 2] = _CARRY_Z if name == "MOVE" else lower_z_t[env_ids]
                grip[env_ids] = _OPEN if name == "OPEN" else _CLOSE
            elif name == "UP":
                tgt_pos[env_ids, 2] = 0.13
                grip[env_ids] = _OPEN
            elif name == "HOME":
                # the RL policy only ever started episodes from the home TCP region --
                # handing over above the basket left it frozen (0 second grasps)
                tgt_pos[env_ids, 0] = 0.34
                tgt_pos[env_ids, 1] = 0.0
                tgt_pos[env_ids, 2] = 0.16
                grip[env_ids] = _OPEN
            if kin and name in ("OPEN", "UP"):
                o_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
                for e in env_ids.tolist():
                    o = o_all[e, int(kin_obj[e])]
                    jump = f" max_jump {float(carry_jump[e]) * 1000:.1f}mm" if name == "OPEN" else ""
                    print(
                        f"[kin] {name} env {e}: can rel basket"
                        f" ({float(o[0] - basket_c[0]):+.3f},{float(o[1] - basket_c[1]):+.3f}) z {float(o[2]):.3f}{jump}"
                    )

        def kin_start(env_ids):
            """(Re)start a kinematic grasp: nearest unplaced object, angle-aware table lookup."""
            if len(env_ids) == 0:
                return
            placed_m = mdp.placed_mask(uenv)
            obj_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
            quat_all = torch.stack([uenv.scene[nm].data.root_quat_w.torch for nm in mdp.OBJECT_NAMES], dim=1)
            tcp = tcp_pos_local()
            for e in env_ids.tolist():
                # nearest-to-TCP selection. Do NOT swap to nearest-to-base ("near can
                # first"): run 9 showed PREK from home into a NEAR-can pre config sweeps
                # the arm through the can annulus and plows BOTH cans (grasped 13/32).
                d = torch.linalg.norm(obj_all[e] - tcp[e], dim=1) + placed_m[e].float() * 1e3
                pick = int(d.argmin())
                o = obj_all[e, pick]
                # can body Y is the cylinder axis (world Z when upright, horizontal when knocked over)
                axis = math_utils.quat_apply(quat_all[e, pick : pick + 1], torch.tensor([[0.0, 1.0, 0.0]], device=device))[0]
                lying = abs(float(axis[2])) < _K_LYING
                aim = o.clone()
                aim[2] = (_K_LIE_Z if lying else float(o[2])) + _K_AIM_DZ
                # top-K nearest pockets, re-ranked by approach angle
                dp = tab_pocket - aim
                cost_k, idx_k = (dp[:, 0] ** 2 + dp[:, 1] ** 2 + (_K_ZW * dp[:, 2]) ** 2).topk(_K_TOPK, largest=False)
                f_k = tab_fwd[idx_k]
                f_xy = f_k[:, :2] / torch.linalg.norm(f_k[:, :2], dim=1, keepdim=True).clamp(min=1e-9)
                r_in = -aim[:2] / torch.linalg.norm(aim[:2]).clamp(min=1e-9)  # radial-entry direction
                align = f_xy @ r_in
                score = 100.0 * cost_k + (1.0 - align)
                perp = None
                if lying:
                    # WAIST GRIP: finger axis near-perpendicular to the can's long axis.
                    # Table f_xy is near-radial (dot-vs-radial 0.71..1.0, offline check),
                    # so the hard |dot|<0.35 filter is only satisfiable for tangentially-
                    # lying cans; for radial lies NO table entry qualifies -- fall back to
                    # the most perpendicular entries the table has (weight 4.0, was 1.5).
                    a_xy = axis[:2] / torch.linalg.norm(axis[:2]).clamp(min=1e-9)
                    perp = (f_xy @ a_xy).abs()
                    if bool((perp < 0.35).any()):
                        score += 100.0 * (perp >= 0.35).float()
                    else:
                        score += 4.0 * perp
                # pre-pocket: backed off along the finger axis, on the larger-radius side
                pre_k = aim - _K_BACKOFF * f_k
                flip = torch.linalg.norm(pre_k[:, :2], dim=1) < torch.linalg.norm(aim[:2])
                pre_k[flip] = (aim + _K_BACKOFF * f_k)[flip]
                # keep the approach corridor (pre-pocket -> pocket, xy) clear of the other cans
                for j in range(len(mdp.OBJECT_NAMES)):
                    if j == pick:
                        continue
                    seg = aim[:2] - pre_k[:, :2]
                    rel = obj_all[e, j, :2] - pre_k[:, :2]
                    frac = ((rel * seg).sum(1) / (seg * seg).sum(1).clamp(min=1e-9)).clamp(0.0, 1.0)
                    clear = torch.linalg.norm(rel - frac.unsqueeze(1) * seg, dim=1)
                    score += 4.0 * (_K_CLEAR - clear).clamp(min=0.0) / _K_CLEAR
                best = int(score.argmin())
                gi = idx_k[best]
                # pre config: pocket near the backed-off point AND a similar finger direction
                dq = tab_pocket - pre_k[best]
                pre_cost = dq[:, 0] ** 2 + dq[:, 1] ** 2 + (_K_ZW * dq[:, 2]) ** 2 + 0.01 * (1.0 - tab_fwd @ tab_fwd[gi])
                pi = int(pre_cost.argmin())
                kin_obj[e] = pick
                kin_q_grasp[e] = tab_q[gi]
                kin_band[e] = 1 if float(torch.linalg.norm(tab_pocket[gi, :2])) > r_split else 0
                kin_lying[e] = lying
                kin_q_pre[e] = tab_q[pi]
                q_ktgt[e] = tab_q[pi]
                extra = f" perp {float(perp[best]):.2f}" if lying else ""
                print(
                    f"[kin] env {e}: obj {pick} upright {float(axis[2]):+.2f}"
                    f"{' LYING' if lying else ''} align {float(align[best]):+.2f}"
                    f" pocket_err {float(cost_k[best]) ** 0.5 * 1000:.1f}mm{extra}"
                )
            q_cmd[env_ids] = robot.data.joint_pos.torch[env_ids][:, arm_joint_ids]
            stage[env_ids] = _K_PRE
            stage_t[env_ids] = 0.0
            grip[env_ids] = _OPEN
            kin_slide_retry[env_ids] = 0

        def kin_diag(name, env_ids):
            """Release diagnostics for the joint-space place stages (same format as enter_stage's)."""
            o_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
            if name == "OPEN":
                quat_all = torch.stack([uenv.scene[nm].data.root_quat_w.torch for nm in mdp.OBJECT_NAMES], dim=1)
            for e in env_ids.tolist():
                o = o_all[e, int(kin_obj[e])]
                extra = ""
                if name == "OPEN":
                    oi = 1 - int(kin_obj[e])  # tip-timing probe: is the OTHER can still upright?
                    ax = math_utils.quat_apply(
                        quat_all[e, oi : oi + 1], torch.tensor([[0.0, 1.0, 0.0]], device=device)
                    )[0]
                    extra = f" max_jump {float(carry_jump[e]) * 1000:.1f}mm other_up {float(ax[2]):+.2f}"
                print(
                    f"[kin] {name} env {e}: can rel basket"
                    f" ({float(o[0] - basket_c[0]):+.3f},{float(o[1] - basket_c[1]):+.3f}) z {float(o[2]):.3f}{extra}"
                )

        def kin_up_log(name, ids):
            """Knock-timing probe: both cans' uprightness at every kin stage transition."""
            if len(ids) == 0:
                return
            quat_all = torch.stack([uenv.scene[nm].data.root_quat_w.torch for nm in mdp.OBJECT_NAMES], dim=1)
            yb = torch.tensor([[0.0, 1.0, 0.0]], device=device)
            for e in ids.tolist():
                ups = ",".join(
                    f"{float(math_utils.quat_apply(quat_all[e, j : j + 1], yb)[0, 2]):+.2f}"
                    for j in range(len(mdp.OBJECT_NAMES))
                )
                print(f"[kin] STG {name} t={t} env {e} tgt {int(kin_obj[e])} up {ups}")

        for t in range(args_cli.steps):
            if kin:
                restart = (stage == _RL) & ~mdp.placed_mask(uenv).all(dim=1)
                if restart.any():
                    kin_start(restart.nonzero(as_tuple=False).squeeze(-1))
                lifting = (stage == _S["LIFT"]) | (stage == _K_LIFT)
                moving = live_off_t & ((stage == _S["MOVE"]) | (stage == _S["LOWER"]))
                if lifting.any() or moving.any():
                    obj_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
                    o_held = obj_all[torch.arange(n, device=device), kin_obj]
                    track = lifting | moving
                    carry_off[track] = (o_held[:, :2] - tcp_pos_local()[:, :2])[track]
                    # the hang offset rotates with the wrist during the carry: re-aim live
                    tgt_pos[moving, 0] = basket_c[0] - carry_off[moving, 0]
                    tgt_pos[moving, 1] = basket_c[1] - carry_off[moving, 1]
            scripted = (stage >= 0) & (stage < _DONE_STAGE)

            # -- RL branch. The policy never saw a placed object in training (it never
            # released), so a set placed-flag/in-basket pose freezes it: feed it a
            # PHANTOM obs (placed can replaced by a typical on-table can, flags zeroed).
            # The recorded BC observations remain the TRUE ones.
            if agent is not None:
                agent_obs = obs.clone()
                placed_now = mdp.placed_mask(uenv)
                one_placed = placed_now.any(dim=1) & ~placed_now.all(dim=1)
                if one_placed.any():
                    phantom = torch.tensor(
                        [0.24, -0.02, 0.018, 0.7071, 0.0, 0.0, 0.7071], device=agent_obs.device
                    )
                    agent_obs[one_placed, 23:30] = phantom
                    agent_obs[one_placed, 30:32] = 0.0
                with torch.inference_mode():
                    rl_actions = agent.get_action(agent.obs_to_torch(agent_obs), is_deterministic=True)
                if rl_actions.dim() == 1:
                    rl_actions = rl_actions.unsqueeze(0)
                rl_actions = rl_actions.to(device)
            else:
                rl_actions = torch.zeros(n, 7, device=device)

            # -- scripted branch: interpolate command, position-only IK
            delta = tgt_pos - cmd_pos
            dist = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-9)
            cmd_pos += delta * (dist.clamp(max=_CMD_STEP) / dist)

            jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, jacobi_joint_ids]
            ee_pose_w = robot.data.body_pose_w.torch[:, ee_body_id]
            root_pose = robot.data.root_pose_w.torch
            ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
                root_pose[:, 0:3], root_pose[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
            if kin:
                fine = stage == _K_FINE
                if fine.any():
                    # FINEK: LIVE pocket target -- small IK corrections kill table quantization
                    f_live = math_utils.quat_apply(
                        ee_quat_b, torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3)
                    )
                    obj_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
                    live = obj_all[torch.arange(n, device=device), kin_obj] + _K_POCKET * f_live
                    live[:, 2] += _K_AIM_DZ
                    cmd_pos[fine] = live[fine]
                    tgt_pos[fine] = live[fine]
            offset = torch.tensor([_TCP_OFFSET, 0.0, 0.0], device=device).expand(n, 3)
            ge_tgt = cmd_pos + math_utils.quat_apply(ee_quat_b, offset)
            ik_pos.set_command(ge_tgt, ee_quat=ee_quat_b)
            q_des = ik_pos.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            script_actions = torch.zeros(n, 7, device=device)
            script_actions[:, :6] = (q_des - q_default) / _ARM_SCALE
            script_actions[:, 6] = grip

            # kinematic joint stages: interpolate the joint command toward the table config.
            # Per-stage speed: no-can stages (PREK, UPK) and the slip-safe yaw run faster;
            # loaded arm/wrist moves (LIFTK, CARRYK) slightly faster than the grasp default.
            if kin:
                # LIFTK is the shake-out stage (run 4: 0.025 rad/step lost cans en masse,
                # run 3: 0.02 mostly held) -> lift gently; the yaw is slip-safe at speed.
                dq = torch.full((n, 1), _K_DQ, device=device)
                for s, v in (
                    (_K_PRE, 0.035),
                    (_K_CHECK, 0.01),
                    (_K_DOWN, 0.01),
                    (_K_LIFT, 0.015),
                    (_K_YAW, 0.035),
                    (_K_CARRY, 0.02),
                    (_K_UP, 0.035),
                ):
                    dq = torch.where((stage == s).unsqueeze(1), torch.full_like(dq, v), dq)
                # SLIDEK creep: the last ~2 cm of the slide-in runs at dq 0.01 (plow guard)
                sl = stage == _K_SLIDE
                if sl.any():
                    jerr_s = (robot.data.joint_pos.torch[:, arm_joint_ids] - q_ktgt).abs().max(dim=1).values
                    dq = torch.where((sl & (jerr_s < _K_SLIDE_SLOW)).unsqueeze(1), torch.full_like(dq, 0.01), dq)
                q_cmd += torch.minimum(torch.maximum(q_ktgt - q_cmd, -dq), dq)
                kjoint = (
                    (stage == _K_PRE)
                    | (stage == _K_SLIDE)
                    | (stage == _K_CHECK)
                    | (stage == _K_DOWN)
                    | (stage == _K_BACK)
                    | (stage == _K_LIFT)
                    | (stage == _K_YAW)
                    | (stage == _K_CARRY)
                    | (stage == _K_LOW)
                    | (stage == _K_OPEN)
                    | (stage == _K_UP)
                    | (stage == _DONE_STAGE)  # kin DONE envs hold the last joint command
                )
                kin_actions = torch.zeros(n, 7, device=device)
                kin_actions[:, :6] = (q_cmd - q_default) / _ARM_SCALE
                kin_actions[:, 6] = grip

            # DONE envs hold the last scripted pose; RL envs use the policy
            use_script = scripted | (stage == _DONE_STAGE)
            if kin:
                use_script |= (stage == _K_FINE) | (stage == _K_CLOSE)
            action = torch.where(use_script.unsqueeze(1), script_actions, rl_actions)
            if kin:
                action = torch.where(kjoint.unsqueeze(1), kin_actions, action)

            obs_buf[t] = obs
            act_buf[t] = action
            with torch.inference_mode():
                obs, _, dones, _ = env.step(action)
            if isinstance(obs, dict):
                obs = obs["obs"]
            dropped |= dones.to(device=device, dtype=torch.bool)

            if writer is not None and rounds == 1:
                out = uenv.scene["workspace_cam"].data.output["rgb"]
                out = getattr(out, "torch", out)
                if callable(out):
                    out = out()
                writer.append_data(out[0, ..., :3].detach().cpu().numpy().astype("uint8"))

            # -- bookkeeping
            stage_t += uenv.step_dt
            placed = mdp.placed_mask(uenv)
            both = placed.all(dim=1)
            newly_done = both & (done_step == args_cli.steps)
            done_step[newly_done] = t

            # RL -> scripted handoff: object held+lifted for _HELD_STEPS consecutive steps
            held = held_unplaced_mask()
            first_grasp = torch.where((first_grasp < 0) & held, t, first_grasp)
            first_place = torch.where((first_place < 0) & placed.any(dim=1), t, first_place)
            held_ctr = torch.where((stage == _RL) & held, held_ctr + 1, torch.zeros_like(held_ctr))
            handoff = (stage == _RL) & (held_ctr >= _HELD_STEPS)
            if handoff.any():
                enter_stage(handoff.nonzero(as_tuple=False).squeeze(-1), _S["LIFT"])

            # scripted stage advancement
            tol = torch.tensor([s[1] for s in _STAGES], device=device)[stage.clamp(min=0, max=_DONE_STAGE - 1)]
            timeout = torch.tensor([s[2] for s in _STAGES], device=device)[stage.clamp(min=0, max=_DONE_STAGE - 1)]
            if kin:
                timeout = torch.where(stage == _S["OPEN"], open_t_t, timeout)
            err = torch.linalg.norm(tcp_pos_local() - tgt_pos, dim=1)
            reached = (tol > 0) & (err < tol)
            advance = scripted & (reached | (stage_t > timeout))
            if advance.any():
                ids = advance.nonzero(as_tuple=False).squeeze(-1)
                nxt = stage[ids] + 1
                for k, e in enumerate(ids.tolist()):
                    nk = int(nxt[k])
                    if nk == _S["MOVE"] and not held[e]:
                        stage[e] = _RL  # lost the object during LIFT: regrasp
                        held_ctr[e] = 0
                    elif nk == _DONE_STAGE or (kin and nk == _S["HOME"]):
                        # HOME only exists for the RL policy; kinematic grasps start anywhere
                        if placed[e].all():
                            stage[e] = _DONE_STAGE  # hold position
                        else:
                            stage[e] = _RL  # next object
                            held_ctr[e] = 0
                    else:
                        enter_stage(torch.tensor([e], device=device), nk)

            # kinematic grasp stage advancement
            if kin:
                # carry divergence diagnostic: max per-step TCP displacement
                tcp_now = tcp_pos_local()
                in_carry = (stage == _K_LIFT) | (stage == _K_YAW) | (stage == _K_CARRY) | (stage == _K_LOW)
                carry_jump = torch.where(
                    in_carry, torch.maximum(carry_jump, torch.linalg.norm(tcp_now - tcp_prev, dim=1)), carry_jump
                )
                tcp_prev = tcp_now
                # abort an empty carry immediately: regrasp instead of placing nothing
                carrying = in_carry
                if carrying.any():
                    obj_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
                    o_held = obj_all[torch.arange(n, device=device), kin_obj]
                    rel_all = o_held - tcp_pos_local()
                    # abort on full loss OR on a >5 cm droop: a half-slipped can hangs at
                    # table height and gets swept through the other can if carried on
                    lost = carrying & ((torch.linalg.norm(rel_all, dim=1) > _HELD_DIST + 0.01) | (-rel_all[:, 2] > 0.05))
                    if lost.any():
                        ids_l = lost.nonzero(as_tuple=False).squeeze(-1)
                        rel = rel_all[ids_l]
                        print(
                            f"[kin] t={t}: lost can mid-carry in envs {ids_l.tolist()}"
                            f" bands {kin_band[ids_l].tolist()} stages {stage[ids_l].tolist()}"
                            f" lying {kin_lying[ids_l].long().tolist()}"
                            f" dq {[round(float(x), 3) for x in dq[ids_l, 0]]}"
                            f" rel {[[round(float(v), 3) for v in r] for r in rel]}"
                        )
                        stage[lost] = _RL
                        grip[lost] = _OPEN
                # SLIDE-ABORT: the moment an upright-selected target starts tipping mid-
                # slide, back out to the pre config (joint interp) and retry once with a
                # fresh azimuth correction; on the second tip re-select.
                sliding = (stage == _K_SLIDE) & ~kin_lying
                if sliding.any():
                    quat_all = torch.stack([uenv.scene[nm].data.root_quat_w.torch for nm in mdp.OBJECT_NAMES], dim=1)
                    yb = torch.tensor([[0.0, 1.0, 0.0]], device=device)
                    for e in sliding.nonzero(as_tuple=False).squeeze(-1).tolist():
                        oi = int(kin_obj[e])
                        up = abs(float(math_utils.quat_apply(quat_all[e, oi : oi + 1], yb)[0, 2]))
                        if up < _K_TIP:
                            kin_slide_retry[e] += 1
                            act = "retry" if kin_slide_retry[e] <= 1 else "reselect"
                            print(f"[kin] SLIDE-ABORT env {e} t={t} obj {oi} up {up:+.2f} -> {act}")
                            if kin_slide_retry[e] <= 1:
                                stage[e] = _K_PRE
                                stage_t[e] = 0.0
                                q_ktgt[e] = kin_q_pre[e]
                            else:
                                stage[e] = _RL
                                grip[e] = _OPEN
                                held_ctr[e] = 0
                joint_err = (robot.data.joint_pos.torch[:, arm_joint_ids] - q_ktgt).abs().max(dim=1).values
                j_done = kjoint & ((joint_err < _K_JTOL) | (stage_t > _K_JTIMEOUT))
                pre_done = j_done & (stage == _K_PRE)
                slide_done = j_done & (stage == _K_SLIDE)
                fine_err = torch.linalg.norm(tcp_pos_local() - tgt_pos, dim=1)
                fine_done = (stage == _K_FINE) & ((fine_err < _K_FINE_TOL) | (stage_t > _K_FINE_T))
                close_done = (stage == _K_CLOSE) & (stage_t > _K_CLOSE_T)
                check_done = (stage == _K_CHECK) & ((joint_err < 0.03) | (stage_t > _K_CHECK_T))
                down_done = (stage == _K_DOWN) & ((joint_err < 0.03) | (stage_t > _K_DOWN_T))
                back_done = (stage == _K_BACK) & ((joint_err < _K_JTOL) | (stage_t > _K_BACK_T))
                lift_done = (stage == _K_LIFT) & ((joint_err < _K_JTOL) | (stage_t > _K_LIFT_T))
                yaw_done = (stage == _K_YAW) & ((joint_err < _K_JTOL) | (stage_t > _K_YAW_T))
                carry_done = (stage == _K_CARRY) & ((joint_err < _K_PLACE_TOL) | (stage_t > _K_CARRY_T))
                low_done = (stage == _K_LOW) & ((joint_err < _K_PLACE_TOL) | (stage_t > _K_LOW_T))
                open_done = (stage == _K_OPEN) & (stage_t > open_t_t)
                up_done = (stage == _K_UP) & ((joint_err < _K_JTOL) | (stage_t > _K_UP_T))
                if pre_done.any():
                    # one-shot analytic lateral alignment: joint1 yaws the pocket azimuth
                    # EXACTLY, so add the measured pocket-vs-can azimuth error to joint1 of
                    # the grasp/pre configs and slide in pure joint space (NO diff IK).
                    ids = pre_done.nonzero(as_tuple=False).squeeze(-1)
                    f_a = math_utils.quat_apply(
                        ee_quat_b[ids], torch.tensor([-1.0, 0.0, 0.0], device=device).expand(len(ids), 3)
                    )
                    pk = tcp_pos_local()[ids] - _K_POCKET * f_a
                    obj_all_a = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
                    o_a = obj_all_a[ids, kin_obj[ids]]
                    dj1 = torch.atan2(o_a[:, 1], o_a[:, 0]) - torch.atan2(pk[:, 1], pk[:, 0])
                    dj1 = ((dj1 + math.pi) % (2.0 * math.pi) - math.pi).clamp(-_K_ALIGN_MAX, _K_ALIGN_MAX)
                    kin_q_grasp[ids, 0] += dj1
                    kin_q_pre[ids, 0] += dj1
                    q_ktgt[ids] = kin_q_grasp[ids]
                    stage[ids] = _K_SLIDE
                    stage_t[ids] = 0.0
                    for k, e in enumerate(ids.tolist()):
                        r_e = float(torch.linalg.norm(pk[k, :2]))
                        print(f"[kin] ALIGN1 env {e}: dj1 {float(dj1[k]):+.3f}rad (~{float(dj1[k]) * r_e * 1000:+.0f}mm)")
                    kin_up_log("SLIDEK", ids)
                if slide_done.any():
                    stage[slide_done] = _K_FINE
                    stage_t[slide_done] = 0.0
                    kin_up_log("FINEK", slide_done.nonzero(as_tuple=False).squeeze(-1))
                if fine_done.any():
                    stage[fine_done] = _K_CLOSE
                    stage_t[fine_done] = 0.0
                    grip[fine_done] = _CLOSE
                    kin_up_log("CLOSEK", fine_done.nonzero(as_tuple=False).squeeze(-1))
                if close_done.any():
                    # CHECKK: slow ~2 cm micro-lift to verify the grip before committing to
                    # the carry (weak lying-can grips droop out of the fingers under gravity;
                    # catching it here costs 0.5 s instead of a lost can + knocked scene).
                    ids = close_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _K_CHECK
                    stage_t[ids] = 0.0
                    q_cmd[ids] = robot.data.joint_pos.torch[ids][:, arm_joint_ids]
                    q_c = kin_q_grasp[ids].clone()
                    q_c[:, 1] = (q_c[:, 1] + _K_CHECK_D2).clamp(j2_lo, j2_hi)
                    q_ktgt[ids] = q_c
                    grip[ids] = _CLOSE
                    carry_jump[ids] = 0.0
                    kin_up_log("CHECKK", ids)
                if check_done.any():
                    ids = check_done.nonzero(as_tuple=False).squeeze(-1)
                    obj_all = torch.stack([object_pos_local(nm) for nm in mdp.OBJECT_NAMES], dim=1)
                    o_h = obj_all[torch.arange(n, device=device), kin_obj]
                    tcp_c = tcp_pos_local()
                    rel_c = o_h - tcp_c
                    droop_c = tcp_c[:, 2] - o_h[:, 2]
                    bad = (droop_c > _K_DROOP) | (torch.linalg.norm(rel_c, dim=1) > 0.06)
                    for e in ids.tolist():
                        print(
                            f"[kin] GRIP env {e}: obj {int(kin_obj[e])} lying {int(kin_lying[e])}"
                            f" droop {float(droop_c[e]) * 1000:.0f}mm"
                            f" rel ({float(rel_c[e, 0]):+.3f},{float(rel_c[e, 1]):+.3f},{float(rel_c[e, 2]):+.3f})"
                            f" {'FAIL' if bool(bad[e]) else 'ok'}"
                        )
                    fail = ids[bad[ids]]
                    if len(fail):
                        # LOWER-BEFORE-REOPEN: reverse the micro-lift with the grip still
                        # closed (put the can back on the table), only then open + back
                        # off. Opening at height tip-and-rolled it into its neighbor.
                        stage[fail] = _K_DOWN
                        stage_t[fail] = 0.0
                        q_ktgt[fail] = kin_q_grasp[fail]
                        grip[fail] = _CLOSE
                        held_ctr[fail] = 0
                        kin_up_log("DOWNK", fail)
                    good = ids[~bad[ids]]
                    if len(good):
                        # LIFTK: raise the pocket ~6-9 cm by lifting joint2 (FK-probed).
                        stage[good] = _K_LIFT
                        stage_t[good] = 0.0
                        q_l = kin_q_grasp[good].clone()
                        q_l[:, 1] = (q_l[:, 1] + d2_lift).clamp(j2_lo, j2_hi)
                        q_ktgt[good] = q_l
                        kin_up_log("LIFTK", good)
                if down_done.any():
                    ids = down_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _K_BACK
                    stage_t[ids] = 0.0
                    grip[ids] = _OPEN
                    q_ktgt[ids] = kin_q_pre[ids]
                    kin_up_log("BACKK", ids)
                if back_done.any():
                    ids = back_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _RL  # re-select and regrasp
                    held_ctr[ids] = 0
                    kin_up_log("RESEL", ids)
                # LIFTK done: verify the can came along, then yaw onto the basket azimuth.
                # YAWK moves ONLY joint1 (finger-to-gravity angle invariant -> no slip
                # torque) so the big azimuth sweep happens before any wrist change.
                if lift_done.any():
                    ids = lift_done.nonzero(as_tuple=False).squeeze(-1)
                    ok = held[ids]
                    fail = ids[~ok]
                    stage[fail] = _RL  # lost the can during the lift: regrasp
                    held_ctr[fail] = 0
                    good = ids[ok]
                    if len(good):
                        stage[good] = _K_YAW
                        stage_t[good] = 0.0
                        q_y = q_ktgt[good].clone()  # the lifted grasp config
                        q_y[:, 0] = q_over_w[kin_band[good], 0]
                        q_ktgt[good] = q_y
                        kin_up_log("YAWK", good)
                if yaw_done.any():
                    ids = yaw_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _K_CARRY
                    stage_t[ids] = 0.0
                    q_ktgt[ids] = q_over_w[kin_band[ids]]
                    kin_up_log("CARRYK", ids)
                if carry_done.any():
                    ids = carry_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _K_LOW
                    stage_t[ids] = 0.0
                    q_ktgt[ids] = q_lower_w[kin_band[ids]]
                    kin_up_log("LOWK", ids)
                if low_done.any():
                    ids = low_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _K_OPEN
                    stage_t[ids] = 0.0
                    grip[ids] = _OPEN
                    kin_diag("OPEN", ids)
                if open_done.any():
                    ids = open_done.nonzero(as_tuple=False).squeeze(-1)
                    stage[ids] = _K_UP
                    stage_t[ids] = 0.0
                    q_ktgt[ids] = q_over_w[kin_band[ids]]
                    grip[ids] = _OPEN
                    kin_diag("UP", ids)
                if up_done.any():
                    all_placed = placed.all(dim=1)
                    stage[up_done & all_placed] = _DONE_STAGE
                    rg = up_done & ~all_placed
                    stage[rg] = _RL
                    held_ctr[rg] = 0
                    kin_up_log("NEXT", up_done.nonzero(as_tuple=False).squeeze(-1))

            if t % 50 == 0:
                hist = [int((stage == s).sum()) for s in range(_DONE_STAGE + 1)]
                if kin:
                    khist = [
                        int((stage == s).sum())
                        for s in (_K_PRE, _K_SLIDE, _K_FINE, _K_CLOSE, _K_CHECK, _K_LIFT, _K_YAW, _K_CARRY, _K_LOW, _K_OPEN, _K_UP, _K_DOWN, _K_BACK)
                    ]
                    print(f"[t={t}] kin {khist} | stages {hist} | placed/env {placed.float().sum(1).mean():.2f}")
                else:
                    n_rl = int((stage == _RL).sum())
                    print(f"[t={t}] RL {n_rl} | stages {hist} | placed/env {placed.float().sum(1).mean():.2f}")

        # -- round results
        if args_cli.grasp_test:
            for b in range(min(len(_SWEEP), (n + _BLOCK - 1) // _BLOCK)):
                m = blk == b
                g = int(((first_grasp >= 0) & (first_grasp <= 250) & m).sum())
                p = int(((first_place >= 0) & m).sum())
                gt = first_grasp[m & (first_grasp >= 0)].float()
                mean_g = f"{float(gt.mean()) / 50:.2f}s" if len(gt) else "n/a"
                print(
                    f"[grasp_test] block {b} {_SWEEP[b]}: grasped<=5s {g}/{int(m.sum())}"
                    f" placed<=5.5s {p}/{int(m.sum())} mean_grasp_t {mean_g}"
                )
            break
        placed = mdp.placed_mask(uenv)
        success = placed.all(dim=1) & ~dropped
        attempted += n
        for e in success.nonzero(as_tuple=False).squeeze(-1).tolist():
            end = min(int(done_step[e]) + 25, args_cli.steps)
            saved.append(
                {
                    "obs": obs_buf[:end, e].cpu().to(torch.float32),
                    "action": act_buf[:end, e].cpu().to(torch.float32),
                }
            )
        total_success += int(success.sum())
        print(
            f"[round {rounds}] success {int(success.sum())}/{n}"
            f" (total {total_success}/{args_cli.episodes}, dropped {int(dropped.sum())})"
            f" | grasped {int((first_grasp >= 0).sum())}/{n} first_placed {int((first_place >= 0).sum())}/{n}"
        )
        flush()
        if writer is not None and rounds == 1:
            writer.close()
            writer = None
            print(f"[video] {os.path.join(out_dir, 'expert_env0.mp4')}")

    flush(force=True)
    meta = {
        "task": args_cli.task,
        "rl_checkpoint": resume_path,
        "obs_dim": 39,
        "action_dim": 7,
        "episodes_saved": total_success,
        "episodes_attempted": attempted,
        "expert_success_rate": total_success / max(attempted, 1),
        "steps_per_episode": args_cli.steps,
        "grasp_mode": args_cli.grasp_mode,
        "notes": (
            "kinematic expert: grasp-table state machine grasps, scripted position-IK places"
            if kin
            else "hybrid expert: RL grasps (run-13 policy), scripted position-IK places"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[expert] success rate {meta['expert_success_rate']:.1%} ({total_success}/{attempted})")
    print(f"[data] {total_success} episodes in {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
