# Copyright (c) 2026. Re3Sim workstation effort.
"""Visual domain randomisation for the vision student (docs/envs/re3sim/05_VISUAL_DR.md).

Everything in this module is APPEARANCE ONLY. Nothing here may touch a collider, a mass, a
spawn position or anything else the MDP can feel: the scripted expert is state-based, and the
whole design rests on visual DR being *free* — every perturbed episode is still a ~95 %
expert demonstration. The one physics-adjacent thing (arm-start jitter) deliberately lives
elsewhere (`events.randomize_arm_start`) and is priced separately.

Sampling contract: **per env, per reset, held for the episode.** A bumped camera stays
bumped; a dim room stays dim. That is what a real deployment looks like — the rig is wrong
in one fixed way per session, not vibrating between frames.

Every axis is gated and sized by environment variables (defaults are the plan of record,
approved 2026-08-09) so the robustness matrix can sweep one axis at a time without touching
code:

    RE3SIM_VDR=0                 master off switch: stock aim, nominal focal, lights zeroed,
                                 backdrop/distractors hidden — the shipped Vision env back
    RE3SIM_VDR_SCALE=1.0         master magnitude multiplier (the matrix's sweep knob).
                                 Continuous magnitudes scale linearly; the background
                                 axis scales its PROBABILITIES (desk swap, distractors)
                                 and lerps colours toward neutral below 1, saturating
                                 above 1 — a colour cannot get more random than uniform
    RE3SIM_VDR_CAM/_FOCAL/_WRIST/_LIGHT/_BG=0   per-group off switches
    RE3SIM_VDR_CAM_POS=0.03      station cam position jitter, ± per axis [m]
    RE3SIM_VDR_CAM_AIM=0.025     station cam look-at target jitter, ± per axis [m]
    RE3SIM_VDR_CAM_ROLL=4.0      station cam roll, ± [deg]
    RE3SIM_VDR_BUMP_P=0.15       probability of the "bumped rig" episode
    RE3SIM_VDR_BUMP_POS=0.08     bumped-rig position jitter, ± [m]
    RE3SIM_VDR_BUMP_ROLL=10.0    bumped-rig roll, ± [deg]
    RE3SIM_VDR_FOCAL_FRAC=0.08   station focal length jitter, ± fraction (wrist gets half)
    RE3SIM_VDR_WRIST_POS=0.005   wrist mount translation jitter, ± per axis [m]
    RE3SIM_VDR_WRIST_ROT=2.0     wrist mount rotation jitter, ± [deg]
    RE3SIM_VDR_SPLATS_OFF_P=0.15 probability the gaussian desk is hidden for the episode
    RE3SIM_VDR_FILL_P=0.25       probability of a coloured fill light

USD-write hygiene: prim/attr handles are found once and cached on the env
(``env._vdr_cache``). Anything not found is reported LOUDLY once — a silent no-op here is
invisible to every automated check this env has (the images still look like images), which
is exactly how the splat-alignment bug shipped.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


_WARNED_FLAGS: set = set()


def _flag(name: str) -> bool:
    """Robust boolean env parse. `RE3SIM_VDR=true` silently disabling ALL DR would be the
    exact silent-no-op this module exists to prevent (review finding 2026-08-11), so
    unrecognised values are treated as ON and shouted about, not swallowed."""
    v = os.environ.get(name, "1").strip().lower()
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("0", "false", "off", "no", ""):
        return False
    if name not in _WARNED_FLAGS:
        print(f"[visual_dr] WARNING: {name}={os.environ.get(name)!r} is not a recognised "
              f"boolean — treating as ON", flush=True)
        _WARNED_FLAGS.add(name)
    return True


def _on(name: str) -> bool:
    """Group gate: on unless the master or the group switch says otherwise."""
    return _flag("RE3SIM_VDR") and _flag(name)


def _scale() -> float:
    return _f("RE3SIM_VDR_SCALE", 1.0)


def _ids(env, env_ids):
    return torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids


def _sensor(env, name: str):
    """The scene entity, or None if this task variant does not carry it."""
    try:
        return env.scene[name]
    except KeyError:
        return None


def _u(n: int, k: int, device) -> torch.Tensor:
    """Uniform in [-1, 1], shape (n, k)."""
    return torch.rand(n, k, device=device) * 2.0 - 1.0


# --------------------------------------------------------------------------- USD helpers


def _cache(env) -> dict:
    c = getattr(env, "_vdr_cache", None)
    if c is None:
        c = {}
        env._vdr_cache = c
    return c


def _stage():
    import omni.usd  # noqa: PLC0415

    return omni.usd.get_context().get_stage()


def _env_prim_path(env, sensor_or_cfg_path: str, i: int) -> str:
    """Resolve a per-env path from an already-resolved cfg path (HANDOFF §4.7)."""
    return sensor_or_cfg_path.replace("env_.*", f"env_{i}")


def _warn_once(env, key: str, msg: str):
    c = _cache(env)
    if not c.get(f"warned:{key}"):
        print(f"[visual_dr] WARNING: {msg}", flush=True)
        c[f"warned:{key}"] = True


def _find_shader_attr(stage, root_path: str, attr: str = "inputs:diffuseColor"):
    """The first shader attribute under ``root_path``, or None. Cached by the caller."""
    from pxr import Usd  # noqa: PLC0415

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        a = prim.GetAttribute(attr)
        if a and a.IsValid() and a.HasValue():
            return a
    return None


def _kelvin_to_rgb(t: float) -> tuple[float, float, float]:
    """Tanner-Helland approximation, normalised to [0, 1]. Good enough for DR."""
    t = t / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    clamp = lambda v: max(0.0, min(255.0, v)) / 255.0  # noqa: E731
    return clamp(r), clamp(g), clamp(b)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


# ------------------------------------------------------------------- station camera pose


def randomize_station_camera(
    env,
    env_ids: torch.Tensor | None,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    sensor_name: str = "station_cam",
):
    """Aim the workstation camera with per-env pose jitter and roll.

    Drop-in replacement for ``events.aim_station_camera`` (same signature, same stock pose
    at zero jitter). Roll cannot go through ``set_world_poses_from_view`` — it has no up or
    roll control — so the look-at rotation is built with the same helper that method uses
    (``create_rotation_matrix_from_view``) and post-multiplied by a rotation about the
    camera's local optical axis. ``scripts/probe_vdr_camera.py`` asserts bit-equality with
    the stock path at zero DR; do not edit one without re-running it.
    """
    import isaaclab.utils.math as math_utils  # noqa: PLC0415

    cam = env.scene[sensor_name]
    ids = _ids(env, env_ids)
    n, dev = len(ids), env.device
    org = env.scene.env_origins[ids]
    eyes = org + torch.tensor(eye, device=dev)
    targets = org + torch.tensor(target, device=dev)

    if not _on("RE3SIM_VDR_CAM"):
        cam.set_world_poses_from_view(eyes, targets, env_ids=ids)
        return

    s = _scale()
    pos_j = _f("RE3SIM_VDR_CAM_POS", 0.03) * s
    aim_j = _f("RE3SIM_VDR_CAM_AIM", 0.025) * s
    roll_j = math.radians(_f("RE3SIM_VDR_CAM_ROLL", 4.0) * s)
    bump = torch.rand(n, device=dev) < _f("RE3SIM_VDR_BUMP_P", 0.15)
    b_pos = _f("RE3SIM_VDR_BUMP_POS", 0.08) * s
    b_roll = math.radians(_f("RE3SIM_VDR_BUMP_ROLL", 10.0) * s)

    pj = torch.where(bump, torch.full_like(bump, b_pos, dtype=torch.float32),
                     torch.full_like(bump, pos_j, dtype=torch.float32))
    rj = torch.where(bump, torch.full_like(bump, b_roll, dtype=torch.float32),
                     torch.full_like(bump, roll_j, dtype=torch.float32))
    eyes = eyes + _u(n, 3, dev) * pj.unsqueeze(1)
    targets = targets + _u(n, 3, dev) * aim_j
    roll = _u(n, 1, dev).squeeze(1) * rj

    rot = math_utils.create_rotation_matrix_from_view(eyes, targets, "Z", device=dev)
    c, si = torch.cos(roll), torch.sin(roll)
    rz = torch.zeros(n, 3, 3, device=dev)
    rz[:, 0, 0], rz[:, 0, 1] = c, -si
    rz[:, 1, 0], rz[:, 1, 1] = si, c
    rz[:, 2, 2] = 1.0
    quat = math_utils.quat_from_matrix(torch.bmm(rot, rz))
    cam.set_world_poses(eyes, quat, env_ids=ids, convention="opengl")


# ----------------------------------------------------------------------- focal length


def randomize_camera_focal(
    env,
    env_ids: torch.Tensor | None,
    sensor_name: str = "station_cam",
    nominal: float = 17.0,
    frac_mult: float = 1.0,
):
    """Per-env focal-length jitter, written to the USD camera prim (RTX reads it live).

    Always writes — at DR-off it writes the *nominal* back, so a run after a DR run is not
    silently inheriting the last random draw. ``frac_mult`` scales the env-var fraction per
    sensor (the wrist's intrinsics are measured, so it gets half the station's jitter).
    """
    sensor = _sensor(env, sensor_name)
    if sensor is None:
        return
    f = _f("RE3SIM_VDR_FOCAL_FRAC", 0.08) * _scale() * frac_mult
    if not _on("RE3SIM_VDR_FOCAL"):
        f = 0.0
    stage = _stage()
    c = _cache(env)
    tmpl = sensor.cfg.prim_path
    ids = _ids(env, env_ids)
    draws = 1.0 + (torch.rand(len(ids)) * 2.0 - 1.0).numpy() * f
    for j, i in enumerate(ids.tolist()):
        key = f"focal:{sensor_name}:{i}"
        attr = c.get(key)
        if attr is None:
            prim = stage.GetPrimAtPath(_env_prim_path(env, tmpl, i))
            if not prim.IsValid():
                # continue, not return: abandoning the loop would also abandon the
                # NOMINAL RESTORE for every env after this one (review 2026-08-11)
                _warn_once(env, f"focal:{sensor_name}",
                           f"no camera prim at {_env_prim_path(env, tmpl, i)} — focal DR "
                           "dead for that env")
                c[key] = False
                continue
            attr = prim.GetAttribute("focalLength")
            c[key] = attr
        if attr is False:
            continue
        attr.Set(float(nominal * draws[j]))


# ----------------------------------------------------------------------- wrist mount


def randomize_wrist_mount(
    env,
    env_ids: torch.Tensor | None,
    sensor_name: str = "wrist_cam",
):
    """Perturb the wrist camera's LOCAL mount transform per env: ±5 mm, ±2°.

    The nominal mount (measured, feed-calibrated — lift/camera_cfg.py) is captured from the
    prim on first call and every draw perturbs *that*, so repeated resets do not random-walk
    the mount away from its bracket.
    """
    sensor = _sensor(env, sensor_name)
    if sensor is None:
        return
    from pxr import Gf  # noqa: PLC0415

    s = _scale()
    dpos = _f("RE3SIM_VDR_WRIST_POS", 0.005) * s
    drot = math.radians(_f("RE3SIM_VDR_WRIST_ROT", 2.0) * s)
    if not _on("RE3SIM_VDR_WRIST"):
        dpos = drot = 0.0
    stage = _stage()
    c = _cache(env)
    tmpl = sensor.cfg.prim_path
    for i in _ids(env, env_ids).tolist():
        key = f"wrist:{i}"
        ent = c.get(key)
        if ent is None:
            prim = stage.GetPrimAtPath(_env_prim_path(env, tmpl, i))
            t_attr = prim.GetAttribute("xformOp:translate") if prim.IsValid() else None
            o_attr = prim.GetAttribute("xformOp:orient") if prim.IsValid() else None
            if not (t_attr and t_attr.IsValid() and o_attr and o_attr.IsValid()
                    and t_attr.HasValue() and o_attr.HasValue()):
                # continue, not return — see randomize_camera_focal
                _warn_once(env, "wrist", f"no usable wrist cam prim at "
                           f"{_env_prim_path(env, tmpl, i)} — wrist-mount DR dead for that env")
                c[key] = False
                continue
            ent = (t_attr, o_attr, Gf.Vec3d(t_attr.Get()), o_attr.Get())
            c[key] = ent
        if ent is False:
            continue
        t_attr, o_attr, t0, q0 = ent
        d = torch.rand(3).numpy() * 2.0 - 1.0
        t_attr.Set(Gf.Vec3d(t0[0] + float(d[0]) * dpos,
                            t0[1] + float(d[1]) * dpos,
                            t0[2] + float(d[2]) * dpos))
        ax = torch.randn(3)
        ax = (ax / (ax.norm() + 1e-9)).numpy()
        ang = float(torch.rand(1)) * 2.0 - 1.0
        half = ang * drot / 2.0
        sw = math.cos(half)
        sv = math.sin(half)
        dq = type(q0)(sw, float(ax[0]) * sv, float(ax[1]) * sv, float(ax[2]) * sv)
        o_attr.Set(dq * q0)


# --------------------------------------------------------------------------- lighting


def randomize_lighting(env, env_ids: torch.Tensor | None):
    """Dome (global, per reset) + per-env key light + occasional coloured fill.

    The dome is ONE prim for the whole stage, so its draw is shared by every env in the
    batch — global exposure/temperature shifts vary across resets, not within one. The
    per-env variety comes from the key/fill sphere lights the DR scene adds.
    """
    stage = _stage()
    c = _cache(env)
    on = _on("RE3SIM_VDR_LIGHT")
    s = _scale()

    # dome ------------------------------------------------------------------
    # ONE prim for the whole stage. Redrawn when env 0 resets: full-batch-only froze the
    # dome at its first draw for entire staggered-reset runs (review 2026-08-11), while
    # every-reset redraws yank the global exposure mid-episode for every other env on
    # every termination. Tying it to env 0's episode boundary bounds both failure modes:
    # other envs see at most one global shift per env-0 episode (a cloud passing), and
    # the dome keeps sampling forever.
    ids_t = _ids(env, env_ids)
    full_batch = bool((ids_t == 0).any())
    dome = c.get("dome")
    if dome is None:
        prim = stage.GetPrimAtPath("/World/light")
        if not prim.IsValid():
            _warn_once(env, "dome", "no /World/light dome prim — dome DR dead")
            dome = False
        else:
            dome = (prim.GetAttribute("inputs:intensity"), prim.GetAttribute("inputs:color"))
        c["dome"] = dome
    if dome and full_batch:
        if on:
            lo, hi = 0.3 ** s, 3.0 ** s  # ×[0.3, 3] at scale 1, log-uniform
            mult = math.exp(float(torch.rand(1)) * (math.log(hi) - math.log(lo)) + math.log(lo))
            temp = 2700.0 + float(torch.rand(1)) * (6500.0 - 2700.0)
            r, g, b = _kelvin_to_rgb(temp)
            tint = 1.0 + (torch.rand(3).numpy() * 2.0 - 1.0) * 0.08 * s
            col = (min(1.0, 0.75 * r * float(tint[0])), min(1.0, 0.75 * g * float(tint[1])),
                   min(1.0, 0.75 * b * float(tint[2])))
        else:
            mult, col = 1.0, (0.75, 0.75, 0.75)
        dome[0].Set(3000.0 * mult)
        dome[1].Set(type(dome[1].Get())(*col))

    # per-env key + fill ----------------------------------------------------
    fill_p = _f("RE3SIM_VDR_FILL_P", 0.25)
    for i in _ids(env, env_ids).tolist():
        for name, is_fill in (("KeyLight", False), ("FillLight", True)):
            key = f"light:{name}:{i}"
            ent = c.get(key)
            if ent is None:
                path = f"/World/envs/env_{i}/{name}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    _warn_once(env, f"light:{name}", f"no prim at {path} — {name} DR dead "
                               "(is this the -VisionDR task?)")
                    ent = False
                else:
                    ent = (prim.GetAttribute("inputs:intensity"),
                           prim.GetAttribute("inputs:color"),
                           prim.GetAttribute("xformOp:translate"))
                    if not all(a and a.IsValid() and a.HasValue() for a in ent):
                        _warn_once(env, f"light:{name}",
                                   f"{path} lacks intensity/color/translate attrs — {name} DR dead")
                        ent = False
                c[key] = ent
            if not ent:
                continue
            inten_a, col_a, tr_a = ent
            if not on:
                inten_a.Set(0.0)
                continue
            if is_fill and float(torch.rand(1)) > fill_p:
                inten_a.Set(0.0)
                continue
            # position: hemisphere around the workspace centre (0.3, 0, 0), in the ENV'S
            # OWN frame. ⭐ No env-origin addition: `xformOp:translate` on a child of
            # /World/envs/env_i is LOCAL, and the cloner already put the origin on the
            # parent — adding it again shipped every light to ~2× its origin, i.e. over a
            # NEIGHBOUR'S env (review 2026-08-11; on a symmetric grid 2×origin lands on
            # another cell, which is why the stills grids looked plausibly lit). Radius
            # capped at 1.4 m to bound cross-env bleed at 2.0 m Play spacing.
            az = float(torch.rand(1)) * 2.0 * math.pi
            el = math.radians(20.0 + float(torch.rand(1)) * 60.0)
            rad = 1.0 + float(torch.rand(1)) * 0.4
            from pxr import Gf  # noqa: PLC0415

            tr_a.Set(Gf.Vec3d(0.3 + rad * math.cos(el) * math.cos(az),
                              rad * math.cos(el) * math.sin(az),
                              rad * math.sin(el)))
            if is_fill:
                r, g, b = _hsv_to_rgb(float(torch.rand(1)), 0.8 + 0.2 * float(torch.rand(1)), 1.0)
                lo_i, hi_i = 2.0e3, 3.0e4
            else:
                r, g, b = _kelvin_to_rgb(2500.0 + float(torch.rand(1)) * 5000.0)
                lo_i, hi_i = 5.0e3, 1.0e5
            inten_a.Set(math.exp(float(torch.rand(1))
                                 * (math.log(hi_i) - math.log(lo_i)) + math.log(lo_i)) * s)
            col_a.Set(type(col_a.Get())(r, g, b))


# ------------------------------------------------------------------------- background


#: Distractors keep this clear of the arm base: object spawns reach r=0.28 (fallback
#: lattice) and the box reaches ~0.43; 0.55 leaves margin. Outer bounds are the VISIBLE
#: gaussian desk, not the analytic slab — the first stills grid (05_VISUAL_DR.md §4) caught
#: a sphere hovering over the void past the splat crop's edge at x ≈ 0.9.
_DISTRACT_R_MIN = 0.55
_DISTRACT_X = (-0.30, 0.85)
_DISTRACT_Y = (-0.42, 0.42)
#: authored half-heights of the four distractor prims, for a plausible rest pose
_DISTRACT_HALF_Z = (0.030, 0.045, 0.025, 0.035)


def _surface_color() -> tuple[float, float, float]:
    """60 % muted (walls, floors), 40 % saturated. Value capped at 0.75 — the first stills
    grid drew near-white so often the wrist backdrop read as a blown-out sky.

    Colour spread lerps toward neutral grey below RE3SIM_VDR_SCALE=1 so the BG axis
    responds to the sweep knob like every other axis (review 2026-08-11); above 1 it
    saturates — a colour cannot get more random than uniform."""
    if float(torch.rand(1)) < 0.6:
        col = _hsv_to_rgb(float(torch.rand(1)), 0.15 * float(torch.rand(1)),
                          0.05 + float(torch.rand(1)) * 0.70)
    else:
        col = _hsv_to_rgb(float(torch.rand(1)), 0.5 + 0.5 * float(torch.rand(1)),
                          0.25 + float(torch.rand(1)) * 0.50)
    t = min(1.0, _scale())
    return tuple(0.4 + (v - 0.4) * t for v in col)


def randomize_background(env, env_ids: torch.Tensor | None):
    """Backdrop + floor-card colour, splat-desk visibility, and 0–4 visual-only distractors.

    The floor card exists because of the first stills grid: the station camera sees past the
    desk to Isaac's default ground plane, whose black-with-white-gridlines material is a
    SIM-ONLY texture in every single frame — a constant cue for the student to anchor on
    that no real deployment reproduces. The card covers it with a per-episode colour."""
    from pxr import Gf, UsdGeom  # noqa: PLC0415

    stage = _stage()
    c = _cache(env)
    on = _on("RE3SIM_VDR_BG")
    # probabilities scale with the sweep knob (capped): at SCALE 0.5 the desk swaps and
    # distractors appear half as often; at 0 the BG axis converges to nominal like the rest
    s_p = min(1.0, _scale())
    splat_off_p = min(0.9, _f("RE3SIM_VDR_SPLATS_OFF_P", 0.15) * s_p)
    distract_p = 0.5 * s_p

    # global ground plane tint — one prim, so full-batch resets only, like the dome. Kills
    # the black-with-gridlines default at the horizon, where no finite floor card reaches
    # (stills grid round 2: visible past the card edge at grazing wrist angles).
    if env_ids is None or len(env_ids) == env.num_envs:
        gp = c.get("groundplane")
        if gp is None:
            attr = (_find_shader_attr(stage, "/World/GroundPlane")
                    or _find_shader_attr(stage, "/World/GroundPlane", "inputs:diffuse_tint"))
            if attr is None:
                _warn_once(env, "groundplane",
                           "no tintable shader under /World/GroundPlane — horizon stays grid")
                gp = False
            else:
                # keep the AUTHORED colour so DR-off restores the stock look exactly — a
                # repainted "nominal" would be a third visual protocol, not the -Vision one
                gp = (attr, attr.Get())
            c["groundplane"] = gp
        if gp:
            attr, orig = gp
            attr.Set(type(orig)(*_surface_color()) if on else orig)

    for i in _ids(env, env_ids).tolist():
        # backdrop + floor card ---------------------------------------------
        for leaf in ("Backdrop", "FloorCard"):
            key = f"bg:{leaf}:{i}"
            ent = c.get(key)
            if ent is None:
                path = f"/World/envs/env_{i}/{leaf}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    _warn_once(env, leaf, f"no prim at {path} — {leaf} DR dead "
                               "(is this the -VisionDR task?)")
                    ent = False
                else:
                    ent = (UsdGeom.Imageable(prim), _find_shader_attr(stage, path))
                    if ent[1] is None:
                        _warn_once(env, f"{leaf}-shader",
                                   f"no diffuseColor shader under {path} — {leaf} stays grey")
                c[key] = ent
            if ent:
                img, shader = ent
                if not on:
                    img.MakeInvisible()
                else:
                    img.MakeVisible()
                    if shader is not None:
                        shader.Set(Gf.Vec3f(*_surface_color()))

        # splat desk <-> plain-slab desk -----------------------------------
        # "Splats off" alone left the cube and box FLOATING — the physics slab is authored
        # invisible under the splats, so hiding them deleted the desk from the image while
        # the objects still rested on the invisible collider (stills grid round 2). A real
        # deployment always has a table; what varies is what the table LOOKS like. So the
        # draw swaps the gaussian desk for the analytic slab with a random tint: a plain,
        # differently-coloured table instead of a hole in the world.
        key = f"bg:splats:{i}"
        ent = c.get(key)
        if ent is None:
            sp = stage.GetPrimAtPath(f"/World/envs/env_{i}/Splats")
            tp = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
            if not sp.IsValid() or not tp.IsValid():
                _warn_once(env, "splats-vis", f"missing Splats or Table prim for env {i} — "
                           "desk-swap DR dead (RE3SIM_SPLATS_PER_ENV=1?)")
                ent = False
            else:
                ent = (UsdGeom.Imageable(sp), UsdGeom.Imageable(tp),
                       _find_shader_attr(stage, f"/World/envs/env_{i}/Table"))
            c[key] = ent
        if ent:
            sp_img, tab_img, tab_shader = ent
            if on and float(torch.rand(1)) < splat_off_p:
                sp_img.MakeInvisible()
                tab_img.MakeVisible()
                if tab_shader is not None:
                    tab_shader.Set(Gf.Vec3f(*_surface_color()))
            else:
                sp_img.MakeVisible()
                tab_img.MakeInvisible()

        # distractors -------------------------------------------------------
        for d in range(4):
            key = f"bg:distract:{i}:{d}"
            ent = c.get(key)
            if ent is None:
                path = f"/World/envs/env_{i}/Distractor{d}"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    if d == 0:
                        _warn_once(env, "distract", f"no prim at {path} — distractor DR dead "
                                   "(is this the -VisionDR task?)")
                    ent = False
                else:
                    ent = (UsdGeom.Imageable(prim), prim.GetAttribute("xformOp:translate"),
                           prim.GetAttribute("xformOp:orient"), _find_shader_attr(stage, path))
                c[key] = ent
            if not ent:
                continue
            img, tr_a, o_a, shader = ent
            if not on or float(torch.rand(1)) > distract_p:
                img.MakeInvisible()
                continue
            img.MakeVisible()
            for _ in range(8):  # rejection-sample clear of the workspace
                x = _DISTRACT_X[0] + float(torch.rand(1)) * (_DISTRACT_X[1] - _DISTRACT_X[0])
                y = _DISTRACT_Y[0] + float(torch.rand(1)) * (_DISTRACT_Y[1] - _DISTRACT_Y[0])
                if math.hypot(x, y) >= _DISTRACT_R_MIN:
                    break
            else:
                x, y = _DISTRACT_R_MIN + 0.1, _DISTRACT_Y[1]
            # env-LOCAL translate — same double-offset trap as the lights (review 2026-08-11)
            tr_a.Set(Gf.Vec3d(x, y, _DISTRACT_HALF_Z[d]))
            if o_a and o_a.IsValid() and o_a.HasValue():
                yaw = float(torch.rand(1)) * math.pi
                o_a.Set(type(o_a.Get())(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
            if shader is not None:
                shader.Set(Gf.Vec3f(*_hsv_to_rgb(float(torch.rand(1)),
                                                 0.6 + 0.4 * float(torch.rand(1)),
                                                 0.4 + 0.6 * float(torch.rand(1)))))
