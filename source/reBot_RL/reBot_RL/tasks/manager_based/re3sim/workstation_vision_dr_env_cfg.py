# Copyright (c) 2026. Re3Sim workstation effort.
"""The vision task under VISUAL domain randomisation (docs/envs/re3sim/05_VISUAL_DR.md).

A separate task id rather than a flag on ``...Vision-v0``, for the same reason the vision
task is separate from the state task: the shipped Vision env is a *verified configuration*
(the round-1 dataset was collected on it) and this variant changes what every camera sees.
Nothing here touches dynamics — the scene additions are visual-only prims (no colliders, no
rigid bodies) and the events only move cameras, lights and paint, so the scripted expert's
95.3 % / 90.6 %-with-arm-jitter numbers carry over unchanged by construction. That claim is
CHECKED, not assumed: phase-2 gate collects a batch on this task and compares success
against the §0 baseline.

Per-env DR needs per-env prims. The base scene has ONE dome light for the whole stage, so
this cfg adds per env: a key sphere light + an occasional coloured fill (`visual_dr.
randomize_lighting`), a backdrop wall behind the arm, and four visual-only distractor
primitives (`visual_dr.randomize_background`). All spawn dark/hidden-by-first-reset; every
visible property is drawn per reset.

Knobs: the ``RE3SIM_VDR_*`` environment variables — see ``mdp/visual_dr.py``'s module
docstring. ``RE3SIM_VDR=0`` turns every axis off and reproduces the shipped Vision env's
appearance (stock aim, nominal focal, dome-only lighting, no backdrop/distractors).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ..lift.camera_cfg import WRIST_CAM_CFG
from . import mdp
from .workstation_env_cfg import STATION_CAM_EYE, STATION_CAM_TARGET
from .workstation_vision_env_cfg import RebotWorkstationPickPlace1VisionEnvCfg


def _visual_only_material(r, g, b):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=(r, g, b))


@configclass
class RebotWorkstationPickPlace1VisionDREnvCfg(RebotWorkstationPickPlace1VisionEnvCfg):
    """Vision task + camera/lighting/background randomisation, per env, per reset."""

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------------- per-env lights
        # Sphere lights, not distant lights: a distant light is directional over the WHOLE
        # stage, so a per-env one would light all 128 envs at once. A sphere light's
        # inverse-square falloff keeps it local-ish at 2.5 m env spacing; the residual bleed
        # into neighbours is just more lighting variety.
        self.scene.key_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/KeyLight",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, 1.5)),
            spawn=sim_utils.SphereLightCfg(intensity=0.0, radius=0.06, color=(1.0, 1.0, 1.0)),
        )
        self.scene.fill_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FillLight",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.5, 1.2)),
            spawn=sim_utils.SphereLightCfg(intensity=0.0, radius=0.06, color=(1.0, 1.0, 1.0)),
        )

        # ------------------------------------------------------------------- backdrop
        # A wall behind the arm (the station cam looks from +x toward -x, tilted down; the
        # upper third of its frame sees past the arm off the desk). Visual only: no
        # collision props, so PhysX never hears about it.
        # 1.9 m wide: 2.6 m overlapped the coplanar neighbour backdrop by 0.6 m at the
        # Play task's 2.0 m spacing — z-fighting between two per-env colours, recoloured
        # mid-episode by the neighbour's resets (review 2026-08-11, same class as the
        # floor-card overlap).
        self.scene.backdrop = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Backdrop",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.62, 0.0, 0.65)),
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 1.9, 1.8),
                visual_material=_visual_only_material(0.5, 0.5, 0.5),
            ),
        )

        # ------------------------------------------------------------------- floor card
        # Covers Isaac's default black-with-gridlines ground plane, which the station cam
        # sees past every desk edge — a sim-only texture in 100 % of frames otherwise (the
        # first stills grid caught it). 1 m below the desk top, like a real floor; visual
        # only. 4 m square stays inside the 2.5 m env spacing enough that neighbours rarely
        # see each other's card, and where they do it is just more background variety.
        # 1.9 m: no overlap at either env spacing (2.5 collection / 2.0 Play). The 6 m
        # version covered more horizon but overlapped every neighbour, and the z-stagger
        # then meant an env saw the HIGHEST neighbour's card — recoloured mid-episode by
        # that neighbour's resets (review 2026-08-11). The per-reset ground-plane tint
        # (visual_dr.randomize_background) owns the region past the card instead.
        self.scene.floor_card = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FloorCard",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, -0.99)),
            spawn=sim_utils.CuboidCfg(
                size=(1.9, 1.9, 0.02),
                visual_material=_visual_only_material(0.35, 0.35, 0.35),
            ),
        )

        # ------------------------------------------------------------------ distractors
        # Four visual-only primitives; the reset event draws visibility (0–4 shown), pose
        # (rejection-sampled clear of the r=0.55 workspace disc), yaw and colour. Their
        # half-heights are mirrored in visual_dr._DISTRACT_HALF_Z — change both together.
        for name, spawn in (
            ("distractor0", sim_utils.CuboidCfg(
                size=(0.09, 0.06, 0.06), visual_material=_visual_only_material(0.7, 0.3, 0.3))),
            ("distractor1", sim_utils.CylinderCfg(
                radius=0.035, height=0.09, axis="Z",
                visual_material=_visual_only_material(0.3, 0.7, 0.3))),
            ("distractor2", sim_utils.SphereCfg(
                radius=0.025, visual_material=_visual_only_material(0.3, 0.3, 0.7))),
            ("distractor3", sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.07), visual_material=_visual_only_material(0.7, 0.7, 0.3))),
        ):
            idx = int(name[-1])
            setattr(self.scene, name, AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Distractor" + str(idx),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.95, -0.35 + 0.23 * idx, 0.05)),
                spawn=spawn,
            ))

        # --------------------------------------------------------------------- events
        # Replaces the fixed aim (same slot, so event order is preserved); identical pose at
        # zero jitter — scripts/probe_vdr_camera.py asserts it.
        self.events.aim_station_cam = EventTerm(
            func=mdp.randomize_station_camera,
            mode="reset",
            params={"eye": STATION_CAM_EYE, "target": STATION_CAM_TARGET},
        )
        from .workstation_vision_env_cfg import STATION_LENS  # noqa: PLC0415

        self.events.vdr_focal_station = EventTerm(
            func=mdp.randomize_camera_focal, mode="reset",
            params={"sensor_name": "station_cam",
                    "nominal": float(STATION_LENS["focal_length"]), "frac_mult": 1.0},
        )
        # The wrist's intrinsics are a measured device: half the station's jitter.
        self.events.vdr_focal_wrist = EventTerm(
            func=mdp.randomize_camera_focal, mode="reset",
            params={"sensor_name": "wrist_cam",
                    "nominal": float(WRIST_CAM_CFG.spawn.focal_length), "frac_mult": 0.5},
        )
        self.events.vdr_wrist_mount = EventTerm(func=mdp.randomize_wrist_mount, mode="reset")
        self.events.vdr_lighting = EventTerm(func=mdp.randomize_lighting, mode="reset")
        self.events.vdr_background = EventTerm(func=mdp.randomize_background, mode="reset")


@configclass
class RebotWorkstationPickPlace1VisionDREnvCfg_PLAY(RebotWorkstationPickPlace1VisionDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
