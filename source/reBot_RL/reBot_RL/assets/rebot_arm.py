# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RS-rebot-dev-arm (8 DOF: 6 revolute + 2-finger prismatic gripper).

Asset: ``data/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda`` (re-export 2026-07-17b).
Drive gains, joint limits and the ``physics`` variant come from the USD itself and are
validated in ``data/RS-rebot-dev-arm/VALIDATION.md`` -- the actuator entries below mirror
those values so nothing is silently retuned when Isaac Lab writes the drives.

The asset's nested rigid-body chain requires Isaac Sim 6.0+ (its physics parser supports
nested rigid bodies); on Isaac Sim 5.1 PhysX drops 8 of the 10 joints and the articulation
comes up with 0 DOFs.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "RS-rebot-dev-arm"))

REBOT_ARM_USD_PATH = os.path.join(_DATA_DIR, "00-arm-rs_asm-v3.usda")
"""Absolute path to the RS-rebot-dev-arm USD asset."""


REBOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=REBOT_ARM_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # asset ships self-collision off: the shipped collision decomposition
            # interpenetrates at rest (see ISAAC_SIM_CODE.md gotchas)
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": -math.pi / 2,  # USD state:angular:physics:position = -90 deg
            "joint3": math.radians(-1.0),
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint_left": 0.02,
            "joint_right": 0.02,
        },
    ),
    # Gains, damping and effort limits are left as ``None`` on purpose: Isaac Lab then keeps
    # whatever the USD parser produced instead of overwriting it. Do NOT copy the raw USD
    # numbers (500/1500/... ) in here -- UsdPhysics authors *angular* drive gains per degree,
    # while Isaac Lab's actuator configs are per radian, so hardcoding them de-scales every
    # revolute gain by 57.3x, the drives saturate against their effort limits and the arm
    # sags ~1 rad at the elbow.
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""RS-rebot-dev-arm with the USD's validated drive gains."""
