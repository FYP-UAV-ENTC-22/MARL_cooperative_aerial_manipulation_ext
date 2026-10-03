# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 6-MAV flyhex.

Generated from ``flypent`` with ``scripts/gen_flycrane_urdf.py --n 6``; the
cable anchors sit in two rows of three on the payload's top face. Cables are
three rigid segments joined by ball joints, as the paper prescribes for teams
larger than three MAVs.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

script_dir = Path(__file__).parent
usd_path = script_dir / "data/AMR/flyhex/flyhex.usd"

##
# Configuration
##

FLYHEX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Flyhex",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path.resolve()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # the initial cable splay comes from the anchor rpy baked into the URDF,
        # so every joint starts at zero (same convention as flypent/flycart)
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "load_odometry_sensor": ImplicitActuatorCfg(
            joint_names_expr=["load_odometry_sensor_joint"],
            stiffness=None,
            damping=None,
        ),
        "falcon_IMU_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon.*_imu_joint"],
            stiffness=None,
            damping=None,
        ),
        "falcon_odom_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon.*_odometry_sensor_joint"],
            stiffness=None,
            damping=None,
        ),
        "facon_rotor_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon.*_rotor_.*_joint"],
            stiffness=None,
            damping=None,
        ),
        "rope_joints": ImplicitActuatorCfg(
            joint_names_expr=["rope_.*_sphere_joint_.*_joint_.*"],
            stiffness=None,
            damping=0.005,
        ),
    },
)
"""Configuration for the Flyhex (6 MAVs)."""
