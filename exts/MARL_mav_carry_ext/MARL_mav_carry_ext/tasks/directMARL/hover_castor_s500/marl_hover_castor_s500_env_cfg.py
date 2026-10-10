# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The hover task on the rig castor-main simulates and PX4 flies.

Same observations, actions, rewards and setpoint integration as the flycrane hover task. What changes is what is
being simulated, so that a policy trained here meets nothing new in castor-main's PX4 run:

* the airframe is the S500 (castor-main components/simulation/assets/config/s500.yaml);
* the cables have a maximum length and can go slack, like castor-main's `distance` cable;
* RAPTOR is called at 250 Hz with its recurrent state advancing every second call, as PX4's mc_raptor does at
  IMU_GYRO_RATEMAX = 250;
* the policy sees its drone and the payload through noise and a delay, as it does through PX4's estimator and the
  DDS link.
"""

from MARL_mav_carry_ext.assets import FLYCASTOR_S500_CFG
from MARL_mav_carry_ext.tasks.directMARL.hover.marl_hover_env_cfg import EventCfg, MARLHoverEnvCfg

import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class CastorS500EventCfg(EventCfg):
    """Resets as in the flycrane task, with the masses of the real rig varied around their nominal values."""

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["load_link"]),
            "mass_distribution_params": (0.32, 0.48),  # 0.4 kg +- 20 %
            "operation": "abs",
        },
    )

    randomize_drone_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Falcon.*_base_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


@configclass
class MARLHoverCastorS500EnvCfg(MARLHoverEnvCfg):
    # Physics at PX4's gyro rate here, the policy still at 50 Hz.
    decimation = 5
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 250.0, render_interval=decimation, gravity=(0.0, 0.0, -9.8066))
    # RAPTOR on every physics step, its recurrent state advancing every second one: mc_raptor at 250 Hz.
    low_level_decimation: int = 1
    raptor_native_every = 2

    # robot
    robot_cfg: ArticulationCfg = FLYCASTOR_S500_CFG.replace(prim_path="/World/envs/env_.*/flycastor")
    robot_cfg.spawn.activate_contact_sensors = True

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/flycastor/.*", update_period=0.0, history_length=3, debug_vis=False
    )

    # body names
    falcon_names = "Falcon.*_base_link"
    payload_name = "load_link"
    # the two halves of each telescoping cable: at the payload, and at the drone
    rope_name = "rope_.*_lower"
    rope_top_name = "rope_.*_upper"

    # Cables that can go slack. At rest each carries 1.5 N.
    cable_slide_joint_names = "rope_.*_slide"
    cable_slack_tolerance = 0.03
    cable_slack_time = 0.2
    slack_penalty_weight = 1.0
    cable_tension_ref = 0.5

    # An S500 spans 0.72 m across its propeller tips; the drones spawn 3.0 m apart.
    drone_collision_threshold = 1.2
    # With 3 m cables the drones fly up to 4.4 m high for the highest goal.
    bounding_box_threshold = 6.5
    max_body_speed = 25.0

    # S500 propulsion as castor-main models it: 11.5 N per motor at 1100 rad/s, thrust with speed squared.
    max_thrust_pp = 11.5  # N
    motor_thrust_coeff = 11.5 / 1100.0**2
    motor_torque_coeff = 0.0138 * 11.5 / 1100.0**2  # 0.0138 m of yaw torque per newton of thrust
    # castor-main's motors answer instantly; a 2216 motor with a 9.4 in propeller takes 0.03 to 0.05 s.
    motor_time_constant = 0.0
    motor_time_constant_range = (0.0, 0.04)
    # PX4 quad-X: rotors 0 (front-right) and 1 (back-left) turn counter-clockwise, so they twist the body clockwise.
    rotor_directions = (-1.0, -1.0, 1.0, 1.0)
    initial_rotor_speed = 650.0  # hover with a third of the payload: 4.06 N per motor
    thrust_scale_range = (0.9, 1.1)
    linear_drag = (0.50, 0.30, 0.0)

    # The mapping castor-main and PX4 use: throttle u in [0, 1] -> 1000 u + 100 rad/s.
    raptor_omega_min = 100.0
    raptor_omega_max = 1100.0
    # RAPTOR emits [front-right, back-right, back-left, front-left]; the rotors are in PX4 order
    # [front-right, back-left, front-left, back-right].
    raptor_rotor_permutation = (0, 2, 3, 1)
    raptor_yaw_offset_deg = 0.0

    # Setpoints move at 1 m/s at most, whatever the action. That is also RAPTOR's velocity-error clip, so the
    # velocity feedforward never asks RAPTOR for more than it was trained to see.
    setpoint_step_scale = 0.02  # m per high-level tick: 1 m/s at unit action
    setpoint_max_speed = 1.0
    # The drone itself: an episode ends if one flies faster than 2 m/s, and the reward falls from 1.8 m/s up.
    drone_speed_limit = 2.0
    drone_speed_soft_limit = 1.8
    drone_speed_penalty_weight = 1.0

    # What the policy is given, as it arrives on the aircraft: RTK-grade positions through an estimator and a link.
    drone_pos_noise_std = 0.02  # m
    drone_rot_noise_std = 0.01  # rad
    drone_lin_vel_noise_std = 0.05  # m/s
    drone_ang_vel_noise_std = 0.02  # rad/s
    payload_pos_noise_std = 0.015  # m
    payload_rot_noise_std = 0.01  # rad
    obs_delay_steps_range = (1, 3)  # 20 to 60 ms, fixed for an episode

    # rewards: the flycrane's, with setpoint jitter costing twice as much
    action_smoothness_weight = 1.0

    events = CastorS500EventCfg()
