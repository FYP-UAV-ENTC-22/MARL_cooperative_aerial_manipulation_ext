# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The S500 hover task as the aircraft flies it: neither controller is given the true state.

The rig, rewards, terminations and observation layout are the S500 task's (hover_castor_s500). What changes:

* Rates are the flight controller's: physics and gyro at 400 Hz (IMU_GYRO_RATEMAX on the aircraft), RAPTOR called on
  every gyro sample with its recurrent state advancing every fourth call, its native 100 Hz. The policy stays at 50 Hz.
* The drone's position is the body origin, the point PX4 estimates and RAPTOR flies, for the policy and RAPTOR both.
* RAPTOR is given PX4's estimate, not the true state: body rates through the 40 Hz gyro filter, attitude at 200 Hz,
  position and velocity at 100 Hz, from an IMU and an RTK receiver at 5 Hz (vehicle/px4_estimator.py).
* The policy is given that same estimate as it arrives over the DDS link, and the payload's pose from the payload's
  own estimator (RTK and a compass) over a Wi-Fi link per drone. These replace the white noise and the whole-frame
  delay of the S500 task.
* The policy's setpoint reaches RAPTOR over the link, and RAPTOR's motor command reaches the rotor through an ESC.

Every level of noise and every delay is drawn per episode from a range: see the config classes named in
vehicle/flight_stack.py. Ranges not taken from a data sheet are placeholders until a log of the real hardware
replaces them.
"""

from MARL_mav_carry_ext.tasks.directMARL.hover_castor_s500.marl_hover_castor_s500_env_cfg import (
    MARLHoverCastorS500EnvCfg,
)
from MARL_mav_carry_ext.vehicle import FlightStackCfg

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class MARLHoverCastorS500Px4EnvCfg(MARLHoverCastorS500EnvCfg):
    # Physics at the aircraft's gyro rate, the policy at 50 Hz.
    decimation = 8
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 400.0, render_interval=decimation, gravity=(0.0, 0.0, -9.8066))
    # RAPTOR on every physics step, its recurrent state advancing every fourth one: mc_raptor at 400 Hz.
    low_level_decimation: int = 1
    raptor_native_every = 4

    drone_point = "origin"

    hold_still_weight = 0.5
    flight_stack: FlightStackCfg | None = FlightStackCfg()

    # The flight stack makes the errors and delays the policy sees; the S500 task's stand-ins are off.
    drone_pos_noise_std = 0.0
    drone_rot_noise_std = 0.0
    drone_lin_vel_noise_std = 0.0
    drone_ang_vel_noise_std = 0.0
    payload_pos_noise_std = 0.0
    payload_rot_noise_std = 0.0
    obs_delay_steps_range = (0, 0)
