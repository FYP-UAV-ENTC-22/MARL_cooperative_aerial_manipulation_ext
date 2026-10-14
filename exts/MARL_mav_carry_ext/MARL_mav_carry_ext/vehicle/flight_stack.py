"""Everything between the simulated bodies and the two controllers, as it is on the aircraft.

    true drone state  -> IMU + RTK -> PX4 estimator -+-> RAPTOR (on the flight controller, every gyro sample)
                                                     +-> DDS link -> the MARL policy (companion computer, 50 Hz)
    true payload pose -> payload estimator -> Wi-Fi link per drone -> the MARL policy
    the policy's setpoint -> DDS link -> RAPTOR
    RAPTOR's motor command -> ESC delay -> rotor

The environment owns the physics, the rewards and the terminations, which stay on the true state. It gives this
class the true state once per physics step and asks it what each controller is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..common import yaw_of
from ..localization import PayloadEstimator, PayloadEstimatorCfg, WifiLink, WifiLinkCfg
from .actuators import EscCfg, EscDelay
from .dds_link import DdsLink, DdsLinkCfg
from .px4_estimator import Px4Estimator, Px4EstimatorCfg


@dataclass
class FlightStackCfg:
    estimator: Px4EstimatorCfg = field(default_factory=Px4EstimatorCfg)
    dds: DdsLinkCfg = field(default_factory=DdsLinkCfg)
    payload: PayloadEstimatorCfg = field(default_factory=PayloadEstimatorCfg)
    wifi: WifiLinkCfg = field(default_factory=WifiLinkCfg)
    esc: EscCfg = field(default_factory=EscCfg)


class FlightStack:
    def __init__(self, cfg: FlightStackCfg, num_envs: int, num_drones: int, physics_dt: float, device):
        self.cfg = cfg
        self.estimator = Px4Estimator(cfg.estimator, num_envs, num_drones, physics_dt, device)
        self.dds = DdsLink(cfg.dds, num_envs, num_drones, physics_dt, device)
        self.payload = PayloadEstimator(cfg.payload, num_envs, physics_dt, device)
        self.wifi = WifiLink(cfg.wifi, num_envs, num_drones, physics_dt, device)
        self.esc = EscDelay(cfg.esc, num_envs, num_drones, physics_dt, device)
        # what the policy last commanded, before the link
        self.commanded = torch.zeros(num_envs, num_drones, DdsLink.SETPOINT_DIM, device=device)

    def reset(self, env_ids, drone_pos, drone_rot, drone_vel, drone_ang_vel_body, payload_pos, payload_rot,
              rotor_speed) -> None:
        """Start a new episode on the given true state: new sensors, new delays, links full of the start state."""
        e = self.estimator
        e.reset(env_ids, drone_pos, drone_rot, drone_vel, drone_ang_vel_body)
        # until the policy's first setpoint, RAPTOR holds where its estimator says it is
        self.commanded[env_ids] = DdsLink.pack_setpoint(e.position, torch.zeros_like(e.velocity), yaw_of(e.attitude))[
            env_ids
        ]
        self.dds.reset(env_ids, self._odometry(), self.commanded)
        self.payload.reset(env_ids, payload_pos, payload_rot)
        self.wifi.reset(env_ids, WifiLink.pack(self.payload.position, self.payload.rotation))
        self.esc.reset(env_ids, rotor_speed)

    def step(self, drone_pos, drone_rot, drone_vel, drone_ang_vel_body, payload_pos, payload_rot) -> None:
        """One physics step: the estimators see the true state, the links carry what they published."""
        self.estimator.step(drone_pos, drone_rot, drone_vel, drone_ang_vel_body)
        self.payload.step(payload_pos, payload_rot)
        self.dds.push(self._odometry(), self.commanded)
        self.wifi.push(WifiLink.pack(self.payload.position, self.payload.rotation))

    def command(self, position, velocity, yaw) -> None:
        """The policy's setpoint, as it leaves the companion computer."""
        self.commanded = DdsLink.pack_setpoint(position, velocity, yaw)

    # --- what each controller is given -----------------------------------------------------------------------
    def raptor_inputs(self):
        """For RAPTOR now: estimated position, attitude, velocity, body rates; setpoint position, velocity, yaw."""
        e = self.estimator
        return (e.position, e.attitude, e.velocity, e.ang_vel_body, *self.dds.read_setpoint())

    def policy_drone_state(self):
        """For the policy now, per drone: position, attitude, velocity, world angular velocity."""
        return self.dds.read_odometry()

    def policy_payload_pose(self):
        """For the policy now, per drone: the payload's position (N, D, 3) and rotation (N, D, 3, 3)."""
        return self.wifi.read()

    def _odometry(self) -> torch.Tensor:
        e = self.estimator
        return DdsLink.pack_odometry(e.odom_position, e.odom_attitude, e.odom_velocity, e.odom_ang_vel)
