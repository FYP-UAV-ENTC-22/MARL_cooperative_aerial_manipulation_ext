"""The uXRCE-DDS link between a drone's flight controller and its companion computer.

Up: PX4 hands its odometry to the link once per estimator update (the topic is limited to 100 Hz), and the policy
reads whatever has arrived when it steps. Down: the policy's setpoint reaches RAPTOR a little later than it was
computed. Both delays are drawn once per episode per drone; the upward one also jitters from sample to sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..common import LagBuffer, draw_uniform, to_steps


@dataclass
class DdsLinkCfg:
    # from the estimate's sample time to the policy having it: PX4, serial link, agent, vehicle_interface
    state_delay_s: tuple[float, float] = (0.008, 0.030)
    state_jitter_s: float = 0.002  # std, per read
    # from the policy's output to RAPTOR using it
    setpoint_delay_s: tuple[float, float] = (0.002, 0.010)


class DdsLink:
    ODOM_DIM = 3 + 9 + 3 + 3  # position, attitude row-major, velocity, world angular velocity
    SETPOINT_DIM = 3 + 3 + 1  # position, velocity, yaw

    def __init__(self, cfg: DdsLinkCfg, num_envs: int, num_drones: int, physics_dt: float, device):
        self.cfg, self.dt, self.device = cfg, physics_dt, device
        self.shape = (num_envs, num_drones)
        longest = cfg.state_delay_s[1] + 4.0 * cfg.state_jitter_s
        self._odom = LagBuffer(longest / physics_dt + 2, num_envs, (num_drones, self.ODOM_DIM), device, per_reader=True)
        self._setpoint = LagBuffer(
            cfg.setpoint_delay_s[1] / physics_dt + 2, num_envs, (num_drones, self.SETPOINT_DIM), device, per_reader=True
        )
        self.state_delay = torch.zeros(self.shape, device=device)  # s
        self.setpoint_lag = torch.zeros(self.shape, dtype=torch.long, device=device)  # steps

    def reset(self, env_ids, odometry: torch.Tensor, setpoint: torch.Tensor) -> None:
        n, d = len(env_ids), self.shape[1]
        self.state_delay[env_ids] = draw_uniform(self.cfg.state_delay_s, (n, d), self.device)
        self.setpoint_lag[env_ids] = to_steps(draw_uniform(self.cfg.setpoint_delay_s, (n, d), self.device), self.dt)
        self._odom.fill(env_ids, odometry[env_ids])
        self._setpoint.fill(env_ids, setpoint[env_ids])

    @staticmethod
    def pack_odometry(position, attitude, velocity, ang_vel) -> torch.Tensor:
        return torch.cat((position, attitude.flatten(-2), velocity, ang_vel), dim=-1)

    @staticmethod
    def pack_setpoint(position, velocity, yaw) -> torch.Tensor:
        return torch.cat((position, velocity, yaw.unsqueeze(-1)), dim=-1)

    def push(self, odometry: torch.Tensor, setpoint: torch.Tensor) -> None:
        """Once per physics step: what PX4 last published, and what the policy last commanded."""
        self._odom.push(odometry)
        self._setpoint.push(setpoint)

    def read_odometry(self):
        """The newest odometry the policy has: position, attitude, velocity, world angular velocity."""
        delay = self.state_delay + torch.randn(self.shape, device=self.device) * self.cfg.state_jitter_s
        o = self._odom.read(to_steps(delay, self.dt))
        return o[..., 0:3], o[..., 3:12].unflatten(-1, (3, 3)), o[..., 12:15], o[..., 15:18]

    def read_setpoint(self):
        """The setpoint RAPTOR has at this instant: position, velocity, yaw."""
        s = self._setpoint.read(self.setpoint_lag)
        return s[..., 0:3], s[..., 3:6], s[..., 6]
