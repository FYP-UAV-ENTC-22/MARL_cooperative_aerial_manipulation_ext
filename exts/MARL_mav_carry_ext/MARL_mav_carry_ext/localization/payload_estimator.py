"""The payload's own pose estimate: a small flight controller on the payload (ArduPilot) fusing an RTK receiver
(Emlid Reach M+, single band) and a separate compass.

Position is as good as the RTK fix. Roll and pitch come from the IMU and gravity. Heading comes from the compass,
whose error is mostly a steady offset: the payload then settles that far from the commanded yaw, and the policy has
to cope with a heading that is slightly wrong all flight. Errors drift (they are a filter's output), so each is a
first-order Gauss-Markov process around a per-episode offset, and the result is published at the telemetry rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..common import draw_gaussian, draw_uniform, rotvec_to_matrix


@dataclass
class PayloadEstimatorCfg:
    rate_hz: float = 50.0  # the telemetry stream the drones receive
    # RTK fixed, Reach M+: 7 mm + 1 ppm horizontal, 14 mm + 2 ppm vertical, widened
    pos_noise_std_xy: tuple[float, float] = (0.007, 0.02)  # m
    pos_noise_std_z: tuple[float, float] = (0.014, 0.04)  # m
    pos_bias_std_xy: float = 0.01  # m, steady for the flight
    pos_bias_std_z: float = 0.015  # m
    tilt_noise_std: tuple[float, float] = (0.0087, 0.026)  # rad: 0.5 to 1.5 degrees of roll and pitch
    yaw_noise_std: tuple[float, float] = (0.0087, 0.0175)  # rad: 0.5 to 1 degree
    yaw_bias_std: float = 0.044  # rad: compass offset, 2.5 degrees, cut at twice that
    correlation_time: float = 0.5  # s, how slowly the errors move


class PayloadEstimator:
    """Position (N, 3) and rotation (N, 3, 3) of the payload as its own estimator reports them."""

    def __init__(self, cfg: PayloadEstimatorCfg, num_envs: int, physics_dt: float, device):
        self.cfg, self.dt, self.device = cfg, physics_dt, device
        self.every = max(1, round(1.0 / (cfg.rate_hz * physics_dt)))
        self._keep = math.exp(-physics_dt / cfg.correlation_time)
        self._drive = math.sqrt(1.0 - self._keep**2)
        self.pos_noise = torch.zeros(num_envs, 3, device=device)
        self.rot_noise = torch.zeros(num_envs, 3, device=device)
        self.pos_bias = torch.zeros(num_envs, 3, device=device)
        self.yaw_bias = torch.zeros(num_envs, device=device)
        self._pos_error = torch.zeros(num_envs, 3, device=device)
        self._rot_error = torch.zeros(num_envs, 3, device=device)
        self._steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.position = torch.zeros(num_envs, 3, device=device)
        self.rotation = torch.eye(3, device=device).repeat(num_envs, 1, 1)

    def reset(self, env_ids, position, rotation) -> None:
        cfg, n, dev = self.cfg, len(env_ids), self.device
        xy = draw_uniform(cfg.pos_noise_std_xy, (n, 1), dev)
        self.pos_noise[env_ids] = torch.cat((xy, xy, draw_uniform(cfg.pos_noise_std_z, (n, 1), dev)), dim=-1)
        tilt = draw_uniform(cfg.tilt_noise_std, (n, 1), dev)
        self.rot_noise[env_ids] = torch.cat((tilt, tilt, draw_uniform(cfg.yaw_noise_std, (n, 1), dev)), dim=-1)
        self.pos_bias[env_ids] = torch.cat(
            (draw_gaussian(cfg.pos_bias_std_xy, (n, 2), dev), draw_gaussian(cfg.pos_bias_std_z, (n, 1), dev)), dim=-1
        )
        self.yaw_bias[env_ids] = draw_gaussian(cfg.yaw_bias_std, (n,), dev, limit=2.0 * cfg.yaw_bias_std)
        # start anywhere in the error's own spread, not at zero
        self._pos_error[env_ids] = torch.randn(n, 3, device=dev) * self.pos_noise[env_ids]
        self._rot_error[env_ids] = torch.randn(n, 3, device=dev) * self.rot_noise[env_ids]
        self._steps[env_ids] = 0
        self.position[env_ids], self.rotation[env_ids] = (x[env_ids] for x in self._estimate(position, rotation))

    def step(self, position, rotation) -> None:
        """Advance by one physics step from the payload's true position (N, 3) and rotation (N, 3, 3)."""
        self._pos_error = self._keep * self._pos_error + self._drive * torch.randn_like(position) * self.pos_noise
        self._rot_error = self._keep * self._rot_error + self._drive * torch.randn_like(position) * self.rot_noise
        self._steps += 1
        new = self._steps % self.every == 0
        est_position, est_rotation = self._estimate(position, rotation)
        self.position = torch.where(new.unsqueeze(-1), est_position, self.position)
        self.rotation = torch.where(new[:, None, None], est_rotation, self.rotation)

    def _estimate(self, position, rotation):
        error = self._rot_error.clone()
        error[:, 2] += self.yaw_bias
        return position + self.pos_bias + self._pos_error, rotvec_to_matrix(error) @ rotation
