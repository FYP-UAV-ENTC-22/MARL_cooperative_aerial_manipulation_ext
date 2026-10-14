"""What a drone's flight controller measures with: the IMU and the RTK receiver.

Every noise level is a range. Each episode draws one level per drone from it, and each sample is then Gaussian with
that level; steady errors (biases) are drawn once per episode. The ranges are the hardware's: a Pixhawk IMU on a
vibrating frame, and a u-blox ZED-F9P (Holybro H-RTK F9P) in RTK fix, which PX4's driver runs at 5 Hz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ImuCfg:
    # per sample at the gyro rate, after frame vibration
    gyro_noise_std: tuple[float, float] = (0.005, 0.03)  # rad/s
    accel_noise_std: tuple[float, float] = (0.05, 0.3)  # m/s^2
    # what is left after the estimator's own bias estimate, per axis
    gyro_bias_std: float = 0.002  # rad/s
    accel_bias_std: float = 0.03  # m/s^2
    # PX4 low-passes the gyro before anything flies on it (IMU_GYRO_CUTOFF)
    gyro_cutoff_hz: float = 40.0


@dataclass
class GnssCfg:
    rate_hz: float = 5.0  # PX4's ubx driver sets an F9P to 5 Hz
    # per fix, RTK fixed: 0.01 m + 1 ppm horizontally by the data sheet, widened
    pos_noise_std_xy: tuple[float, float] = (0.008, 0.02)  # m
    pos_noise_std_z: tuple[float, float] = (0.015, 0.04)  # m
    vel_noise_std: tuple[float, float] = (0.03, 0.07)  # m/s, data sheet 0.05
    # an offset that stays for the flight (multipath, antenna): relative to the payload it does not cancel
    pos_bias_std_xy: float = 0.01  # m
    pos_bias_std_z: float = 0.015  # m


class LowPass2:
    """PX4's LowPassFilter2p: a second-order Butterworth low-pass, one state per element."""

    def __init__(self, sample_hz: float, cutoff_hz: float, shape: tuple[int, ...], device):
        self.enabled = 0.0 < cutoff_hz < 0.5 * sample_hz
        ohm = math.tan(math.pi * cutoff_hz / sample_hz) if self.enabled else 1.0
        c = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm
        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c
        self.d1 = torch.zeros(shape, device=device)
        self.d2 = torch.zeros(shape, device=device)

    def reset(self, env_ids, value: torch.Tensor) -> None:
        """Start settled on `value`, so a reset does not ring."""
        settled = value / (1.0 + self.a1 + self.a2)
        self.d1[env_ids] = settled
        self.d2[env_ids] = settled

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return sample
        d0 = sample - self.d1 * self.a1 - self.d2 * self.a2
        out = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2 = self.d1
        self.d1 = d0
        return out
