"""What PX4's estimator hands to RAPTOR and to the DDS link, modelled from the true state.

On the aircraft nothing flies on the true state. RAPTOR and the companion computer both get the output of EKF2,
which integrates the IMU and corrects the result with RTK fixes:

  body rates           every gyro sample, through a 40 Hz low-pass          (vehicle_angular_velocity)
  attitude             every integrated IMU sample, 200 Hz by default        (vehicle_attitude)
  position, velocity   once per estimator update, 100 Hz by default          (vehicle_local_position)
  odometry             the same instant, for the DDS link                    (vehicle_odometry)

so between two estimator updates RAPTOR is called several times on the same position and velocity.

Running EKF2 itself for thousands of environments is not practical. This keeps the *error* of an IMU and GNSS
filter instead and adds it to the truth, which is the same thing for a linear filter and needs no simulated
accelerometer:

  * the gyro's noise and bias tilt the attitude estimate; gravity and the magnetometer pull it back slowly;
  * the accelerometer's noise and bias, and gravity seen through that tilt, integrate into a velocity error and
    that into a position error, growing between fixes;
  * each RTK fix (5 Hz) pulls both back towards the receiver's own error;
  * the correction reaches the output smoothly (EKF2's output predictor, EKF2_TAU_VEL / EKF2_TAU_POS), so the
    velocity RAPTOR sees does not step at every fix.

The errors that come out drift slowly and are shared by RAPTOR and the policy, as on the aircraft. The gains are
fixed, not EKF2's time-varying ones: they are to be set against a log of the real board (docs in the task config).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..common import draw_gaussian, draw_uniform, rotvec_to_matrix
from .sensors import GnssCfg, ImuCfg, LowPass2

GRAVITY = 9.8066


@dataclass
class Px4EstimatorCfg:
    imu: ImuCfg = field(default_factory=ImuCfg)
    gnss: GnssCfg = field(default_factory=GnssCfg)
    position_rate_hz: float = 100.0  # EKF2_PREDICT_US = 10 ms
    attitude_rate_hz: float = 200.0  # IMU_INTEG_RATE
    tilt_correction_time: float = 2.0  # s, how fast gravity and the magnetometer pull the attitude error back
    yaw_bias_std: float = 0.03  # rad, magnetometer heading error, steady for the flight
    # how much of the difference to a fix is taken at that fix
    gnss_pos_gain: float = 0.2
    gnss_vel_gain: float = 0.2
    gnss_pos_to_vel_gain: float = 0.5  # 1/s
    # the filter also learns the steady part of the acceleration error (accelerometer bias, tilt) from the fixes, as
    # EKF2 does with its bias states; without it a constant error would leave a constant velocity offset
    gnss_vel_to_accel_gain: float = 0.05  # 1/s
    output_tau: float = 0.25  # s, EKF2_TAU_VEL and EKF2_TAU_POS


class Px4Estimator:
    """One estimator per drone of every environment, stepped once per physics step (the gyro rate).

    After ``step`` the attributes hold what PX4 would publish at that instant:

      ang_vel_body (N, D, 3)   position, velocity (N, D, 3)   attitude (N, D, 3, 3)
      odometry: position, attitude, velocity, world angular velocity, as last handed to the DDS link
    """

    def __init__(self, cfg: Px4EstimatorCfg, num_envs: int, num_drones: int, physics_dt: float, device):
        self.cfg, self.dt, self.device = cfg, physics_dt, device
        shape = (num_envs, num_drones)
        gyro_hz = 1.0 / physics_dt
        self.position_every = max(1, round(gyro_hz / cfg.position_rate_hz))
        self.attitude_every = max(1, round(gyro_hz / cfg.attitude_rate_hz))
        self.gnss_every = max(1, round(gyro_hz / cfg.gnss.rate_hz))

        # this episode's levels
        self.gyro_noise = torch.zeros(*shape, 1, device=device)
        self.accel_noise = torch.zeros(*shape, 1, device=device)
        self.gyro_bias = torch.zeros(*shape, 3, device=device)
        self.accel_bias = torch.zeros(*shape, 3, device=device)
        self.gnss_pos_noise = torch.zeros(*shape, 3, device=device)
        self.gnss_vel_noise = torch.zeros(*shape, 1, device=device)
        self.gnss_bias = torch.zeros(*shape, 3, device=device)
        self.yaw_bias = torch.zeros(*shape, device=device)
        self.gnss_phase = torch.zeros(shape, dtype=torch.long, device=device)

        # error state, world frame
        self.attitude_error = torch.zeros(*shape, 3, device=device)  # small rotation, without the yaw bias
        self.filter_pos_error = torch.zeros(*shape, 3, device=device)
        self.filter_vel_error = torch.zeros(*shape, 3, device=device)
        self.accel_correction = torch.zeros(*shape, 3, device=device)  # what the filter has learned to subtract
        # the part of past corrections the output has not taken yet
        self.pos_remainder = torch.zeros(*shape, 3, device=device)
        self.vel_remainder = torch.zeros(*shape, 3, device=device)

        self.rate_filter = LowPass2(gyro_hz, cfg.imu.gyro_cutoff_hz, (*shape, 3), device)
        self._gyro_sum = torch.zeros(*shape, 3, device=device)
        self._steps = torch.zeros(shape, dtype=torch.long, device=device)

        # outputs
        self.ang_vel_body = torch.zeros(*shape, 3, device=device)
        self.position = torch.zeros(*shape, 3, device=device)
        self.velocity = torch.zeros(*shape, 3, device=device)
        self.attitude = torch.eye(3, device=device).repeat(*shape, 1, 1)
        self.odom_position = torch.zeros(*shape, 3, device=device)
        self.odom_velocity = torch.zeros(*shape, 3, device=device)
        self.odom_attitude = torch.eye(3, device=device).repeat(*shape, 1, 1)
        self.odom_ang_vel = torch.zeros(*shape, 3, device=device)  # world frame

    # ---------------------------------------------------------------------------------------------------- reset
    def reset(self, env_ids, position, rotation, velocity, ang_vel_body) -> None:
        """New sensors for a new episode, and outputs that start on the (now known) state of the drones."""
        cfg, n, d, dev = self.cfg, len(env_ids), self.position.shape[1], self.device
        self.gyro_noise[env_ids] = draw_uniform(cfg.imu.gyro_noise_std, (n, d, 1), dev)
        self.accel_noise[env_ids] = draw_uniform(cfg.imu.accel_noise_std, (n, d, 1), dev)
        self.gyro_bias[env_ids] = draw_gaussian(cfg.imu.gyro_bias_std, (n, d, 3), dev)
        self.accel_bias[env_ids] = draw_gaussian(cfg.imu.accel_bias_std, (n, d, 3), dev)
        self.gnss_pos_noise[env_ids] = torch.cat(
            (
                draw_uniform(cfg.gnss.pos_noise_std_xy, (n, d, 1), dev).expand(n, d, 2),
                draw_uniform(cfg.gnss.pos_noise_std_z, (n, d, 1), dev),
            ),
            dim=-1,
        )
        self.gnss_vel_noise[env_ids] = draw_uniform(cfg.gnss.vel_noise_std, (n, d, 1), dev)
        self.gnss_bias[env_ids] = torch.cat(
            (
                draw_gaussian(cfg.gnss.pos_bias_std_xy, (n, d, 2), dev),
                draw_gaussian(cfg.gnss.pos_bias_std_z, (n, d, 1), dev),
            ),
            dim=-1,
        )
        self.yaw_bias[env_ids] = draw_gaussian(cfg.yaw_bias_std, (n, d), dev)
        self.gnss_phase[env_ids] = torch.randint(0, self.gnss_every, (n, d), device=dev)

        # a filter that has been running for a while sits on the receiver's offset
        self.attitude_error[env_ids] = 0.0
        self.filter_pos_error[env_ids] = self.gnss_bias[env_ids]
        self.filter_vel_error[env_ids] = 0.0
        self.accel_correction[env_ids] = -(rotation[env_ids] @ self.accel_bias[env_ids].unsqueeze(-1)).squeeze(-1)
        self.pos_remainder[env_ids] = 0.0
        self.vel_remainder[env_ids] = 0.0
        self._gyro_sum[env_ids] = 0.0
        self._steps[env_ids] = 0

        gyro = ang_vel_body[env_ids] + self.gyro_bias[env_ids]
        self.rate_filter.reset(env_ids, gyro)
        self.ang_vel_body[env_ids] = gyro
        attitude = self._estimated_attitude(rotation)[env_ids]
        self.attitude[env_ids] = attitude
        self.position[env_ids] = position[env_ids] + self.filter_pos_error[env_ids]
        self.velocity[env_ids] = velocity[env_ids]
        self.odom_position[env_ids] = self.position[env_ids]
        self.odom_velocity[env_ids] = self.velocity[env_ids]
        self.odom_attitude[env_ids] = attitude
        self.odom_ang_vel[env_ids] = (attitude @ gyro.unsqueeze(-1)).squeeze(-1)

    # ----------------------------------------------------------------------------------------------------- step
    def step(self, position, rotation, velocity, ang_vel_body) -> None:
        """Advance by one gyro sample. Inputs are the true state of the body origin: (N, D, 3), rotation (N, D, 3, 3)."""
        cfg, dt = self.cfg, self.dt

        # gyro: what the rate loop gets, and what tilts the attitude estimate
        gyro_error = self.gyro_bias + torch.randn_like(ang_vel_body) * self.gyro_noise
        gyro = ang_vel_body + gyro_error
        self.ang_vel_body = self.rate_filter(gyro)
        self._gyro_sum += gyro
        gyro_error_w = (rotation @ gyro_error.unsqueeze(-1)).squeeze(-1)
        self.attitude_error += gyro_error_w * dt - self.attitude_error * (dt / cfg.tilt_correction_time)

        # accelerometer: its own error, and gravity seen through the tilt error
        accel_error = self.accel_bias + torch.randn_like(ang_vel_body) * self.accel_noise
        accel_error_w = (rotation @ accel_error.unsqueeze(-1)).squeeze(-1)
        accel_error_w[..., 0] += GRAVITY * self.attitude_error[..., 1]
        accel_error_w[..., 1] -= GRAVITY * self.attitude_error[..., 0]
        self.filter_vel_error += (accel_error_w + self.accel_correction) * dt
        self.filter_pos_error += self.filter_vel_error * dt

        # the output catches up with earlier corrections
        decay = 1.0 - dt / cfg.output_tau
        self.pos_remainder *= decay
        self.vel_remainder *= decay

        # an RTK fix
        self._steps += 1
        fix = ((self._steps + self.gnss_phase) % self.gnss_every == 0).unsqueeze(-1)
        if fix.any():
            pos_innovation = self.gnss_bias + torch.randn_like(position) * self.gnss_pos_noise - self.filter_pos_error
            vel_innovation = torch.randn_like(velocity) * self.gnss_vel_noise - self.filter_vel_error
            pos_step = torch.where(fix, cfg.gnss_pos_gain * pos_innovation, torch.zeros_like(position))
            vel_step = torch.where(
                fix,
                cfg.gnss_vel_gain * vel_innovation + cfg.gnss_pos_to_vel_gain * cfg.gnss_pos_gain * pos_innovation,
                torch.zeros_like(velocity),
            )
            self.accel_correction += torch.where(
                fix, cfg.gnss_vel_to_accel_gain * vel_innovation, torch.zeros_like(velocity)
            )
            self.filter_pos_error += pos_step
            self.filter_vel_error += vel_step
            self.pos_remainder -= pos_step
            self.vel_remainder -= vel_step

        # what PX4 publishes at this instant; everything else keeps its last value
        new_attitude = (self._steps % self.attitude_every == 0)[..., None, None]
        attitude = self._estimated_attitude(rotation)
        self.attitude = torch.where(new_attitude, attitude, self.attitude)

        update = self._steps % self.position_every == 0
        new_position = update.unsqueeze(-1)
        self.position = torch.where(new_position, position + self.filter_pos_error + self.pos_remainder, self.position)
        self.velocity = torch.where(new_position, velocity + self.filter_vel_error + self.vel_remainder, self.velocity)
        # odometry carries the rates averaged over the update, not the low-passed ones
        mean_gyro = self._gyro_sum / self.position_every
        self.odom_position = torch.where(new_position, self.position, self.odom_position)
        self.odom_velocity = torch.where(new_position, self.velocity, self.odom_velocity)
        self.odom_attitude = torch.where(update[..., None, None], attitude, self.odom_attitude)
        self.odom_ang_vel = torch.where(
            new_position, (attitude @ mean_gyro.unsqueeze(-1)).squeeze(-1), self.odom_ang_vel
        )
        self._gyro_sum = torch.where(new_position, torch.zeros_like(self._gyro_sum), self._gyro_sum)

    def _estimated_attitude(self, rotation: torch.Tensor) -> torch.Tensor:
        error = self.attitude_error.clone()
        error[..., 2] += self.yaw_bias
        return rotvec_to_matrix(error) @ rotation
