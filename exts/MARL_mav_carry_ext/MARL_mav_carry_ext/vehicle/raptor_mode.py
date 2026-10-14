"""PX4's RAPTOR flight mode (src/modules/mc_raptor): what the frozen policy is given, and how it is called.

``raptor_observation`` is mc_raptor.cpp's ``observe``: the position and velocity errors and the attitude are
expressed in the setpoint's yaw frame, the body rates are taken as they are. The sign flips in the C++ are PX4's
NED/FRD to FLU conversion and do not appear here, where everything is already ENU/FLU. Whether the inputs are the
true state or an estimator's output is the caller's choice; on the aircraft they are the estimator's.

Not modelled: mc_raptor holding position when setpoints stop for 200 ms. Setpoints never stop in training.
"""

from __future__ import annotations

import torch


def yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """Rotation matrices about z for a batch of yaw angles, shaped (..., 3, 3)."""
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    zeros, ones = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        (
            torch.stack((cos_yaw, -sin_yaw, zeros), dim=-1),
            torch.stack((sin_yaw, cos_yaw, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )


def raptor_core_observation(position, rotation, velocity, ang_vel_body, sp_position, sp_velocity, sp_yaw,
                            frame_rotation: torch.Tensor | None = None):
    """Position error, attitude, velocity error and body rates as RAPTOR takes them, before clipping.

    All (num_envs, num_drones, ...). ``frame_rotation`` turns the body frame into the X-configuration frame RAPTOR
    was trained on, for an airframe that is not one.
    """
    target = yaw_rotation(sp_yaw)
    position_error = torch.einsum("ndji,ndj->ndi", target, position - sp_position)
    velocity_error = torch.einsum("ndji,ndj->ndi", target, velocity - sp_velocity)
    orientation = target.transpose(-1, -2) @ rotation
    if frame_rotation is not None:
        orientation = orientation @ frame_rotation
        ang_vel_body = ang_vel_body @ frame_rotation
    return position_error, orientation, velocity_error, ang_vel_body
