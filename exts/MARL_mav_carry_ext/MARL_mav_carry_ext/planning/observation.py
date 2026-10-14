"""The MARL policy's observation frame. castor-main builds the same one onboard (castor_policy/flycrane.hpp).

    frame = [payload position (3), payload rotation (9, row-major), one-hot slot (D),
             drone position (3), drone rotation (9), drone linear velocity (3), drone angular velocity (3),
             goal position - payload position (3), goal rotation @ payload rotation^T (9)]

Everything in one world frame, both velocities too.
"""

from __future__ import annotations

import torch


def policy_frame(slot: int, num_drones: int, payload_position, payload_rotation, drone_position, drone_rotation,
                 drone_lin_vel, drone_ang_vel, goal_position_error, goal_rotation_error) -> torch.Tensor:
    """One frame for the drone in `slot`. Every argument is (num_envs, ...) for that drone."""
    num_envs = payload_position.shape[0]
    one_hot = torch.zeros(num_envs, num_drones, device=payload_position.device)
    one_hot[:, slot] = 1.0
    return torch.cat(
        (
            payload_position,
            payload_rotation.reshape(num_envs, -1),
            one_hot,
            drone_position.reshape(num_envs, -1),
            drone_rotation.reshape(num_envs, -1),
            drone_lin_vel.reshape(num_envs, -1),
            drone_ang_vel.reshape(num_envs, -1),
            goal_position_error,
            goal_rotation_error.reshape(num_envs, -1),
        ),
        dim=-1,
    )
