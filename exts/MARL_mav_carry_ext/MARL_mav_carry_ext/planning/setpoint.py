"""The position setpoint each drone's RAPTOR is given, moved by the policy's actions.

The setpoint is absolute and persists across steps. Re-anchoring it to the drone every step instead would cancel the
position out of RAPTOR's observation: the error would reduce to the negated action, leaving the inner loop with no
feedback and the cascade with no absolute reference. A leash keeps it within reach of a drone that cannot keep up.
castor-main integrates the same way onboard (castor_policy/flycrane.hpp, advance).
"""

from __future__ import annotations

import torch


class SetpointIntegrator:
    def __init__(self, num_envs: int, num_drones: int, step_dt: float, step_scale: float, leash: float,
                 max_speed: float | None, device):
        self.step_dt, self.step_scale, self.leash, self.max_speed = step_dt, step_scale, leash, max_speed
        self.position = torch.zeros(num_envs, num_drones, 3, device=device)
        self.velocity = torch.zeros(num_envs, num_drones, 3, device=device)
        # Yaw is a free degree of freedom, a single cable transmits no yaw torque, but free means it costs nothing to
        # hold, not that it must be zero: the drones are mounted at different azimuths around the payload.
        self.yaw = torch.zeros(num_envs, num_drones, device=device)
        # Set on reset and used at the next step: the drones' pose is only readable after the observation pass.
        self.seed_pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        # How often the policy asks for a setpoint further ahead than the leash allows.
        self.leash_active_steps = torch.zeros(num_envs, device=device)
        self.steps = torch.zeros(num_envs, device=device)

    def reset(self, env_ids) -> None:
        self.seed_pending[env_ids] = True
        self.velocity[env_ids] = 0.0
        self.leash_active_steps[env_ids] = 0.0
        self.steps[env_ids] = 0.0

    def step(self, actions: torch.Tensor, drone_position: torch.Tensor, drone_yaw: torch.Tensor) -> None:
        """Move every setpoint by its action. `actions` and `drone_position` are (num_envs, num_drones, 3)."""
        if self.seed_pending.any():
            seed = self.seed_pending.view(-1, 1, 1)
            self.position = torch.where(seed, drone_position, self.position)
            self.velocity = torch.where(seed, torch.zeros_like(self.velocity), self.velocity)
            self.yaw = torch.where(seed.squeeze(-1), drone_yaw, self.yaw)
            self.seed_pending[:] = False

        increment = actions * self.step_scale
        if self.max_speed is not None:
            max_step = self.max_speed * self.step_dt
            step = increment.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            increment = increment * (max_step / step).clamp(max=1.0)

        self.position += increment
        # The increment is the velocity command, so the feedforward cannot contradict the position.
        self.velocity[:] = increment / self.step_dt

        offset = self.position - drone_position
        distance = offset.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.leash / distance).clamp(max=1.0)
        self.position[:] = drone_position + offset * scale

        self.leash_active_steps += (scale < 1.0).any(dim=1).squeeze(-1).float()
        self.steps += 1.0
