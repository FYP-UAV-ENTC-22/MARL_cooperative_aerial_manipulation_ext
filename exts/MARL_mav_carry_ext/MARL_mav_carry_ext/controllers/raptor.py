"""RAPTOR foundation policy as a batched, frozen low-level controller.

RAPTOR (Eschmann et al., Science Robotics 2025) replaces the geometric + INDI cascade:
it maps a 22-dim observation to four normalised motor commands and is trained across a
distribution of quadrotors, so it adapts online through its recurrent state.

The network is tiny -- Dense(22 -> 16, ReLU) -> GRU(16) -> Dense(16 -> 4, identity), about
5k parameters -- so evaluating every drone of every environment in one batched call costs
nothing next to physics.

Observation contract (see ``PX4-Autopilot/src/modules/mc_raptor/mc_raptor.cpp:344``, which
is the code that actually flies)::

     0:3   position error      drone_pos - setpoint_pos, in the target-yaw frame, +-0.5 m
     3:12  orientation         rotation matrix of (q_target* x q_drone), row-major
    12:15  velocity error      drone_vel - setpoint_vel, in the target-yaw frame, +-1.0 m/s
    15:18  angular velocity    body frame, rad/s
    18:22  previous action     last four motor commands, zero on reset

Note the sign: error is *current minus setpoint*, and the frame is FLU. The sign flips in
mc_raptor.cpp are PX4's NED/FRD -> FLU conversion and do not apply in Isaac, which is
already FLU.

The action is four motor commands in [-1, 1] ordered **front-right, back-right, back-left,
front-left**, mapped to rotor speed by :func:`action_to_omega`.

Weights come from ``data/raptor_v1.npz``, produced by
``scripts/tools/extract_raptor_weights.py`` from the checkpoint PX4 flies. rl_tools stores
the GRU gates in the order (r, z, n), which is exactly PyTorch's convention, so the
parameters load into ``nn.GRUCell`` unpermuted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_CHECKPOINT = Path(__file__).parent / "data" / "raptor_v1.npz"

# Error clips the policy was trained behind, from mc_raptor.hpp:117.
MAX_POSITION_ERROR = 0.5  # m
MAX_VELOCITY_ERROR = 1.0  # m/s


class RaptorController(nn.Module):
    """Frozen quadrotor foundation policy, evaluated for all drones at once.

    Args:
        num_envs: number of parallel environments.
        num_drones: drones per environment.
        device: device to hold the parameters and recurrent state on.
        checkpoint: path to the extracted ``.npz``.
    """

    OBS_DIM = 22
    ACTION_DIM = 4
    HIDDEN_DIM = 16

    def __init__(
        self,
        num_envs: int,
        num_drones: int,
        device: str | torch.device,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
    ):
        super().__init__()
        self.num_envs = num_envs
        self.num_drones = num_drones

        blob = np.load(Path(checkpoint))

        self.dense0 = nn.Linear(self.OBS_DIM, self.HIDDEN_DIM)
        self.gru = nn.GRUCell(self.HIDDEN_DIM, self.HIDDEN_DIM)
        self.dense2 = nn.Linear(self.HIDDEN_DIM, self.ACTION_DIM)

        def load(param: nn.Parameter, key: str):
            tensor = torch.from_numpy(blob[key].astype(np.float32))
            if tensor.shape != param.shape:
                raise ValueError(f"{key}: checkpoint has {tuple(tensor.shape)}, layer wants {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(tensor)

        load(self.dense0.weight, "dense0_weight")
        load(self.dense0.bias, "dense0_bias")
        load(self.gru.weight_ih, "gru_weight_ih")
        load(self.gru.bias_ih, "gru_bias_ih")
        load(self.gru.weight_hh, "gru_weight_hh")
        load(self.gru.bias_hh, "gru_bias_hh")
        load(self.dense2.weight, "dense2_weight")
        load(self.dense2.bias, "dense2_bias")

        # The trained initial hidden state. Resetting to zeros instead costs the policy its
        # learned starting estimate, so keep it around.
        self.register_buffer("h0", torch.from_numpy(blob["gru_h0"].astype(np.float32)))

        # RAPTOR is part of the environment, not of the trained graph.
        self.requires_grad_(False)
        self.eval()
        self.to(device)

        # (num_envs, num_drones, HIDDEN_DIM), flattened for the GRU cell.
        self.hidden_state = self.initial_hidden_state()
        self.previous_action = torch.zeros(num_envs, num_drones, self.ACTION_DIM, device=device)

    @property
    def device(self) -> torch.device:
        return self.h0.device

    def initial_hidden_state(self) -> torch.Tensor:
        """The checkpoint's ``h0``, broadcast to every drone of every environment."""
        return self.h0.expand(self.num_envs, self.num_drones, self.HIDDEN_DIM).clone()

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        """Restore the recurrent state and action history for the given environments.

        The GRU state is RAPTOR's online estimate of the vehicle it is flying. Carrying it
        across an episode boundary hands the policy a stale adaptation for a drone that just
        teleported, so this must be called from the environment's ``_reset_idx``.
        """
        if env_ids is None:
            env_ids = slice(None)
        self.hidden_state[env_ids] = self.h0
        self.previous_action[env_ids] = 0.0

    @torch.no_grad()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Step the policy for every drone.

        Args:
            observation: ``(num_envs, num_drones, 22)``, laid out per the module docstring.

        Returns:
            ``(num_envs, num_drones, 4)`` motor commands in [-1, 1], ordered
            front-right, back-right, back-left, front-left.
        """
        flat = observation.reshape(-1, self.OBS_DIM)
        hidden = self.hidden_state.reshape(-1, self.HIDDEN_DIM)

        hidden = self.gru(torch.relu(self.dense0(flat)), hidden)
        action = self.dense2(hidden)

        self.hidden_state = hidden.view(self.num_envs, self.num_drones, self.HIDDEN_DIM)
        self.previous_action = action.view(self.num_envs, self.num_drones, self.ACTION_DIM)
        return self.previous_action


def assemble_observation(
    position_error: torch.Tensor,
    rotation_matrix: torch.Tensor,
    velocity_error: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Clip and concatenate the RAPTOR observation.

    Both errors must already be expressed in the target-yaw frame; the clipping that keeps
    the policy inside its trained envelope is applied here so it cannot be forgotten.

    Args:
        position_error: ``(..., 3)`` drone position minus setpoint position.
        rotation_matrix: ``(..., 3, 3)`` orientation relative to the target yaw frame.
        velocity_error: ``(..., 3)`` drone velocity minus setpoint velocity.
        angular_velocity_body: ``(..., 3)`` angular velocity in the *body* frame.
        previous_action: ``(..., 4)`` the policy's own last output.

    Returns:
        ``(..., 22)`` observation.
    """
    return torch.cat(
        (
            position_error.clamp(-MAX_POSITION_ERROR, MAX_POSITION_ERROR),
            rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9),
            velocity_error.clamp(-MAX_VELOCITY_ERROR, MAX_VELOCITY_ERROR),
            angular_velocity_body,
            previous_action,
        ),
        dim=-1,
    )


def action_to_omega(action: torch.Tensor, omega_min: float, omega_max: float) -> torch.Tensor:
    """Map motor commands in [-1, 1] onto rotor speeds, per the RAPTOR convention."""
    return (omega_max - omega_min) * (action + 1.0) * 0.5 + omega_min
