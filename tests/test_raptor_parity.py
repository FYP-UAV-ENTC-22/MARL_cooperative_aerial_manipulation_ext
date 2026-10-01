"""Check the PyTorch port of RAPTOR against the reference trajectory PX4 ships.

``policy.tar`` carries a 500-step input/output pair that the flight controller replays at
boot to verify the policy it loaded (mc_raptor.cpp:216). Running the same pair through the
port is the strongest correctness evidence available without flying, so it is the gate on
any change to the controller or the extracted weights.

Runs standalone -- no Isaac Sim needed::

    python tests/test_raptor_parity.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

# Load the module by path: importing it through the package would pull in the task registry,
# and with it Isaac Sim, which this test deliberately does not need.
_source = (
    Path(__file__).resolve().parents[1]
    / "exts/MARL_mav_carry_ext/MARL_mav_carry_ext/controllers/raptor.py"
)
_spec = importlib.util.spec_from_file_location("raptor", _source)
raptor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(raptor)

DEFAULT_CHECKPOINT = raptor.DEFAULT_CHECKPOINT
RaptorController = raptor.RaptorController
action_to_omega = raptor.action_to_omega

TOLERANCE = 1e-5


def test_reference_trajectory(device: str = "cpu") -> float:
    """Roll the policy over the reference input and compare against the reference output."""
    blob = np.load(DEFAULT_CHECKPOINT)
    reference_input = torch.from_numpy(blob["example_input"]).to(device)  # (500, 2, 22)
    reference_output = torch.from_numpy(blob["example_output"]).to(device)  # (500, 2, 4)

    num_steps, batch, _ = reference_input.shape
    # The reference pair is one environment holding `batch` independent sequences, which maps
    # onto the controller's (num_envs, num_drones) layout directly.
    policy = RaptorController(num_envs=1, num_drones=batch, device=device)

    predicted = torch.stack([policy(reference_input[t].unsqueeze(0)).squeeze(0) for t in range(num_steps)])

    error = (predicted - reference_output).abs().max().item()
    assert error < TOLERANCE, f"max abs error {error:.3g} exceeds {TOLERANCE:.0e}"
    return error


def test_reset_restores_initial_state(device: str = "cpu") -> None:
    """A reset environment must reproduce its very first action exactly."""
    blob = np.load(DEFAULT_CHECKPOINT)
    reference_input = torch.from_numpy(blob["example_input"]).to(device)

    policy = RaptorController(num_envs=2, num_drones=2, device=device)
    first = policy(reference_input[0].unsqueeze(0).expand(2, -1, -1)).clone()

    for t in range(1, 20):
        policy(reference_input[t].unsqueeze(0).expand(2, -1, -1))

    policy.reset(torch.tensor([0], device=device))
    after = policy(reference_input[0].unsqueeze(0).expand(2, -1, -1))

    assert torch.equal(after[0], first[0]), "reset environment did not return to the initial state"
    assert not torch.equal(after[1], first[1]), "untouched environment was reset as well"


def test_action_to_omega() -> None:
    """The mapping must span exactly the motor's range."""
    action = torch.tensor([-1.0, 0.0, 1.0])
    omega = action_to_omega(action, 150.0, 2800.0)
    assert torch.allclose(omega, torch.tensor([150.0, 1475.0, 2800.0]))


if __name__ == "__main__":
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        error = test_reference_trajectory(device)
        print(f"reference trajectory [{device}]  max abs error = {error:.3g}   (tolerance {TOLERANCE:.0e})")

    test_reset_restores_initial_state()
    print("reset semantics              ok")
    test_action_to_omega()
    print("action -> omega mapping      ok")
    print("\nall parity checks passed")
