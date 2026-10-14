"""Checks of the flight-stack models (vehicle/, localization/, common/) without Isaac Sim.

    ~/isaacsim/python.sh tests/test_flight_stack.py

The package itself cannot be imported here (its __init__ registers the tasks, which needs Isaac Sim), so the
sub-packages are loaded under a stub parent.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

_root = Path(__file__).resolve().parents[1] / "exts/MARL_mav_carry_ext/MARL_mav_carry_ext"
_stub = types.ModuleType("MARL_mav_carry_ext")
_stub.__path__ = [str(_root)]
sys.modules.setdefault("MARL_mav_carry_ext", _stub)

from MARL_mav_carry_ext.common import LagBuffer, rotvec_to_matrix  # noqa: E402
from MARL_mav_carry_ext.vehicle import FlightStack, FlightStackCfg, LowPass2, Px4Estimator, Px4EstimatorCfg  # noqa: E402

DT = 1.0 / 400.0


def test_lag_buffer():
    buffer = LagBuffer(8, 2, (3, 1), "cpu", per_reader=True)
    for k in range(20):
        buffer.push(torch.full((2, 3, 1), float(k)))
    lag = torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert buffer.read(lag).squeeze(-1).tolist() == [[19.0, 18.0, 17.0], [16.0, 15.0, 14.0]]
    shared = LagBuffer(8, 2, (1,), "cpu")
    for k in range(20):
        shared.push(torch.full((2, 1), float(k)))
    assert shared.read(lag).squeeze(-1).tolist() == [[19.0, 18.0, 17.0], [16.0, 15.0, 14.0]]
    print("lag buffer: ok")


def test_low_pass():
    """-3 dB at the cut-off, as a second-order Butterworth, and unity at DC."""
    for hz, expected in ((1.0, 1.0), (40.0, 1.0 / math.sqrt(2.0))):
        f = LowPass2(400.0, 40.0, (1,), "cpu")
        t = torch.arange(4000) * DT
        out = torch.stack([f(torch.sin(2 * math.pi * hz * x).reshape(1)) for x in t]).squeeze()
        gain = out[2000:].abs().max().item()
        assert abs(gain - expected) < 0.03, (hz, gain)
    print("gyro low-pass: ok (0 dB at 1 Hz, -3 dB at 40 Hz)")


def test_estimator():
    """A drone standing still: how large the estimate's errors are and how often each output changes."""
    torch.manual_seed(0)
    n, d = 512, 3
    est = Px4Estimator(Px4EstimatorCfg(), n, d, DT, "cpu")
    pos = torch.zeros(n, d, 3)
    rot = torch.eye(3).repeat(n, d, 1, 1)
    zero = torch.zeros(n, d, 3)
    est.reset(torch.arange(n), pos, rot, zero, zero)
    positions, velocities, tilts, rates = [], [], [], []
    for k in range(int(20.0 / DT)):
        est.step(pos, rot, zero, zero)
        if k >= int(5.0 / DT):
            positions.append(est.position.clone())
            velocities.append(est.velocity.clone())
            tilts.append(torch.acos(est.attitude[..., 2, 2].clamp(-1, 1)))
            rates.append(est.ang_vel_body.clone())
    positions, velocities = torch.stack(positions), torch.stack(velocities)
    changed = (positions[1:] != positions[:-1]).any(dim=-1).float().mean().item()
    assert abs(changed - 0.25) < 0.01, changed  # a new position on every fourth gyro sample
    pos_rms = positions.pow(2).mean().sqrt().item()
    vel_rms = velocities.pow(2).mean().sqrt().item()
    vel_step = (velocities[1:] - velocities[:-1]).abs().max().item()
    tilt_deg = math.degrees(torch.stack(tilts).mean().item())
    rate_rms = torch.stack(rates).pow(2).mean().sqrt().item()
    print(f"estimator, drone at rest: position error {100 * pos_rms:.1f} cm rms per axis, velocity error "
          f"{100 * vel_rms:.1f} cm/s rms, largest velocity change between two gyro samples {100 * vel_step:.1f} cm/s, "
          f"tilt error {tilt_deg:.2f} deg mean, body-rate noise after the filter {rate_rms:.4f} rad/s rms")
    assert 0.005 < pos_rms < 0.04 and 0.003 < vel_rms < 0.06 and tilt_deg < 1.0


def test_stack_delays():
    """A drone moving at 1 m/s: the policy sees it where it was, RAPTOR gets the setpoint late."""
    torch.manual_seed(0)
    n, d = 64, 3
    cfg = FlightStackCfg()
    for c in (cfg.estimator.imu, cfg.estimator.gnss):  # no noise: only the timing is under test
        for name in vars(c):
            if name.endswith("_std") or name.endswith("_std_xy") or name.endswith("_std_z"):
                value = getattr(c, name)
                setattr(c, name, (0.0, 0.0) if isinstance(value, tuple) else 0.0)
    cfg.estimator.yaw_bias_std = 0.0
    cfg.dds.state_jitter_s = 0.0
    cfg.wifi.jitter_s, cfg.wifi.late_probability = 0.0, 0.0
    for name in ("pos_noise_std_xy", "pos_noise_std_z", "tilt_noise_std", "yaw_noise_std"):
        setattr(cfg.payload, name, (0.0, 0.0))
    cfg.payload.pos_bias_std_xy = cfg.payload.pos_bias_std_z = cfg.payload.yaw_bias_std = 0.0
    stack = FlightStack(cfg, n, d, DT, "cpu")
    rot = torch.eye(3).repeat(n, d, 1, 1)
    vel = torch.zeros(n, d, 3)
    vel[..., 0] = 1.0
    zero = torch.zeros(n, d, 3)
    prot = torch.eye(3).repeat(n, 1, 1)
    stack.reset(torch.arange(n), zero, rot, vel, zero, torch.zeros(n, 3), prot, torch.zeros(n, d, 4))
    for k in range(1, 401):
        t = k * DT
        stack.command(torch.full((n, d, 3), t), zero, torch.zeros(n, d))
        stack.step(vel * t, rot, vel, zero, torch.tensor([t, 0.0, 0.0]).repeat(n, 1), prot)
    t = 400 * DT
    drone_age = t - stack.policy_drone_state()[0][..., 0]
    payload_age = t - stack.policy_payload_pose()[0][..., 0]
    setpoint_age = t - stack.raptor_inputs()[4][..., 0]
    raptor_age = t - stack.raptor_inputs()[0][..., 0]
    print(f"ages at the policy: drone state {1e3 * drone_age.min():.1f} to {1e3 * drone_age.max():.1f} ms, payload "
          f"pose {1e3 * payload_age.min():.1f} to {1e3 * payload_age.max():.1f} ms; at RAPTOR: its own position "
          f"{1e3 * raptor_age.min():.1f} to {1e3 * raptor_age.max():.1f} ms, the setpoint "
          f"{1e3 * setpoint_age.min():.1f} to {1e3 * setpoint_age.max():.1f} ms")
    lo, hi = cfg.dds.state_delay_s
    assert drone_age.min() >= lo - 1e-3 and drone_age.max() <= hi + 0.0125 + 1e-3
    assert raptor_age.max() <= 0.0075 + 1e-4 and setpoint_age.max() <= cfg.dds.setpoint_delay_s[1] + DT + 1e-4


def test_rotation():
    r = rotvec_to_matrix(torch.tensor([[0.0, 0.0, math.pi / 2]]))
    assert torch.allclose(r[0] @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    print("rotation: ok")


if __name__ == "__main__":
    test_rotation()
    test_lag_buffer()
    test_low_pass()
    test_estimator()
    test_stack_delays()
    print("PASS")
