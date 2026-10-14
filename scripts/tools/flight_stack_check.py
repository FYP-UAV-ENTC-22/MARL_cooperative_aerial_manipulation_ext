"""Fly a trained policy with parts of the flight stack switched on, and measure how steadily the drones fly.

The question this answers: which of the things RAPTOR and the policy are given on the aircraft, and not in an ideal
simulation, makes the drones shake. One variant per run:

    ideal        both controllers on the true state (no flight stack)
    full         the whole stack as configured
    timing       the stack's rates and delays, no noise and no steady errors
    hold         only PX4's position and velocity refreshed at 100 Hz
    gyro_filter  only the 40 Hz low-pass on the body rates
    delays       only the DDS setpoint delay and the ESC delay
    noise        only the sensor noise and steady errors, every rate ideal
    px4_sim      the timing measured in castor-main's PX4 simulation run (run 12, checks 7 and 8), no noise: position
                 and velocity every second gyro sample, a 40 Hz gyro filter, 8 to 12 ms from RAPTOR's command to the
                 rotor, drone state 16 to 20 ms late, payload pose 0 to 4 ms late. For any RAPTOR task, at its rate.

Reported over all drones of the environments that did not end early, from --settle seconds on: the vertical velocity
between 5 and 12 Hz (where the PX4 run shook), the frequency of the largest peak between 2 and 20 Hz, roll and pitch
rate, and the payload's distance from its goal.

    ~/isaacsim/python.sh scripts/tools/flight_stack_check.py --variant full \\
        --checkpoint logs/skrl/<experiment>/<run>/checkpoints/best_agent.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

VARIANTS = ("ideal", "full", "timing", "hold", "gyro_filter", "delays", "noise", "px4_sim")
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-castor-s500-px4-payload-decentralized-hovering-v0")
parser.add_argument("--checkpoint", type=Path, default=None, help="policy to fly; without one the setpoints stay put")
parser.add_argument("--variant", choices=VARIANTS, default="full")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=16.0)
parser.add_argument("--settle", type=float, default=8.0, help="seconds left out of the measurement")
parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                    help="override one flight-stack setting after the variant, e.g. esc.delay_s=0,0 or "
                         "estimator.position_rate_hz=250; repeatable")
parser.add_argument("--no_feedforward", action="store_true", help="send RAPTOR a zero setpoint velocity")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app
sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import MARL_mav_carry_ext.tasks  # noqa: F401, E402  -- registers the task


def zero_noise(stack) -> None:
    """Every noise level and steady error of the stack to zero; rates and delays untouched."""
    for cfg in (stack.estimator.imu, stack.estimator.gnss, stack.estimator, stack.payload):
        for name, value in vars(cfg).items():
            if "_std" in name:
                setattr(cfg, name, (0.0, 0.0) if isinstance(value, tuple) else 0.0)
    stack.dds.state_jitter_s = 0.0
    stack.wifi.jitter_s = 0.0
    stack.wifi.late_probability = 0.0


def ideal_timing(stack, physics_hz: float) -> None:
    """Every rate at the physics rate and every delay zero; noise untouched."""
    stack.estimator.position_rate_hz = physics_hz
    stack.estimator.attitude_rate_hz = physics_hz
    stack.estimator.imu.gyro_cutoff_hz = 0.0
    stack.payload.rate_hz = physics_hz
    stack.dds.state_delay_s = stack.dds.setpoint_delay_s = (0.0, 0.0)
    stack.wifi.delay_s = (0.0, 0.0)
    stack.esc.delay_s = (0.0, 0.0)


def configure(env_cfg) -> None:
    stack, physics_hz = env_cfg.flight_stack, 1.0 / env_cfg.sim.dt
    if args.variant == "ideal":
        env_cfg.flight_stack = None
        return
    if args.variant == "px4_sim":
        from MARL_mav_carry_ext.vehicle import FlightStackCfg

        stack = env_cfg.flight_stack = FlightStackCfg()
        zero_noise(stack)
        stack.estimator.position_rate_hz = physics_hz / 2.0
        stack.estimator.attitude_rate_hz = physics_hz
        stack.dds.state_delay_s = (0.016, 0.020)
        stack.dds.setpoint_delay_s = (0.0, 0.004)
        stack.esc.delay_s = (0.008, 0.012)
        stack.wifi.delay_s = (0.0, 0.004)
        for name in ("drone_pos", "drone_rot", "drone_lin_vel", "drone_ang_vel", "payload_pos", "payload_rot"):
            setattr(env_cfg, f"{name}_noise_std", 0.0)
        env_cfg.obs_delay_steps_range = (0, 0)
        return
    if stack is None:
        raise SystemExit(f"{args.task} has no flight stack: only --variant ideal applies")
    reference = type(stack)()
    if args.variant in ("timing", "hold", "gyro_filter", "delays"):
        zero_noise(stack)
    if args.variant in ("hold", "gyro_filter", "delays", "noise"):
        ideal_timing(stack, physics_hz)
    if args.variant == "hold":
        stack.estimator.position_rate_hz = reference.estimator.position_rate_hz
    elif args.variant == "gyro_filter":
        stack.estimator.imu.gyro_cutoff_hz = reference.estimator.imu.gyro_cutoff_hz
    elif args.variant == "delays":
        stack.dds.setpoint_delay_s = reference.dds.setpoint_delay_s
        stack.esc.delay_s = reference.esc.delay_s


def override(stack) -> None:
    for item in args.set:
        path, text = item.split("=", 1)
        *parents, name = path.split(".")
        target = stack
        for parent in parents:
            target = getattr(target, parent)
        values = tuple(float(x) for x in text.split(","))
        setattr(target, name, values if isinstance(getattr(target, name), tuple) else values[0])


def band(signal: np.ndarray, rate: float, low: float, high: float):
    """rms of `signal` (time, ...) within [low, high] Hz, and the frequency of its largest peak in 2 to 20 Hz."""
    signal = signal - signal.mean(axis=0)
    power = (np.abs(np.fft.rfft(signal, axis=0)) ** 2).mean(axis=1)  # mean over drones
    freq = np.fft.rfftfreq(len(signal), 1.0 / rate)
    inside = (freq >= low) & (freq <= high)
    rms = float(np.sqrt(2.0 * power[inside].sum()) / len(signal))
    wide = (freq >= 2.0) & (freq <= 20.0)
    return rms, float(freq[wide][np.argmax(power[wide])])


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    configure(env_cfg)
    if env_cfg.flight_stack is not None:
        override(env_cfg.flight_stack)
    agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    agents = list(u.cfg.possible_agents)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = None
    if args.checkpoint is not None:
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, agent_cfg)
        runner.agent.load(str(args.checkpoint))
        runner.agent.set_running_mode("eval")
    if args.no_feedforward:
        step = u.setpoint.step

        def without_velocity(*a, **k):
            step(*a, **k)
            u.setpoint.velocity[:] = 0.0
            if u.stack is not None:
                u.stack.command(u.setpoint.position, u.setpoint.velocity, u.setpoint.yaw)

        u.setpoint.step = without_velocity

    steps, first = int(args.seconds / u.step_dt), int(args.settle / u.step_dt)
    ended = torch.zeros(args.num_envs, dtype=torch.bool, device=u.device)
    vz, rate, error = [], [], []
    obs, _ = env.reset()
    with torch.inference_mode():
        for k in range(steps):
            if runner is None:
                actions = {a: torch.zeros(args.num_envs, u.cfg.action_spaces[a], device=u.device) for a in agents}
            else:
                out = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in agents}
            obs, _, terminated, truncated, _ = env.step(actions)
            ended |= (terminated[agents[0]] | truncated[agents[0]]).reshape(-1)
            if k >= first:
                vz.append(u.drone_linear_velocities[..., 2].cpu().numpy().copy())
                body_rate = torch.einsum("ndji,ndj->ndi", u.drone_rot_matrices, u.drone_angular_velocities)
                rate.append(body_rate[..., :2].norm(dim=-1).cpu().numpy().copy())
                error.append((u.pose_command_w[:, :3] - u.load_position).norm(dim=-1).cpu().numpy().copy())
    alive = ~ended.cpu().numpy()
    vz, rate, error = (np.stack(x)[:, alive] for x in (vz, rate, error))
    vz = vz.reshape(len(vz), -1)
    shake, peak = band(vz, 1.0 / u.step_dt, 5.0, 12.0)
    stack = "none" if u.stack is None else type(u.stack).__name__
    print(f"RESULT variant={args.variant}{''.join(' ' + x for x in args.set)} policy={'yes' if runner else 'no'} feedforward={'off' if args.no_feedforward else 'on'} "
          f"stack={stack} physics={1.0 / u.physics_dt:.0f}Hz envs_kept={int(alive.sum())}/{args.num_envs} "
          f"vz_5_12Hz={1e3 * shake:.1f}mm/s peak={peak:.1f}Hz vz_rms={1e3 * vz.std(axis=0).mean():.1f}mm/s "
          f"rollpitch_rate_rms={np.degrees(np.sqrt((rate ** 2).mean())):.1f}deg/s "
          f"payload_error_last2s={error[-int(2.0 / u.step_dt):].mean():.3f}m")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # Kit swallows uncaught exceptions and still exits 0
        import traceback

        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
