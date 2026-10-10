"""Check the CASTOR rig before training on it: asset layout, rotor mapping, and RAPTOR holding the rig still.

With zero actions the setpoints stay at the spawn pose, so RAPTOR alone has to carry the payload. If
that does not hold, no policy trained on top of it will; the causes are the same ones
``raptor_single_drone.py`` isolates (rotor order, spin directions, the rotor-speed range), plus the
asset itself.

    ~/isaacsim/python.sh scripts/MARL_mav_carry/testing_scripts/castor_hold_check.py --headless
    ~/isaacsim/python.sh scripts/MARL_mav_carry/testing_scripts/castor_hold_check.py --num_envs 4   # watch it

Exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-castor-payload-decentralized-hovering-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=15.0, help="kept under episode_length_s to avoid a time-out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app
sys.stdout.reconfigure(line_buffering=True)  # SimulationApp.close() drops buffered output

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import MARL_mav_carry_ext.tasks  # noqa: F401, E402  -- registers the task
from MARL_mav_carry_ext.controllers.raptor import (  # noqa: E402
    MAX_POSITION_ERROR,
    MAX_VELOCITY_ERROR,
    assemble_observation,
)
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

RAPTOR_ORDER = ["front-right", "back-right", "back-left", "front-left"]


def quadrant(offset: torch.Tensor) -> str:
    return f"{'front' if offset[0] > 0 else 'back'}-{'right' if offset[1] < 0 else 'left'}"  # FLU: +y is left


def main() -> bool:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    robot, cfg, n = u.robot, u.cfg, u._num_drones
    env.reset()
    ok = True

    print("\n--- asset ---")
    print(f"  {robot.num_bodies} bodies, {robot.num_joints} joints, total mass {robot.data.default_mass[0].sum():.3f} kg")
    names = robot.body_names
    for label, ids in (("drones", u._falcon_idx), ("payload", u._payload_idx), ("cable, payload end", u._rope_idx),
                       ("cable, drone end", u._rope_top_idx)):
        print(f"  {label:20s} {[names[i] for i in ids]}")
    print(f"  {'rotors':20s} {[names[i] for i in u._falcon_rotor_idx]}")

    state = robot.data.body_com_state_w
    load = state[0, u._payload_idx[0], :3]
    load_rot = matrix_from_quat(state[0, u._payload_idx[0], 3:7].unsqueeze(0))[0]
    print("\n--- layout in the payload frame (one-hot slot = drone index) ---")
    for i in range(n):
        offset = load_rot.T @ (state[0, u._falcon_idx[i], :3] - load)
        azimuth = torch.rad2deg(torch.atan2(offset[1], offset[0])).item()
        print(f"  slot {i}: azimuth {azimuth:+7.1f} deg, {offset[:2].norm():.3f} m out, {offset[2]:.3f} m up")

    print("\n--- rotor order ---")
    permutation = []
    for slot in range(4):
        base, rotor = u._falcon_idx[0], u._falcon_rotor_idx[slot]
        rotation = matrix_from_quat(state[0, base, 3:7].unsqueeze(0))[0]
        permutation.append(RAPTOR_ORDER.index(quadrant(rotation.T @ (state[0, rotor, :3] - state[0, base, :3]))))
    match = tuple(permutation) == tuple(cfg.raptor_rotor_permutation)
    ok &= match
    print(f"  measured {tuple(permutation)}, configured {tuple(cfg.raptor_rotor_permutation)}  {'ok' if match else 'MISMATCH'}")

    print(f"\n--- holding the spawn pose for {args.seconds:.0f} s with zero actions, {args.num_envs} envs ---")
    zero = {a: torch.zeros(args.num_envs, cfg.action_spaces[a], device=u.device) for a in cfg.possible_agents}
    steps = int(args.seconds / u.step_dt)
    start_load, start_yaw = u.load_position.clone(), None
    terminated = torch.zeros(args.num_envs, dtype=torch.bool, device=u.device)
    track, yaw_rate, tilt, throttle = [], [], [], []
    with torch.inference_mode():
        for step in range(steps):
            _, _, term, _, _ = env.step(zero)
            terminated |= term[cfg.possible_agents[0]]
            yaw = torch.atan2(u.drone_rot_matrices[..., 1, 0], u.drone_rot_matrices[..., 0, 0])
            start_yaw = yaw.clone() if start_yaw is None else start_yaw
            if step * u.step_dt >= 3.0:  # after the start transient
                track.append((u.sp_pos - u.drone_positions).norm(dim=-1))
                # body_com_state_w gives the angular velocity in the world frame
                yaw_rate.append(u.drone_angular_velocities[..., 2].abs())
                tilt.append(torch.rad2deg(torch.acos(u.drone_rot_matrices[..., 2, 2].clamp(-1, 1))))
                throttle.append((u.raptor.previous_action + 1.0) / 2.0)
    alive = ~terminated
    track, yaw_rate, tilt, throttle = (torch.stack(x)[:, alive] for x in (track, yaw_rate, tilt, throttle))
    drift = (u.load_position - start_load)[alive]
    yaw_drift = torch.rad2deg((yaw - start_yaw + torch.pi) % (2 * torch.pi) - torch.pi)[alive].abs()

    checks = [
        ("episodes terminated early", float(terminated.sum()), 0.0, ""),
        ("setpoint tracking error, mean", track.mean().item(), 0.30, "m"),
        ("payload drift, max", drift.norm(dim=-1).max().item(), 0.30, "m"),
        ("drone yaw drift, max", yaw_drift.max().item(), 15.0, "deg"),
        ("drone yaw rate, mean", yaw_rate.mean().item(), 0.2, "rad/s"),
        ("drone tilt, max", tilt.max().item(), 20.0, "deg"),
    ]
    for name, value, limit, unit in checks:
        passed = value <= limit
        ok &= passed
        print(f"  {name:32s} {value:8.3f} {unit:5s} (limit {limit:g})  {'ok' if passed else 'FAIL'}")
    print(f"  {'throttle, mean / max':32s} {throttle.mean().item():8.3f} / {throttle.max().item():.3f}")
    print(f"  {'payload sag below spawn, mean':32s} {-drift[:, 2].mean().item():8.3f} m")

    if u._cable_length_joint_ids is not None:
        print("\n--- cable lengths (each cable gets its own offset every episode) ---")
        ids = u._cable_length_joint_ids
        position, target = robot.data.joint_pos[:, ids], robot.data.joint_pos_target[:, ids]
        mounts = robot.find_bodies(cfg.falcon_names.replace("base_link", "mount_link"))[0]
        # seg_0's link frame sits on the payload anchor; the mount link is the tie point on the drone
        span = (robot.data.body_link_pos_w[:, mounts] - robot.data.body_link_pos_w[:, u._rope_idx]).norm(dim=-1)
        low, high = cfg.cable_length_offset_range
        print(f"  offsets drawn in [{low:+.2f}, {high:+.2f}] m; this run: {target.min().item():+.3f} .. {target.max().item():+.3f} m")
        print(f"  anchor-to-mount distance: {span[alive].min().item():.3f} .. {span[alive].max().item():.3f} m")
        hold_error = (position - target).abs()[alive].max().item()
        spread = (target.max() - target.min()).item()
        for name, value, limit, passed in (
            ("joint off its target, max", hold_error, 0.005, hold_error <= 0.005),
            ("spread of the offsets", spread, 0.5 * (high - low), spread >= 0.5 * (high - low)),
        ):
            ok &= passed
            print(f"  {name:32s} {value:8.4f} m     (limit {limit:g})  {'ok' if passed else 'FAIL'}")

    if u._cable_slide_joint_ids is not None:
        print("\n--- cables (slide 0 = full length, negative = shorter) ---")
        slide = robot.data.joint_pos[:, u._cable_slide_joint_ids][alive]
        tension = u._cable_tension[alive]
        print(f"  slide position: {slide.min().item():+.4f} .. {slide.max().item():+.4f} m")
        print(f"  tension from the joint reaction: mean {tension.mean().item():.2f} N, "
              f"min {tension.min().item():.2f}, max {tension.max().item():.2f} (1.5 N expected at rest)")
        taut = slide.min().item() > -cfg.cable_slack_tolerance
        ok &= taut and 0.5 < tension.mean().item() < 4.0
        print(f"  {'all cables taut':32s} {'ok' if taut else 'FAIL'}")

    if cfg.setpoint_max_speed is not None:
        print("\n--- setpoint speed limit, with every action at +10 along x for 2 s ---")
        push = {a: torch.zeros(args.num_envs, cfg.action_spaces[a], device=u.device) for a in cfg.possible_agents}
        for action in push.values():
            action[:, 0] = 10.0
        setpoint_speed, drone_speed, resets = 0.0, 0.0, 0
        with torch.inference_mode():
            for _ in range(int(2.0 / u.step_dt)):
                before = u.sp_pos.clone()
                _, _, term, _, _ = env.step(push)
                resets += int(term[cfg.possible_agents[0]].sum())
                # An environment that reset reseeds its setpoint on its first step: a jump, not motion.
                moved = u.episode_length_buf > 1
                setpoint_speed = max(setpoint_speed, ((u.sp_pos - before).norm(dim=-1)[moved] / u.step_dt).max().item())
                drone_speed = max(drone_speed, u.drone_linear_velocities.norm(dim=-1)[moved].max().item())
        passed = setpoint_speed <= cfg.setpoint_max_speed * 1.001
        ok &= passed
        print(f"  {'setpoint speed, max':32s} {setpoint_speed:8.3f} m/s   (limit {cfg.setpoint_max_speed:g})  {'ok' if passed else 'FAIL'}")
        print(f"  {'drone speed, max':32s} {drone_speed:8.3f} m/s   ({resets} episodes ended during the push)")
        if cfg.drone_speed_limit is not None:
            print(f"  (the task ends an episode above {cfg.drone_speed_limit} m/s; this push is the worst case, "
                  f"every setpoint driven at full speed the same way)")

    print("\n--- RAPTOR input clipping ---")
    big = torch.full((1, 3), 9.0, device=u.device)
    clipped = assemble_observation(big, torch.eye(3, device=u.device).unsqueeze(0), -big, big, torch.zeros(1, 4, device=u.device))
    position_ok = clipped[0, 0:3].abs().max().item() <= MAX_POSITION_ERROR
    velocity_ok = clipped[0, 12:15].abs().max().item() <= MAX_VELOCITY_ERROR
    ok &= position_ok and velocity_ok
    print(f"  position error clipped to +-{MAX_POSITION_ERROR} m      {'ok' if position_ok else 'FAIL'}")
    print(f"  velocity error clipped to +-{MAX_VELOCITY_ERROR} m/s    {'ok' if velocity_ok else 'FAIL'}")

    env.close()
    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    try:
        passed = main()
    except Exception:  # Kit swallows uncaught exceptions and still exits 0
        import traceback

        traceback.print_exc()
        passed = False
    if not passed:
        os._exit(1)  # simulation_app.close() ends the process itself, with status 0
    simulation_app.close()
