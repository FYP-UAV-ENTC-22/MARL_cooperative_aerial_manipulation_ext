"""Calibration rig for the RAPTOR inner loop, one drone at a time.

Run this before wiring RAPTOR into the MARL environment. Everything that can silently go
wrong in the hand-off -- rotor ordering, the rotor-speed range, the control rate -- produces
the same vague symptom once three drones and a swinging payload are in the loop. Here each
failure has exactly one cause.

Three checks:

``--check-rotor-order``
    RAPTOR emits [front-right, back-right, back-left, front-left]. Reads the rotor body
    poses out of the articulation and prints the permutation to put in
    ``raptor_rotor_permutation``. Run this first -- a wrong map reads as a tuning problem.

``--check-hover``
    Holds the setpoint at the spawn pose and reports steady-state error. Also prints the
    mean motor command: it should sit near zero, because the env's initial rotor speed of
    about 1400 rad/s maps back through the action formula to roughly -0.06.

``--check-step``
    Steps the setpoint by ``--step-size`` along each axis in turn and reports rise time,
    overshoot and settling, which is where a double motor lag would show up.

Example::

    ./isaaclab.sh -p scripts/MARL_mav_carry/testing_scripts/raptor_single_drone.py \
        --check-rotor-order --check-hover --check-step
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--check-rotor-order", action="store_true", help="derive the rotor permutation")
parser.add_argument("--check-hover", action="store_true", help="hold the spawn pose and report drift")
parser.add_argument("--check-step", action="store_true", help="step response on each axis")
parser.add_argument(
    "--hover-seconds", type=float, default=15.0, help="kept under episode_length_s to avoid a reset"
)
parser.add_argument("--step-size", type=float, default=0.4, help="step magnitude in metres")
parser.add_argument("--drone", type=int, default=0, help="which falcon to exercise")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.num_envs = 1
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import MARL_mav_carry_ext.tasks  # noqa: F401, E402  -- registers the task
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# RAPTOR's own output order, for reporting.
RAPTOR_ROTOR_NAMES = ["front-right", "back-right", "back-left", "front-left"]


def quadrant(offset: torch.Tensor) -> str:
    """Classify a body-frame rotor offset into RAPTOR's naming."""
    front = offset[0] > 0
    right = offset[1] < 0  # FLU: +y is left
    return f"{'front' if front else 'back'}-{'right' if right else 'left'}"


def azimuth_deg(offset: torch.Tensor) -> float:
    """Angle of a body-frame rotor offset about the body z axis, in degrees."""
    return math.degrees(math.atan2(float(offset[1]), float(offset[0])))


def check_rotor_order(env, drone: int) -> None:
    """Derive the permutation from RAPTOR's output order to the articulation's rotor order."""
    robot = env.robot
    base_index = env._falcon_idx[drone]
    rotor_indices = env._falcon_rotor_idx[drone * 4 : drone * 4 + 4]
    rotor_names = [robot.body_names[i] for i in rotor_indices]

    base_position = robot.data.body_com_state_w[0, base_index, :3]
    base_quat = robot.data.body_com_state_w[0, base_index, 3:7]
    # The drones hang off cables and sit tilted at spawn, so world-frame offsets carry that
    # tilt. Rotate into the body frame before reasoning about the airframe layout.
    rotation = matrix_from_quat(base_quat.unsqueeze(0))[0]

    print("\n--- rotor geometry (body frame) ---")
    offsets, azimuths = [], []
    for slot, (index, name) in enumerate(zip(rotor_indices, rotor_names)):
        world_offset = robot.data.body_com_state_w[0, index, :3] - base_position
        offset = rotation.T @ world_offset
        offsets.append(offset)
        azimuths.append(azimuth_deg(offset))
        rounded = [round(v, 4) for v in offset.tolist()]
        print(f"  slot {slot}  {name:28s} offset = {rounded}  azimuth = {azimuths[-1]:+7.1f} deg")

    spacing = sorted(a % 360 for a in azimuths)
    gaps = [(spacing[(i + 1) % 4] - spacing[i]) % 360 for i in range(4)]
    print(f"\n  angular gaps between arms: {[round(g, 1) for g in gaps]} deg")
    if max(abs(g - 90.0) for g in gaps) > 12.0:
        print("  !! arms are not 90 deg apart -- this is not a symmetric quad, inspect by hand")
        return

    # Where the arms sit relative to the body x axis. 45 deg means an X airframe, which is
    # what RAPTOR expects; 0 deg means a plus airframe, whose roll and pitch axes are rotated
    # 45 deg from RAPTOR's assumption.
    arm_angle = sum(a % 90.0 for a in azimuths) / 4.0
    yaw_offset = 45.0 - arm_angle
    print(f"  arms sit at {arm_angle:.1f} deg (mod 90) from the body x axis")

    if abs(yaw_offset) < 5.0:
        print("  -> X airframe, matching RAPTOR's layout. No yaw offset needed.")
        yaw_offset = 0.0
    else:
        print(f"  -> plus airframe. RAPTOR assumes X, so the frame it sees must be rotated")
        print(f"     by {yaw_offset:+.1f} deg about body z, otherwise roll and pitch are swapped.")

    # Classify in the virtual frame, the one RAPTOR actually sees. The env sets
    # R_body->virtual = Rz(yaw_offset), so a point's virtual-frame coordinates are
    # Rz(-yaw_offset) applied to its body-frame coordinates -- note the negation.
    theta = -math.radians(yaw_offset)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    labels = []
    for offset in offsets:
        x, y = float(offset[0]), float(offset[1])
        rotated = torch.tensor([x * cos_t - y * sin_t, x * sin_t + y * cos_t])
        labels.append(quadrant(rotated))

    if sorted(labels) != sorted(RAPTOR_ROTOR_NAMES):
        print(f"\n  !! rotors did not classify into four distinct quadrants: {labels}")
        return

    # The config entry is consumed as a gather -- action[..., perm] -- so entry j must name
    # the RAPTOR output that drives articulation slot j.
    permutation = tuple(RAPTOR_ROTOR_NAMES.index(label) for label in labels)
    print("\n  articulation slot <- RAPTOR output:")
    for slot, raptor_index in enumerate(permutation):
        print(f"    slot {slot} <- output {raptor_index} ({RAPTOR_ROTOR_NAMES[raptor_index]})")

    print("\n  --- put these in marl_hover_env_cfg.py ---")
    print(f"    raptor_rotor_permutation = {permutation}")
    print(f"    raptor_yaw_offset_deg = {yaw_offset:.1f}")
    matches = permutation == tuple(env.cfg.raptor_rotor_permutation) and abs(
        yaw_offset - env.cfg.raptor_yaw_offset_deg
    ) < 0.1
    print("  config already matches" if matches else "  !! config differs -- update it")

    print("\n  --- rotor spin directions ---")
    print("  RotorMotor.direction is applied in articulation-slot order; confirm the two")
    print("  rotors on each diagonal spin the same way:")
    for slot, label in enumerate(labels):
        print(f"    slot {slot}  {label}")


def _hold(env, steps: int, increment: torch.Tensor):
    """Drive the env with a fixed per-step setpoint increment, recording drone position.

    Stops early if the episode ends. Measuring across a reset compares positions either side
    of a teleport, which reads as a large spurious drift.
    """
    actions = {agent: increment.clone() for agent in env.cfg.possible_agents}
    track, ended = [], False
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(actions)
        track.append(env.drone_positions[0].clone())
        if any(bool(terminated[a][0]) or bool(truncated[a][0]) for a in env.cfg.possible_agents):
            ended = True
            break
    return torch.stack(track), ended


def _yaw_state(env, drone: int) -> tuple[float, float]:
    """Yaw angle (deg) and body yaw rate (deg/s) of one drone."""
    quat = env.drone_orientations[0, drone]
    rotation = matrix_from_quat(quat.unsqueeze(0))[0]
    yaw = math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    body_rate = rotation.T @ env.drone_angular_velocities[0, drone]
    return yaw, math.degrees(float(body_rate[2]))


def check_hover(env, drone: int, seconds: float) -> None:
    """Zero action: the setpoint stays put and the drone should hold it."""
    budget = 0.9 * env.cfg.episode_length_s
    if seconds > budget:
        print(f"  (capping hover to {budget:.1f} s: episode_length_s is {env.cfg.episode_length_s})")
        seconds = budget
    steps = int(seconds / env.step_dt)
    zero = torch.zeros(1, 3, device=env.device)
    start = env.drone_positions[0, drone].clone()

    yaw_start, _ = _yaw_state(env, drone)
    track, ended = _hold(env, steps, zero)
    yaw_end, yaw_rate = _yaw_state(env, drone)
    actual_seconds = len(track) * env.step_dt
    error = track[:, drone] - start
    settled = error[steps // 2 :]  # ignore the initial transient

    print(f"\n--- hover hold, {actual_seconds:.1f} s, falcon{drone + 1} ---")
    if ended:
        print(f"  !! episode ended early at {actual_seconds:.1f} s -- a termination fired.")
        print("     Check Episode_Termination causes; the numbers below cover only this span.")
    print(f"  final error       {[round(v, 4) for v in error[-1].tolist()]}")
    print(f"  steady-state rms  {settled.norm(dim=-1).mean().item():.4f} m")
    print(f"  max excursion     {error.norm(dim=-1).max().item():.4f} m")
    # Report what the command implies physically. The absolute value is not predictable from
    # the env's initial_rpms -- those are a starting motor state, not the loaded equilibrium.
    command = env.raptor.previous_action[0, drone].mean().item()
    omega = (env.cfg.raptor_omega_max - env.cfg.raptor_omega_min) * (command + 1) / 2 + env.cfg.raptor_omega_min
    thrust = 1.562522e-06 * omega**2
    print(f"  mean motor cmd    {command:+.3f}  -> {omega:.0f} rad/s, {thrust:.2f} N/rotor")
    print(f"  implied mass      {12 * thrust / 9.8066:.2f} kg for the whole system")
    if thrust > env.cfg.max_thrust_pp:
        print(f"  !! exceeds max_thrust_pp ({env.cfg.max_thrust_pp} N) -- check the omega range")

    # The rotor geometry confirms which motor is where, but not the absolute sense of the
    # spin directions. If RotorMotor.direction is inverted relative to RAPTOR's convention,
    # yaw control becomes positive feedback and the drone spins up.
    yaw_drift = (yaw_end - yaw_start + 180.0) % 360.0 - 180.0
    print(f"  yaw drift         {yaw_drift:+7.1f} deg over {actual_seconds:.1f} s")
    print(f"  final yaw rate    {yaw_rate:+7.1f} deg/s")
    if abs(yaw_rate) > 30.0:
        print("  !! yaw is running away -- RotorMotor.direction is likely inverted relative")
        print("     to RAPTOR's convention. Negate it and re-run.")


def check_step(env, drone: int, size: float) -> None:
    """Step the setpoint along each axis and report the transient."""
    print(f"\n--- step response, {size} m, falcon{drone + 1} ---")
    settle_steps = int(4.0 / env.step_dt)

    for axis, name in enumerate("xyz"):
        start = env.drone_positions[0, drone].clone()
        target = start.clone()
        target[axis] += size

        # One large increment moves the setpoint; the leash caps how far ahead it can sit.
        increment = torch.zeros(1, 3, device=env.device)
        increment[0, axis] = size / env.cfg.setpoint_step_scale
        env.step({agent: increment.clone() for agent in env.cfg.possible_agents})

        track, ended = _hold(env, settle_steps, torch.zeros(1, 3, device=env.device))
        if ended:
            print(f"  {name}:  !! episode ended during the step -- result unreliable")
            env.reset()
            continue
        displacement = (track[:, drone, axis] - start[axis]) / size

        overshoot = (displacement.max().item() - 1.0) * 100.0
        risen = (displacement > 0.9).nonzero()
        rise_time = risen[0].item() * env.step_dt if len(risen) else float("nan")
        final_error = abs(track[-1, drone, axis].item() - target[axis].item())

        print(
            f"  {name}:  rise(90%) {rise_time:5.2f} s   overshoot {overshoot:+6.1f} %"
            f"   final error {final_error:.4f} m"
        )


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    if env.cfg.control_mode != "raptor":
        raise SystemExit(f"control_mode is '{env.cfg.control_mode}' -- set it to 'raptor' in the env cfg")

    env.reset()

    if args.check_rotor_order:
        check_rotor_order(env, args.drone)
    if args.check_hover:
        env.reset()
        check_hover(env, args.drone, args.hover_seconds)
    if args.check_step:
        env.reset()
        check_step(env, args.drone, args.step_size)
    if not (args.check_rotor_order or args.check_hover or args.check_step):
        print("nothing to do -- pass --check-rotor-order, --check-hover and/or --check-step")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
