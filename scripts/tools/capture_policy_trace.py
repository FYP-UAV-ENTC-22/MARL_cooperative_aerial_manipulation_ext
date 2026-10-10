"""Record one episode of a trained policy as a reference for code that has to reproduce the pipeline elsewhere.

The onboard policy runner and the castor-main simulation test rebuild three things outside this
environment: the observation, the policy's action, and the setpoint integration. This writes what each
of them consumed and produced, step by step, so they can be checked against it:

    payload_pos, payload_quat          payload pose, world frame (quaternion w x y z)
    drone_pos, drone_rot               each drone's position and rotation matrix, world frame
    drone_lin_vel, drone_ang_vel       world frame
    goal_pos, goal_quat                the commanded payload pose
    obs                                the observation each agent was given, (steps, agents, obs_dim)
    action                             the mean action each agent returned for it
    sp_pos_before, sp_pos, sp_vel      the position setpoint before and after integrating that action
    sp_yaw                             the yaw setpoint

Row k of every array belongs to the same policy step. The episode is cut at the first reset.

    ~/isaacsim/python.sh scripts/tools/capture_policy_trace.py \\
        --task Isaac-castor-payload-decentralized-hovering-v0 \\
        --checkpoint logs/skrl/<experiment>/<run>/checkpoints/best_agent.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-castor-payload-decentralized-hovering-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=Path, default=None, help="default: <run>/exported/policy_trace.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app
sys.stdout.reconfigure(line_buffering=True)  # SimulationApp.close() drops buffered output

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import MARL_mav_carry_ext.tasks  # noqa: F401, E402  -- registers the task


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    # The trace is a reference for arithmetic: the observation must be the recorded state, exactly. Tasks that give
    # the policy a noisy, delayed state are recorded without either.
    for name in ("drone_pos_noise_std", "drone_rot_noise_std", "drone_lin_vel_noise_std", "drone_ang_vel_noise_std",
                 "payload_pos_noise_std", "payload_rot_noise_std"):
        setattr(env_cfg, name, 0.0)
    env_cfg.obs_delay_steps_range = (0, 0)
    env_cfg.payload_obs_delay_steps = 0
    agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
    env_cfg.seed = agent_cfg["seed"]
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    agents = list(u.cfg.possible_agents)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    runner.agent.load(str(args.checkpoint))
    runner.agent.set_running_mode("eval")

    rows = {k: [] for k in ("payload_pos", "payload_quat", "drone_pos", "drone_rot", "drone_lin_vel", "drone_ang_vel",
                            "goal_pos", "goal_quat", "obs", "action", "sp_pos_before", "sp_pos", "sp_vel", "sp_yaw")}

    def cpu(x):
        return x[0].detach().cpu().numpy().copy()

    obs, _ = env.reset()
    with torch.inference_mode():
        for step in range(args.steps):
            out = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = {a: out[-1][a].get("mean_actions", out[0][a]) for a in agents}
            # what the policy saw: the buffers the observation was built from
            rows["payload_pos"].append(cpu(u.load_position))
            rows["payload_quat"].append(cpu(u.load_orientation))
            rows["drone_pos"].append(cpu(u.drone_positions))
            rows["drone_rot"].append(cpu(u.drone_rot_matrices))
            rows["drone_lin_vel"].append(cpu(u.drone_linear_velocities))
            rows["drone_ang_vel"].append(cpu(u.drone_angular_velocities))
            rows["goal_pos"].append(cpu(u.pose_command_w[:, :3]))
            rows["goal_quat"].append(cpu(u.pose_command_w[:, 3:7]))
            rows["obs"].append(np.stack([cpu(obs[a]) for a in agents]))
            rows["action"].append(np.stack([cpu(actions[a]) for a in agents]))
            # On the first step the environment seeds the setpoint from the drone positions before integrating.
            rows["sp_pos_before"].append(cpu(u.drone_positions) if bool(u._sp_seed_pending[0]) else cpu(u.sp_pos))
            obs, _, terminated, truncated, _ = env.step(actions)
            rows["sp_pos"].append(cpu(u.sp_pos))
            rows["sp_vel"].append(cpu(u.sp_vel))
            rows["sp_yaw"].append(cpu(u.sp_yaw))
            if any(bool(v.any()) for v in (*terminated.values(), *truncated.values())):
                print(f"episode ended at step {step}; the trace stops there")
                break

    out_path = args.out or args.checkpoint.resolve().parents[1] / "exported" / "policy_trace.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "task": args.task,
        "agents": agents,
        "policy_dt": float(u.step_dt),
        "history_len": int(u.cfg.history_len),
        "setpoint_step_scale": float(u.cfg.setpoint_step_scale),
        "setpoint_max_speed": u.cfg.setpoint_max_speed,
        "setpoint_leash": float(u.cfg.setpoint_leash),
    }
    np.savez(out_path, meta=np.array(str(meta)), **{k: np.array(v) for k, v in rows.items()})
    steps = len(rows["obs"])
    error = np.linalg.norm(rows["goal_pos"][-1] - rows["payload_pos"][-1])
    print(f"wrote {out_path}: {steps} steps, payload {error:.3f} m from the goal at the end")
    print(f"  {meta}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # Kit swallows uncaught exceptions and still exits 0
        import traceback

        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
