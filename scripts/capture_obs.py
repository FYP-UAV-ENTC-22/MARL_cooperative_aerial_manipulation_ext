"""Capture a real falcon1 observation + geometry from the env (ground truth for the bridge)."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-flycrane-payload-decentralized-hovering-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.num_envs = 1
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import MARL_mav_carry_ext.tasks  # noqa: register

env_cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
obs, _ = env.reset()
u = env.unwrapped

def npv(x):
    return np.round(x.detach().cpu().numpy().reshape(-1), 3)

print("\n================ GROUND TRUTH ================")
o1 = obs["falcon1"].detach().cpu().numpy().reshape(-1)
print("falcon1 obs shape:", o1.shape, "(45*history)")
f = o1[:45]
labels = [("load_pos", 0, 3), ("load_matrix", 3, 12), ("one_hot", 12, 15),
          ("drone_pos", 15, 18), ("drone_matrix", 18, 27), ("drone_lin_vel", 27, 30),
          ("drone_ang_vel", 30, 33), ("goal_pos_error", 33, 36), ("diff_matrix", 36, 45)]
for name, a, b in labels:
    print(f"  {name:16s} = {f[a:b].round(3).tolist()}")

# raw geometry
print("\n--- raw world geometry (all 3 drones) ---")
print("load_position       :", npv(u.load_position))
print("drone_positions     :", u.drone_positions.detach().cpu().numpy().reshape(3, 3).round(3).tolist())
for i in range(3):
    off = (u.drone_positions[0, i] - u.load_position[0]).detach().cpu().numpy().round(3)
    print(f"  falcon{i+1} pos - load = {off.tolist()}   (this is the corner offset)")
print("goal (pose_command_w):", npv(u.pose_command_w))

# step a few times with zero action to see obs evolution
print("\n--- 3 zero-action steps: goal_pos_error ---")
za = {a: torch.zeros((1, u.cfg.action_spaces[a])) for a in u.cfg.possible_agents}
for k in range(3):
    obs, *_ = env.step(za)
    ge = obs["falcon1"].detach().cpu().numpy().reshape(-1)[33:36]
    print(f"  step {k}: goal_pos_error = {ge.round(3).tolist()}")

env.close()
app.close()
