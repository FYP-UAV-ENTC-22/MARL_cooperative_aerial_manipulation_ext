"""The payload's pose on its way to each drone: a direct Wi-Fi link per drone.

Each drone has its own delay for the episode, each packet a little jitter, and now and then one arrives late: the
drone then steps on the previous pose.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..common import LagBuffer, draw_uniform, to_steps


@dataclass
class WifiLinkCfg:
    delay_s: tuple[float, float] = (0.005, 0.030)  # per drone per episode
    jitter_s: float = 0.003  # std, per read
    late_probability: float = 0.01  # share of reads that find the newest packet missing
    late_extra_s: float = 0.1  # how late such a packet can be


class WifiLink:
    POSE_DIM = 3 + 9  # position, rotation row-major

    def __init__(self, cfg: WifiLinkCfg, num_envs: int, num_drones: int, physics_dt: float, device):
        self.cfg, self.dt, self.device = cfg, physics_dt, device
        self.shape = (num_envs, num_drones)
        longest = cfg.delay_s[1] + 4.0 * cfg.jitter_s + cfg.late_extra_s
        self._pose = LagBuffer(longest / physics_dt + 2, num_envs, (self.POSE_DIM,), device)
        self.delay = torch.zeros(self.shape, device=device)

    @staticmethod
    def pack(position, rotation) -> torch.Tensor:
        return torch.cat((position, rotation.flatten(-2)), dim=-1)

    def reset(self, env_ids, pose: torch.Tensor) -> None:
        self.delay[env_ids] = draw_uniform(self.cfg.delay_s, (len(env_ids), self.shape[1]), self.device)
        self._pose.fill(env_ids, pose[env_ids])

    def push(self, pose: torch.Tensor) -> None:
        self._pose.push(pose)

    def read(self):
        """The payload pose each drone has now: position (N, D, 3), rotation (N, D, 3, 3)."""
        cfg = self.cfg
        delay = self.delay + torch.randn(self.shape, device=self.device) * cfg.jitter_s
        late = torch.rand(self.shape, device=self.device) < cfg.late_probability
        delay = delay + late * torch.rand(self.shape, device=self.device) * cfg.late_extra_s
        p = self._pose.read(to_steps(delay, self.dt))
        return p[..., 0:3], p[..., 3:12].unflatten(-1, (3, 3))
