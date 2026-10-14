"""From RAPTOR's motor command to rotor thrust: the ESC's delay, the rotor's lag, the thrust it then makes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..common import LagBuffer, draw_uniform, to_steps


@dataclass
class EscCfg:
    # From RAPTOR's command to the rotor acting on it, drawn per drone per episode. On the aircraft this is the
    # output protocol and the ESC, a few ms. In castor-main's PX4 simulation it is 8 to 12 ms (the simulator applies
    # the command it already holds before asking PX4 for the next), and with the run 12 policy that delay is what
    # makes the drones shake there (scripts/tools/flight_stack_check.py --variant px4_sim). The range covers both.
    delay_s: tuple[float, float] = (0.0, 0.012)


class EscDelay:
    """Holds each drone's rotor-speed commands back by its ESC's delay."""

    def __init__(self, cfg: EscCfg, num_envs: int, num_drones: int, physics_dt: float, device):
        self.cfg, self.dt, self.device = cfg, physics_dt, device
        self.shape = (num_envs, num_drones)
        self._commands = LagBuffer(cfg.delay_s[1] / physics_dt + 2, num_envs, (num_drones, 4), device, per_reader=True)
        self.lag = torch.zeros(self.shape, dtype=torch.long, device=device)

    def reset(self, env_ids, rotor_speed: torch.Tensor) -> None:
        n, d = len(env_ids), self.shape[1]
        self.lag[env_ids] = to_steps(draw_uniform(self.cfg.delay_s, (n, d), self.device), self.dt)
        self._commands.fill(env_ids, rotor_speed[env_ids])

    def __call__(self, rotor_speed: torch.Tensor) -> torch.Tensor:
        self._commands.push(rotor_speed)
        return self._commands.read(self.lag)
