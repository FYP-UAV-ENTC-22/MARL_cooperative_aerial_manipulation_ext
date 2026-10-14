"""Signals that arrive late, and the per-episode draws every model here uses."""

from __future__ import annotations

import torch


def draw_uniform(low_high: tuple[float, float], shape: tuple[int, ...], device) -> torch.Tensor:
    """One value per element, uniform in [low, high]."""
    low, high = low_high
    return torch.rand(shape, device=device) * (high - low) + low


def draw_gaussian(std: float, shape: tuple[int, ...], device, limit: float | None = None) -> torch.Tensor:
    """One value per element, zero-mean Gaussian, optionally cut at +-limit."""
    value = torch.randn(shape, device=device) * std
    return value if limit is None else value.clamp(-limit, limit)


def to_steps(seconds: torch.Tensor, dt: float) -> torch.Tensor:
    """A delay in seconds as a whole number of physics steps."""
    return torch.round(seconds / dt).long().clamp(min=0)


class LagBuffer:
    """The last `length` samples of a signal, one pushed per physics step, read back with a lag.

    The signal has shape (num_envs, *shape). ``read(lag)`` takes a lag in steps of shape (num_envs,) or
    (num_envs, readers) and returns the sample pushed that many steps before the newest. With
    ``per_reader`` the signal's first own dimension is the reader (one stream per drone); without it every
    reader gets the same stream at its own lag (one payload, three listeners).
    """

    def __init__(self, length: int, num_envs: int, shape: tuple[int, ...], device, per_reader: bool = False):
        self.length = max(int(length), 1)
        self.data = torch.zeros(self.length, num_envs, *shape, device=device)
        self.head = 0
        self.per_reader = per_reader
        self._envs = torch.arange(num_envs, device=device)
        self._readers = torch.arange(shape[0], device=device) if per_reader else None

    def fill(self, env_ids, value: torch.Tensor) -> None:
        """Make the whole history of the given environments equal one sample (after a reset)."""
        self.data[:, env_ids] = value

    def push(self, value: torch.Tensor) -> None:
        self.head = (self.head + 1) % self.length
        self.data[self.head] = value

    def read(self, lag: torch.Tensor) -> torch.Tensor:
        index = (self.head - lag.clamp(0, self.length - 1)) % self.length
        if lag.dim() == 1:
            return self.data[index, self._envs]
        if self.per_reader:
            return self.data[index, self._envs[:, None], self._readers[None, :]]
        return self.data[index, self._envs[:, None]]
