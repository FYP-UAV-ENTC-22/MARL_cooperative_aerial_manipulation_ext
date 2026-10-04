#!/usr/bin/env python3
"""Export a trained MAPPO actor to ONNX for the onboard policy runner.

A skrl checkpoint holds, per agent, the actor, the critic, the optimisers and the running input scaler.
Only the actor and its scaler fly: this writes, for each agent, one self-contained network

    observation (1, obs_dim)  ->  clamp((obs - mean) / (sqrt(var) + 1e-8), -5, 5)  ->  MLP  ->  action (1, act_dim)

so the drone feeds it the raw observation and gets the mean action, exactly what ``play.py`` uses.
The actors share weights during training but each agent keeps its own scaler, so there is one file per
agent; drone with one-hot slot i loads ``policy_falcon<i + 1>.onnx``.

Each file comes with ``<name>_test.npz``: random observations drawn around the scaler's mean and the
PyTorch outputs for them, for checking any other runtime against. Needs no Isaac Sim::

    ~/isaacsim/python.sh scripts/tools/export_policy_onnx.py logs/skrl/<experiment>/<run>/checkpoints/best_agent.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCALER_EPSILON = 1e-8  # skrl RunningStandardScaler defaults
SCALER_CLIP = 5.0


class Actor(nn.Module):
    """The scaler and the actor MLP of one agent, as skrl evaluates them in eval mode."""

    def __init__(self, policy: dict, scaler: dict):
        super().__init__()
        self.register_buffer("mean", scaler["running_mean"].float())
        self.register_buffer("std", torch.sqrt(scaler["running_variance"].float()) + SCALER_EPSILON)

        # net_container is Linear, ELU, Linear, ELU, ..., Linear: the even indices are the layers.
        indices = sorted({int(k.split(".")[1]) for k in policy if k.startswith("net_container.")})
        layers = []
        for n, index in enumerate(indices):
            weight, bias = policy[f"net_container.{index}.weight"], policy[f"net_container.{index}.bias"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            if n < len(indices) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        scaled = torch.clamp((observation - self.mean) / self.std, -SCALER_CLIP, SCALER_CLIP)
        return self.net(scaled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <run>/exported)")
    parser.add_argument("--samples", type=int, default=64, help="test vectors per agent")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out_dir = args.out or args.checkpoint.resolve().parents[1] / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(0)
    for agent, data in checkpoint.items():
        if "state_preprocessor" not in data:
            raise SystemExit(f"{agent}: the checkpoint has no input scaler; was it trained without one?")
        actor = Actor(data["policy"], data["state_preprocessor"]).eval()
        obs_dim, act_dim = actor.net[0].in_features, actor.net[-1].out_features

        path = out_dir / f"policy_{agent}.onnx"
        torch.onnx.export(
            actor,
            torch.zeros(1, obs_dim),
            str(path),
            input_names=["observation"],
            output_names=["action"],
            opset_version=17,
            dynamo=False,
        )

        # Test vectors spread like the data the scaler saw, a few of them far enough out to hit the clamp.
        spread = torch.randn(args.samples, obs_dim, generator=generator) * torch.linspace(0.5, 8.0, args.samples).unsqueeze(1)
        observations = actor.mean + spread * actor.std
        with torch.no_grad():
            actions = actor(observations)
        np.savez(path.with_name(f"policy_{agent}_test.npz"), observation=observations.numpy(), action=actions.numpy())

        line = f"{agent}: {obs_dim} -> {act_dim}, wrote {path}"
        try:
            import onnxruntime

            session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            ort_actions = np.concatenate(
                [session.run(["action"], {"observation": o[None]})[0] for o in observations.numpy()]
            )
            error = float(np.abs(ort_actions - actions.numpy()).max())
            line += f"; ONNX Runtime max error {error:.2e}"
            if error > 1e-4:
                raise SystemExit(line + "  MISMATCH")
        except ImportError:
            line += "; onnxruntime not installed, export not cross-checked"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
