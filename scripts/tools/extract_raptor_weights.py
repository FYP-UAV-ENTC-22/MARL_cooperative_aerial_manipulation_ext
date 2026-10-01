"""Convert the RAPTOR foundation-policy checkpoint (rl_tools tar) into a plain .npz.

The PX4 module ships the policy as ``blob/policy.tar``: a directory tree of ``meta``
(plain text) and ``data`` (raw little-endian float32) entries written by
``rl_tools/persist/backends/tar``.  This pulls out the five parameter tensors plus the
reference input/output pair that PX4 uses as its boot-time self test, so the extension
can load the policy with no rl-tools dependency.

Usage:
    python scripts/tools/extract_raptor_weights.py \
        --tar ~/Documents/Repos/PX4-Autopilot/src/modules/mc_raptor/blob/policy.tar
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np

# name in the archive -> name in the npz
TENSORS = {
    "actor/layers/0/weights/parameters": "dense0_weight",
    "actor/layers/0/biases/parameters": "dense0_bias",
    "actor/layers/1/weights_input/parameters": "gru_weight_ih",
    "actor/layers/1/biases_input/parameters": "gru_bias_ih",
    "actor/layers/1/weights_hidden/parameters": "gru_weight_hh",
    "actor/layers/1/biases_hidden/parameters": "gru_bias_hh",
    "actor/layers/1/initial_hidden_state/parameters": "gru_h0",
    "actor/layers/2/weights/parameters": "dense2_weight",
    "actor/layers/2/biases/parameters": "dense2_bias",
    "example/input": "example_input",
    "example/output": "example_output",
}

EXPECTED_SHAPES = {
    "dense0_weight": (16, 22),
    "dense0_bias": (16,),
    "gru_weight_ih": (48, 16),
    "gru_bias_ih": (48,),
    "gru_weight_hh": (48, 16),
    "gru_bias_hh": (48,),
    "gru_h0": (16,),
    "dense2_weight": (4, 16),
    "dense2_bias": (4,),
    "example_input": (500, 2, 22),
    "example_output": (500, 2, 4),
}


def _read(tar: tarfile.TarFile, member: str) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise KeyError(f"'{member}' missing from the archive")
    return handle.read()


def _parse_meta(raw: bytes) -> dict[str, str]:
    meta = {}
    for line in raw.decode().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def extract(tar_path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with tarfile.open(tar_path) as tar:
        checkpoint = _parse_meta(_read(tar, "actor/meta")).get("checkpoint_name", "unknown")
        print(f"checkpoint: {checkpoint}")

        for member, name in TENSORS.items():
            meta = _parse_meta(_read(tar, f"{member}/meta"))
            if meta.get("dtype") != "float32":
                raise ValueError(f"{member}: expected float32, got {meta.get('dtype')}")
            shape = tuple(int(meta[f"dim_{i}"]) for i in range(int(meta["num_dims"])))
            array = np.frombuffer(_read(tar, f"{member}/data"), dtype="<f4").reshape(shape)
            out[name] = array.copy()

    for name, expected in EXPECTED_SHAPES.items():
        got = out[name].shape
        if got != expected:
            raise ValueError(f"{name}: expected shape {expected}, got {got}")
        print(f"  {name:16s} {str(got):14s} ok")

    return out


def main() -> None:
    default_tar = Path.home() / "Documents/Repos/PX4-Autopilot/src/modules/mc_raptor/blob/policy.tar"
    default_out = (
        Path(__file__).resolve().parents[2]
        / "exts/MARL_mav_carry_ext/MARL_mav_carry_ext/controllers/data/raptor_v1.npz"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, default=default_tar, help="path to policy.tar")
    parser.add_argument("--out", type=Path, default=default_out, help="destination .npz")
    args = parser.parse_args()

    tensors = extract(args.tar)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **tensors)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
