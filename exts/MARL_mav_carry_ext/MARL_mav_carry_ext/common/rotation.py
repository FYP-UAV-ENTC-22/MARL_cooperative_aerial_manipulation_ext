"""The few rotation helpers the models need, so they do not depend on Isaac Lab."""

from __future__ import annotations

import torch


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) of rotation vectors (..., 3), by Rodrigues' formula."""
    angle = rotvec.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    axis = rotvec / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*rotvec.shape[:-1], 3, 3)
    sin, cos = torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)
    eye = torch.eye(3, device=rotvec.device).expand_as(skew)
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def yaw_of(rotation: torch.Tensor) -> torch.Tensor:
    """Heading of body-to-world rotation matrices (..., 3, 3)."""
    return torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
