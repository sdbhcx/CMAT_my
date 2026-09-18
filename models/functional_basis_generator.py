"""Query-independent functional basis generation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class FunctionalBasisGenerator(nn.Module):
    """Generate point-wise functional basis maps from point features only."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_basis: int = 8):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_basis <= 0:
            raise ValueError(f"num_basis must be positive, got {num_basis}")

        self.in_dim = in_dim
        self.num_basis = num_basis
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_basis),
        )

    def forward(self, point_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            point_features: Query-independent point features with shape ``[B, N, D]``.

        Returns:
            A tuple ``(basis_logits, basis_maps)``, both with shape ``[B, N, K]``.
        """
        if point_features.ndim != 3:
            raise ValueError(
                "point_features must have shape [B, N, D], "
                f"got {tuple(point_features.shape)}"
            )
        if point_features.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected point feature dimension {self.in_dim}, "
                f"got {point_features.shape[-1]}"
            )

        basis_logits = self.head(point_features)
        basis_maps = torch.sigmoid(basis_logits)
        return basis_logits, basis_maps


def pool_basis_descriptors(
    point_features: torch.Tensor,
    basis_maps: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pool one point-feature descriptor for each functional basis."""
    if point_features.ndim != 3:
        raise ValueError(
            "point_features must have shape [B, N, D], "
            f"got {tuple(point_features.shape)}"
        )
    if basis_maps.ndim != 3:
        raise ValueError(
            "basis_maps must have shape [B, N, K], "
            f"got {tuple(basis_maps.shape)}"
        )
    if point_features.shape[:2] != basis_maps.shape[:2]:
        raise ValueError(
            "point_features and basis_maps must share [B, N], got "
            f"{tuple(point_features.shape[:2])} and {tuple(basis_maps.shape[:2])}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    normalizer = basis_maps.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = basis_maps / normalizer
    return torch.einsum("bnk,bnd->bkd", weights, point_features)
