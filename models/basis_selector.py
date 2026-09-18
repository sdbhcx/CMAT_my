"""Object-conditioned selection over functional basis descriptors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisSelector(nn.Module):
    """Select object-specific functional bases for each affordance query."""

    def __init__(
        self,
        query_dim: int,
        basis_dim: int,
        hidden_dim: int = 256,
        temperature: float = 0.07,
        min_temperature: float = 0.02,
    ):
        super().__init__()
        for name, value in (
            ("query_dim", query_dim),
            ("basis_dim", basis_dim),
            ("hidden_dim", hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if min_temperature <= 0:
            raise ValueError(
                f"min_temperature must be positive, got {min_temperature}"
            )

        self.query_dim = query_dim
        self.basis_dim = basis_dim
        self.min_temperature = min_temperature
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.basis_proj = nn.Linear(basis_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(
        self,
        query: torch.Tensor,
        basis_descriptors: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Affordance queries with shape ``[B, A, Dq]``.
            basis_descriptors: Object basis descriptors with shape ``[B, K, D]``.
            valid: Optional valid-affordance mask with shape ``[B, A]``.

        Returns:
            Basis coefficients with shape ``[B, A, K]``. Invalid rows are zero.
        """
        if query.ndim != 3:
            raise ValueError(f"query must have shape [B, A, Dq], got {tuple(query.shape)}")
        if basis_descriptors.ndim != 3:
            raise ValueError(
                "basis_descriptors must have shape [B, K, D], "
                f"got {tuple(basis_descriptors.shape)}"
            )
        if query.shape[0] != basis_descriptors.shape[0]:
            raise ValueError("query and basis_descriptors must share the batch dimension")
        if query.shape[-1] != self.query_dim:
            raise ValueError(
                f"Expected query dimension {self.query_dim}, got {query.shape[-1]}"
            )
        if basis_descriptors.shape[-1] != self.basis_dim:
            raise ValueError(
                f"Expected basis dimension {self.basis_dim}, "
                f"got {basis_descriptors.shape[-1]}"
            )

        q = F.normalize(self.query_proj(query), dim=-1)
        d = F.normalize(self.basis_proj(basis_descriptors), dim=-1)
        scores = torch.einsum("bah,bkh->bak", q, d)
        alpha = torch.softmax(
            scores / self.temperature.clamp_min(self.min_temperature), dim=-1
        )

        if valid is not None:
            if valid.shape != query.shape[:2]:
                raise ValueError(
                    f"valid must have shape {tuple(query.shape[:2])}, "
                    f"got {tuple(valid.shape)}"
                )
            alpha = alpha * valid.to(dtype=alpha.dtype).unsqueeze(-1)

        return alpha
