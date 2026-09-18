"""Minimal functional-basis affordance head.

This module intentionally operates on already encoded point and query features. It
keeps the first implementation independent from dataset pairing and pretrained
encoder loading while defining the tensor contract needed for later integration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .basis_selector import BasisSelector
from .functional_basis_generator import (
    FunctionalBasisGenerator,
    pool_basis_descriptors,
)


def pool_prompt_tokens(
    prompt_features: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean-pool prompt token features from ``[B, A, T, D]`` to ``[B, A, D]``."""
    if prompt_features.ndim != 4:
        raise ValueError(
            "prompt_features must have shape [B, A, T, D], "
            f"got {tuple(prompt_features.shape)}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if attention_mask is None:
        return prompt_features.mean(dim=2)

    if attention_mask.shape != prompt_features.shape[:3]:
        raise ValueError(
            f"attention_mask must have shape {tuple(prompt_features.shape[:3])}, "
            f"got {tuple(attention_mask.shape)}"
        )
    weights = attention_mask.to(dtype=prompt_features.dtype).unsqueeze(-1)
    summed = (prompt_features * weights).sum(dim=2)
    normalizer = weights.sum(dim=2).clamp_min(eps)
    return summed / normalizer


class RelationalAffordanceHead(nn.Module):
    """Compose query-independent functional bases into affordance logits."""

    def __init__(
        self,
        point_dim: int,
        query_dim: int,
        basis_hidden_dim: int = 256,
        selector_hidden_dim: int = 256,
        num_basis: int = 8,
        selector_temperature: float = 0.07,
    ):
        super().__init__()
        self.basis_generator = FunctionalBasisGenerator(
            in_dim=point_dim,
            hidden_dim=basis_hidden_dim,
            num_basis=num_basis,
        )
        self.basis_selector = BasisSelector(
            query_dim=query_dim,
            basis_dim=point_dim,
            hidden_dim=selector_hidden_dim,
            temperature=selector_temperature,
        )

    def forward(
        self,
        point_features: torch.Tensor,
        query_tokens: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            point_features: Query-independent point features ``[B, N, D]``.
            query_tokens: Pooled affordance queries ``[B, A, Dq]``.
            valid: Optional valid-affordance mask ``[B, A]``.
        """
        if point_features.shape[0] != query_tokens.shape[0]:
            raise ValueError("point_features and query_tokens must share batch size")

        basis_logits, basis_maps = self.basis_generator(point_features)
        basis_descriptors = pool_basis_descriptors(point_features, basis_maps)
        alpha = self.basis_selector(query_tokens, basis_descriptors, valid=valid)
        segmentation_logits = torch.einsum("bnk,bak->ban", basis_logits, alpha)

        if valid is not None:
            segmentation_logits = segmentation_logits * valid.to(
                dtype=segmentation_logits.dtype
            ).unsqueeze(-1)

        return {
            "segmentation_logits": segmentation_logits,
            "basis_logits": basis_logits,
            "basis_maps": basis_maps,
            "basis_descriptors": basis_descriptors,
            "alpha": alpha,
            "query_tokens": query_tokens,
            "point_features": point_features,
        }
