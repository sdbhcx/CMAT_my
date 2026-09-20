"""Stage-A loss for functional-basis affordance training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dice_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.reshape(probabilities.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probabilities * targets).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)
    return (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


class FunctionalBasisLoss(nn.Module):
    """Compute valid-aware focal/dice segmentation and basis-union coverage."""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        segmentation_weight: float = 1.0,
        union_weight: float = 0.2,
        segmentation_loss: str = "fbd",
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be in [0, 1]")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if segmentation_weight < 0 or union_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if segmentation_loss not in ("fbd", "las"):
            raise ValueError("segmentation_loss must be 'fbd' or 'las'")
        if focal_weight < 0 or dice_weight < 0:
            raise ValueError("focal_weight and dice_weight must be non-negative")
        self.segmentation_loss = segmentation_loss
        self.las_segmentation = None
        if segmentation_loss == "las":
            # Reuse the actual LAS implementation, including soft-target focal
            # and batch-global foreground/background Dice reductions.
            from models.las_model import LASLoss
            self.las_segmentation = LASLoss(
                focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                focal_weight=focal_weight, dice_weight=dice_weight)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.segmentation_weight = segmentation_weight
        self.union_weight = union_weight

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        binary_cross_entropy = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        probability_correct = probabilities * targets + (1.0 - probabilities) * (
            1.0 - targets
        )
        alpha = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (
            1.0 - targets
        )
        return (
            alpha
            * (1.0 - probability_correct).pow(self.focal_gamma)
            * binary_cross_entropy
        ).mean()

    def forward(self, outputs, batch):
        logits = outputs["segmentation_logits"]
        targets = batch["masks"].to(dtype=logits.dtype).clamp(0.0, 1.0)
        valid = batch["valid"].bool()
        basis_maps = outputs["basis_maps"]

        if logits.shape != targets.shape:
            raise ValueError(
                f"segmentation logits and masks must share shape, got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        if valid.shape != logits.shape[:2]:
            raise ValueError(
                f"valid must have shape {tuple(logits.shape[:2])}, got {tuple(valid.shape)}"
            )
        if basis_maps.shape[:2] != (logits.shape[0], logits.shape[2]):
            raise ValueError("basis_maps must have shape [B, N, K]")
        if not valid.any():
            raise ValueError("At least one valid affordance is required per batch")

        valid_logits = logits[valid]
        valid_targets = targets[valid]
        if self.las_segmentation is not None:
            segmentation, parts = self.las_segmentation(
                valid_logits.unsqueeze(-1), valid_targets.unsqueeze(-1))
            focal, dice = parts["focal_loss"], parts["dice_loss"]
        else:
            # Preserve the historical FBD objective, which did not consume
            # focal_weight/dice_weight from configuration.
            focal = self._focal_loss(valid_logits, valid_targets)
            dice = _dice_loss(torch.sigmoid(valid_logits), valid_targets)
            segmentation = focal + dice

        masked_targets = targets * valid.to(dtype=targets.dtype).unsqueeze(-1)
        target_union = masked_targets.amax(dim=1)
        basis_union = 1.0 - torch.prod(1.0 - basis_maps, dim=-1)
        union_bce = F.binary_cross_entropy(
            basis_union.clamp(1e-6, 1.0 - 1e-6), target_union
        )
        union_dice = _dice_loss(basis_union, target_union)
        union = union_bce + union_dice

        total = (
            self.segmentation_weight * segmentation + self.union_weight * union
        )
        return total, {
            "total_loss": total,
            "segmentation": segmentation,
            "focal": focal,
            "dice": dice,
            "union": union,
            "union_bce": union_bce,
            "union_dice": union_dice,
        }
