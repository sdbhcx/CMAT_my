"""End-to-end LAS integration for the functional-basis affordance head."""

from __future__ import annotations

import torch

from .las_model import LASModel
from .relational_affordance_model import (
    RelationalAffordanceHead,
    pool_prompt_tokens,
)


class FunctionalBasisAffordanceModel(LASModel):
    """Trainable visual-prompt LAS variant using a functional-basis decoder."""

    def __init__(self, config):
        super().__init__(config)
        if self.prompt_type != "visual":
            raise ValueError("The functional-basis MVP currently supports visual prompts only")

        model_config = config["model"]
        self.functional_basis_head = RelationalAffordanceHead(
            point_dim=self.unified_dim,
            query_dim=self.unified_dim,
            basis_hidden_dim=model_config.get("basis_hidden_dim", 256),
            selector_hidden_dim=model_config.get("selector_hidden_dim", 256),
            num_basis=model_config.get("num_basis", 8),
            selector_temperature=model_config.get("selector_temperature", 0.07),
        )

        # The original LAS decoder is not part of the FBD computation graph.
        # Removing it avoids unused parameters under DDP.
        self.co_attention_transformer = None
        self.segmentation_head = None
        self.point_type_embedding = None
        self.prompt_type_embedding = None

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(p.requires_grad for p in self.point_encoder.parameters()):
            self.point_encoder.eval()
        if not any(p.requires_grad for p in self.prompt_encoder.parameters()):
            self.prompt_encoder.eval()
        return self

    def forward(self, batch):
        points = batch["points"]
        images = batch.get("images")
        if images is None and "image" in batch:
            images = batch["image"].unsqueeze(1)
        if images is None or images.ndim != 5:
            shape = None if images is None else tuple(images.shape)
            raise ValueError(f"images must have shape [B, A, 3, H, W], got {shape}")
        if images.shape[0] != points.shape[0]:
            raise ValueError("points and images must share the batch dimension")

        batch_size, num_affordances = images.shape[:2]
        valid = batch.get("valid")
        if valid is None:
            valid = torch.ones(
                batch_size, num_affordances, dtype=torch.bool, device=images.device
            )

        point_features = self.encode_points(points)
        flat_images = images.reshape(-1, *images.shape[2:])
        prompt_features, prompt_attention_mask = self.encode_prompts(images=flat_images)
        token_count, feature_dim = prompt_features.shape[1:]
        prompt_features = prompt_features.reshape(
            batch_size, num_affordances, token_count, feature_dim
        )
        prompt_attention_mask = prompt_attention_mask.reshape(
            batch_size, num_affordances, token_count
        )
        query_tokens = pool_prompt_tokens(prompt_features, prompt_attention_mask)
        return self.functional_basis_head(point_features, query_tokens, valid=valid)


def create_functional_basis_model(config):
    """Create the trainable functional-basis affordance model."""
    return FunctionalBasisAffordanceModel(config)
