"""Model package exports with lazy loading for optional encoder dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "LASModel",
    "LASLoss",
    "create_las_model",
    "create_model",
    "get_loss_function",
    "get_supported_models",
    "FunctionalBasisGenerator",
    "pool_basis_descriptors",
    "BasisSelector",
    "RelationalAffordanceHead",
    "pool_prompt_tokens",
    "FunctionalBasisAffordanceModel",
    "create_functional_basis_model",
]


if TYPE_CHECKING:
    from .basis_selector import BasisSelector
    from .functional_basis_generator import (
        FunctionalBasisGenerator,
        pool_basis_descriptors,
    )
    from .las_model import LASLoss, LASModel, create_las_model
    from .model_factory import create_model, get_loss_function, get_supported_models
    from .functional_basis_model import (
        FunctionalBasisAffordanceModel,
        create_functional_basis_model,
    )
    from .relational_affordance_model import (
        RelationalAffordanceHead,
        pool_prompt_tokens,
    )


def __getattr__(name: str):
    if name in {"LASModel", "LASLoss", "create_las_model"}:
        from . import las_model

        return getattr(las_model, name)
    if name in {"create_model", "get_loss_function", "get_supported_models"}:
        from . import model_factory

        return getattr(model_factory, name)
    if name in {"FunctionalBasisGenerator", "pool_basis_descriptors"}:
        from . import functional_basis_generator

        return getattr(functional_basis_generator, name)
    if name == "BasisSelector":
        from .basis_selector import BasisSelector

        return BasisSelector
    if name in {"RelationalAffordanceHead", "pool_prompt_tokens"}:
        from . import relational_affordance_model

        return getattr(relational_affordance_model, name)
    if name in {"FunctionalBasisAffordanceModel", "create_functional_basis_model"}:
        from . import functional_basis_model

        return getattr(functional_basis_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
