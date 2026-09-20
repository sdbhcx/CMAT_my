"""Model and loss factory for LAS and functional-basis training."""

from .las_model import create_las_model
from .functional_basis_model import create_functional_basis_model

def create_model(config):
    """
    Create model based on configuration
    
    Args:
        config: Configuration dictionary containing model specifications
    
    Returns:
        model: Model instance
    
    Raises:
        ValueError: If model type is not supported
    """
    model_name = config['model']['name'].lower()
    
    if model_name == 'las':
        return create_las_model(config)
    if model_name == 'fbd_afford':
        return create_functional_basis_model(config)

    raise ValueError(
        f"Unsupported model type: {model_name}. Supported models: {get_supported_models()}"
    )

def get_loss_function(config):
    """
    Get appropriate loss function based on model type
    
    Args:
        config: Configuration dictionary
    
    Returns:
        loss_fn: Loss function instance
    """
    model_name = config['model']['name'].lower()
    
    if model_name == 'las':
        from .las_model import LASLoss
        return LASLoss(
            focal_alpha=config['loss']['focal_alpha'],
            focal_gamma=config['loss']['focal_gamma'],
            focal_weight=config['loss']['focal_weight'],
            dice_weight=config['loss']['dice_weight']
        )
    if model_name == 'fbd_afford':
        from losses import FunctionalBasisLoss

        loss_config = config['loss']
        unsupported_losses = (
            'relation',
            'coefficient_relation',
            'basis_diversity',
            'coefficient_sparsity',
            'cross_object_correspondence',
        )
        enabled_unsupported = [
            name for name in unsupported_losses if float(loss_config.get(name, 0.0)) != 0.0
        ]
        if enabled_unsupported:
            raise ValueError(
                "The Stage-A MVP does not implement these losses yet: "
                + ", ".join(enabled_unsupported)
            )
        return FunctionalBasisLoss(
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            segmentation_weight=loss_config.get('segmentation', 1.0),
            union_weight=loss_config.get('union', 0.2),
            segmentation_loss=loss_config.get('segmentation_loss', 'fbd'),
            focal_weight=loss_config.get('focal_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
        )

    raise ValueError(f"Unsupported model type: {model_name}")

def get_supported_models():
    """
    Get list of supported model types
    
    Returns:
        list: List of supported model names
    """
    return ['las', 'fbd_afford']
