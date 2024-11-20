import torch
import segmentation_models_pytorch as smp
from typing import Dict, Any


def load_model_segmentation(config: Dict[str, Any]) -> smp.DeepLabV3Plus:
    """
    Load a segmentation model with a pre-trained encoder and custom configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters.

    Returns:
        smp.DeepLabV3Plus: The loaded DeepLabV3+ segmentation model ready for evaluation.

    Raises:
        KeyError: If required keys are missing in the configuration.
        RuntimeError: If there is an error loading the model weights.
    """
    # Extract model configuration
    model_config = config.get('model', {})
    required_keys = ['path_to_model', 'encoder', 'encoder_weights', 'num_classes']
    missing_keys = [key for key in required_keys if key not in model_config]
    if missing_keys:
        raise KeyError(f"Missing required config keys in 'model': {missing_keys}")

    path_to_model: str = model_config['path_to_model']
    encoder_name: str = model_config['encoder']
    encoder_weights: str = model_config['encoder_weights']
    num_classes: int = model_config['num_classes']

    # Initialize the DeepLabV3+ segmentation model
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=num_classes,
    )

    # Load the trained model weights
    try:
        state_dict = torch.load(path_to_model, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model weights from '{path_to_model}': {e}")

    model.eval()
    return model