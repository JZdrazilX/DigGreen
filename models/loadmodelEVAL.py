import torch
import torch.nn as nn
import timm
from typing import Dict, Any


def load_model_evaluation(config: Dict[str, Any]) -> nn.Module:
    """
    Load an evaluation (classification) model with a specified architecture and custom configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters.

    Returns:
        nn.Module: The loaded classification model ready for evaluation.

    Raises:
        KeyError: If required keys are missing in the configuration.
        RuntimeError: If there is an error loading the model weights.
        AttributeError: If the model does not have a classifier layer that can be modified.
    """
    # Extract model configuration
    model_config = config.get('model', {})
    required_keys = ['path_to_model', 'num_outputs', 'model_name', 'pretrained']
    missing_keys = [key for key in required_keys if key not in model_config]
    if missing_keys:
        raise KeyError(f"Missing required config keys in 'model': {missing_keys}")

    path_to_model: str = model_config['path_to_model']
    num_outputs: int = model_config['num_outputs']
    model_name: str = model_config['model_name']
    pretrained: bool = model_config['pretrained']

    # Initialize the model using timm
    try:
        model = timm.create_model(model_name, pretrained=pretrained)
    except Exception as e:
        raise RuntimeError(f"Error creating model '{model_name}': {e}")

    # Modify the classifier to match the number of outputs
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_outputs)
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_outputs)
    else:
        # Handle other possible classifier attributes
        modified = False
        for attr_name in ['head', 'last_linear', 'output_layer']:
            if hasattr(model, attr_name) and isinstance(getattr(model, attr_name), nn.Linear):
                in_features = getattr(model, attr_name).in_features
                setattr(model, attr_name, nn.Linear(in_features, num_outputs))
                modified = True
                break
        if not modified:
            raise AttributeError("The model does not have a classifier layer that can be modified.")

    # Load the trained model weights
    try:
        state_dict = torch.load(path_to_model, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model weights from '{path_to_model}': {e}")

    model.eval()
    return model