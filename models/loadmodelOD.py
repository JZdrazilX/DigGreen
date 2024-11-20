import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Dict, Any, Tuple, List


def load_model_od(config: Dict[str, Any]) -> FasterRCNN:
    """
    Load a pre-trained Faster R-CNN model with a specified backbone and RPN configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters.

    Returns:
        FasterRCNN: The loaded Faster R-CNN model ready for evaluation.

    Raises:
        KeyError: If required keys are missing in the configuration.
        RuntimeError: If there is an error loading the model weights.
    """
    # Extract model configuration
    model_config = config.get('model', {})
    required_keys = [
        'path_to_model', 'num_classes', 'num_feature_maps',
        'sizes', 'aspect_ratios', 'backbone'
    ]
    missing_keys = [key for key in required_keys if key not in model_config]
    if missing_keys:
        raise KeyError(f"Missing required config keys in 'model': {missing_keys}")

    path_to_model: str = model_config['path_to_model']
    num_classes: int = model_config['num_classes']
    num_feature_maps: int = model_config['num_feature_maps']
    sizes: Tuple[int, ...] = tuple(model_config['sizes'])
    aspect_ratios: Tuple[float, ...] = tuple(model_config['aspect_ratios'])
    backbone_name: str = model_config['backbone']

    # Load the backbone model
    backbone = resnet_fpn_backbone(backbone_name, pretrained=True)

    # Create the anchor generator
    anchor_sizes: Tuple[Tuple[int, ...], ...] = (sizes,) * num_feature_maps
    anchor_aspect_ratios: Tuple[Tuple[float, ...], ...] = (aspect_ratios,) * num_feature_maps
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=anchor_aspect_ratios
    )

    # Initialize the Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    # Load the trained model weights
    try:
        state_dict = torch.load(path_to_model, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading model weights from '{path_to_model}': {e}")

    model.eval()
    return model
    