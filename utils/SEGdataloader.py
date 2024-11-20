import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    """
    A custom dataset for loading and preprocessing images with optional transformations.

    This dataset loads images from a specified directory, resizes them to a target size,
    and provides an option to apply additional transformations.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing data loading parameters.
        transform (Optional[Callable]): Optional transformation to be applied on the images.

    Attributes:
        root_dir (str): The root directory containing image files.
        resize_size (Tuple[int, int]): Target size to resize images to (width, height).
        transform (Optional[Callable]): Transformations to apply to the images.
        image_list (List[str]): List of image filenames in the root directory.
    """

    def __init__(self, config: Dict[str, Any], transform: Optional[Callable] = None):
        dataloading_config = config.get('dataloading', {})
        required_keys = ['data_path', 'resize_size']
        missing_keys = [key for key in required_keys if key not in dataloading_config]
        if missing_keys:
            raise KeyError(f"Missing required config keys in 'dataloading': {missing_keys}")

        self.root_dir: str = dataloading_config['data_path']
        self.resize_size: Tuple[int, int] = tuple(dataloading_config['resize_size'])
        self.transform = transform

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        self.image_list: List[str] = os.listdir(self.root_dir)
        if not self.image_list:
            raise ValueError(f"No images found in directory: {self.root_dir}")

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve an image by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Any: The image after applying resizing and optional transformations.

        Raises:
            IndexError: If the index is out of range.
            IOError: If there is an error opening the image file.
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if idx < 0 or idx >= len(self.image_list):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self)}.")

        image_name = os.path.join(self.root_dir, self.image_list[idx])

        try:
            image = Image.open(image_name)
        except Exception as e:
            raise IOError(f"Error opening image '{image_name}': {e}")

        # Resize the image
        image = image.resize(self.resize_size)

        # Convert RGBA images to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Apply optional transformations
        if self.transform:
            image = self.transform(image)

        return image, image_name