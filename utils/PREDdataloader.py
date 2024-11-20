import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    """
    A custom dataset for loading pairs of images for prediction tasks.

    This dataset reads image filenames from a directory, sorts them, and pairs consecutive images.
    It applies optional transformations and returns the image pairs in the required format.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing data loading parameters.
        transform (Optional[Callable]): Optional transformation to be applied on the images.

    Attributes:
        folder_path (str): Directory containing the image files.
        filenames (List[str]): Sorted list of image filenames.
        transform (Optional[Callable]): Transformations to apply to the images.
    """

    def __init__(self, config: Dict[str, Any], transform: Optional[Callable] = None):
        # Extract data loading configuration
        dataloading_config = config.get('dataloading', {})
        required_keys = ['data_path']
        missing_keys = [key for key in required_keys if key not in dataloading_config]
        if missing_keys:
            raise KeyError(f"Missing required config keys in 'dataloading': {missing_keys}")

        self.folder_path: str = dataloading_config['data_path']
        self.transform = transform

        # Validate the image directory
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(f"Image directory not found: {self.folder_path}")

        # Get the list of image filenames
        filenames = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        if not filenames:
            raise ValueError(f"No image files found in directory: {self.folder_path}")

        # Sort filenames based on numerical order in filenames (adjust if needed)
        # Example filename format: 'image-1.png', 'image-2.png', etc.
        self.filenames = sorted(filenames, key=lambda x: int(''.join(filter(str.isdigit, x))))
        # Alternatively, adjust the key function based on your filename format

    def __len__(self) -> int:
        # Subtract 1 because we're pairing images
        return len(self.filenames) - 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve a pair of images by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            images (torch.Tensor): A tensor of shape (2, C, H, W) containing the image pair.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self)}.")

        # Load the current and next image to form a pair
        img_name_1 = self.filenames[idx]
        img_name_2 = self.filenames[idx + 1]

        img_path_1 = os.path.join(self.folder_path, img_name_1)
        img_path_2 = os.path.join(self.folder_path, img_name_2)

        # Load images
        try:
            image_1 = Image.open(img_path_1).convert("RGB")
            image_2 = Image.open(img_path_2).convert("RGB")
        except Exception as e:
            raise IOError(f"Error opening images '{img_path_1}' and '{img_path_2}': {e}")

        # Apply transformations
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # Stack images to have them in the format (2, C, H, W)
        images = torch.stack([image_1, image_2])

        return images