import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    """
    A custom dataset for loading images with optional preprocessing transformations.

    This dataset loads images from a specified directory, applies a custom masking
    operation, and provides an option to apply additional transformations.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing data loading parameters.
        transform (Optional[Callable]): Optional transformation to be applied on the images.

    Attributes:
        root_dir (str): The root directory containing image files.
        transform (Optional[Callable]): Transformations to apply to the images.
        image_list (List[str]): List of image filenames in the root directory.
    """

    def __init__(self, config: Dict[str, Any], transform: Optional[Callable] = None):
        self.root_dir: str = config['dataloading']['data_path']
        self.transform = transform

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        self.image_list: List[str] = os.listdir(self.root_dir)
        if not self.image_list:
            raise ValueError(f"No images found in directory: {self.root_dir}")

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        """
        Retrieve an image and its filename by index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[Any, str]: A tuple containing the image and its filename.

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

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = self._apply_mask(image)

        if self.transform:
            image = self.transform(image)

        return image, self.image_list[idx]

    def _apply_mask(self, image: Image.Image) -> Image.Image:
        """
        Apply a custom mask to the image to composite it with a white background.

        Args:
            image (Image.Image): The original image.

        Returns:
            Image.Image: The image after applying the mask.
        """
        white_img = Image.new('RGB', image.size, 'white')
        mask = Image.new('L', image.size, 0)
        border = (340, 340, image.width, image.height - 340)
        mask.paste(255, border)
        masked_image = Image.composite(image, white_img, mask)
        return masked_image