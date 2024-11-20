import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    """
    A custom dataset for loading images and their corresponding labels from a CSV file.

    This dataset reads image filenames and labels from a CSV file, standardizes the labels
    using the statistics computed from the entire dataset, and provides an option to apply
    image transformations. It ensures that only images present in the specified image directory
    are included in the final dataset.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing data loading parameters.
        transform (Optional[Callable]): Optional transformation to be applied on the images.
        compute_stats (bool): Whether to compute and apply standardization to the labels.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image filenames and labels.
        img_dir (str): Directory containing the image files.
        transform (Optional[Callable]): Transformations to apply to the images.
        scalers (Dict[str, StandardScaler]): Dictionary of scalers for each label column.
        column_names (List[str]): List of label column names.
    """

    def __init__(self, config: Dict[str, Any], transform: Optional[Callable] = None, compute_stats: bool = True):
        # Extract data loading configuration
        dataloading_config = config.get('dataloading', {})
        required_keys = ['data_path_image', 'data_path_csv']
        missing_keys = [key for key in required_keys if key not in dataloading_config]
        if missing_keys:
            raise KeyError(f"Missing required config keys in 'dataloading': {missing_keys}")

        self.img_dir: str = dataloading_config['data_path_image']
        self.csv_path: str = dataloading_config['data_path_csv']
        self.transform = transform
        self.scalers: Dict[str, StandardScaler] = {}

        # Validate the image directory and CSV file
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load the DataFrame from CSV
        try:
            dataframe = pd.read_csv(self.csv_path)
        except Exception as e:
            raise IOError(f"Error reading CSV file '{self.csv_path}': {e}")

        # Ensure 'Image Name' column exists
        if 'Image Name' not in dataframe.columns:
            raise KeyError("The CSV file must contain an 'Image Name' column.")

        # Exclude 'Image Name' column to get label columns
        self.column_names = dataframe.columns.difference(['Image Name']).to_list()

        # Compute standardization for label columns on the entire dataset
        if compute_stats:
            for col in self.column_names:
                scaler = StandardScaler()
                # Fit the scaler on the entire column
                dataframe[col] = scaler.fit_transform(dataframe[[col]])
                self.scalers[col] = scaler

        # Filter the dataframe to include only images that exist in the image directory
        available_images = set(os.listdir(self.img_dir))
        available_images = {os.path.splitext(f)[0] for f in available_images}  # Remove extensions
        dataframe['Image Name'] = dataframe['Image Name'].astype(str)
        dataframe = dataframe[dataframe['Image Name'].isin(available_images)]

        if dataframe.empty:
            raise ValueError("No matching images found between the CSV file and the image directory.")

        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[Any, torch.FloatTensor]:
        """
        Retrieve an image and its labels by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Any, torch.FloatTensor]: A tuple containing the image and its labels as a tensor.

        Raises:
            IndexError: If the index is out of range.
            FileNotFoundError: If the image file does not exist.
            IOError: If there is an error opening the image file.
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if idx < 0 or idx >= len(self.dataframe):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self)}.")

        image_name = self.dataframe.iloc[idx]['Image Name']
        img_path = os.path.join(self.img_dir, f"{image_name}.png")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error opening image '{img_path}': {e}")

        # Resize the image to a standard size (e.g., 224x224)
        image = image.resize((224, 224))

        # Get labels and convert to float tensor
        labels = self.dataframe.iloc[idx][self.column_names].values.astype(float)
        labels_tensor = torch.FloatTensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels_tensor
