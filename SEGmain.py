import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import SEGdataloader, configloader
from models.loadmodelSEG import load_model_segmentation


def main():
    """
    Main function to perform image segmentation on a dataset of images,
    apply the predicted masks to the images, and save the masked images
    to the specified output directory.
    """
    # Load configuration
    path_to_config = '../configs/paramsSEG.yaml'
    config = configloader.load_config(path_to_config)

    # Set computation device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(config['dataloading']['resize_size']),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = SEGdataloader.CustomImageDataset(
        config=config,
        transform=transform,
    )

    print(f'Number of images: {len(dataset)}')

    # Create data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['dataloading']['batch_size'],
        shuffle=False,
    )

    # Load the pre-trained segmentation model
    model = load_model_segmentation(config)
    model.to(device)
    model.eval()

    # Output directory for saving masked images
    output_directory = config['datasaving']['output_folder']
    os.makedirs(output_directory, exist_ok=True)

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch_idx, (images, image_names) in enumerate(data_loader):
            images = images.to(device)

            # Perform inference
            outputs = model(images)

            # Get the predicted masks
            _, predicted_masks = torch.max(outputs, dim=1)

            # Process each image in the batch
            for idx in range(len(images)):
                # Get the original image
                original_image = images[idx].cpu().permute(1, 2, 0).numpy()
                original_image = (original_image * 255).astype(np.uint8)

                # Get the predicted mask
                mask = predicted_masks[idx].cpu().numpy()

                # Convert the mask to binary format
                binary_mask = (mask > 0).astype(np.uint8)

                # Apply the mask to the original image
                masked_image = original_image * np.expand_dims(binary_mask, axis=-1)

                # Corrected line: Extract the base filename
                image_name = os.path.splitext(os.path.basename(image_names[idx]))[0]
                output_path = os.path.join(output_directory, f"{image_name}_masked.png")

                # Save the masked image
                masked_image_pil = Image.fromarray(masked_image)
                masked_image_pil.save(output_path)

    print(f"Processing complete. Output images saved to: {output_directory}")


if __name__ == "__main__":
    main()
