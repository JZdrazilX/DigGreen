import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import PREDdataloader, configloader
from models.loadmodelPREDIC import load_model_prediction  

def main():
    path_to_config = '/Users/janzdrazil/Desktop/new_age_image/PLANTAIGIT/configs/paramsPRED.yaml'
    config = configloader.load_config(path_to_config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(config['dataloading']['resize_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5534967, 0.56618726, 0.49286446],
                             std=[0.16803835, 0.1536287, 0.21336597])
    ])


    dataset = PREDdataloader.CustomImageDataset(
        config=config,
        transform=transform,
    )

    print(f'Number of image pairs: {len(dataset)}')

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['dataloading']['batch_size'],
        shuffle=False,
    )

    model = load_model_prediction(config)
    model.to(device)
    model.eval()

    mean = np.array([0.5534967, 0.56618726, 0.49286446])
    std = np.array([0.16803835, 0.1536287, 0.21336597])

    output_folder = config['datasaving']['output_folder']
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad(): 
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)  # Move the batch to the appropriate device
            prediction = model(batch)  # Get model prediction for this batch
            # prediction shape: [batch_size, T, C, H, W]
            predicted_frame = prediction[:, -1]  # Shape: [batch_size, C, H, W]

            # Move to CPU
            predicted_frame = predicted_frame.cpu()

            # Denormalize and save images
            for i in range(predicted_frame.size(0)):
                img_tensor = predicted_frame[i]
                img_tensor = denormalize(img_tensor, mean, std)
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img = transforms.ToPILImage()(img_tensor)
                img.save(os.path.join(output_folder, f'prediction_{batch_idx * predicted_frame.size(0) + i}.png'))

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor by applying the inverse of normalization.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor

if __name__ == '__main__':
    main()
