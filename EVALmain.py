import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import EVALdataloader, configloader
from models.loadmodelEVAL import load_model_evaluation


def main():
    """
    Main function to perform evaluation on a dataset of images using a pre-trained model.
    It computes the Mean Squared Error (MSE) between the standardized model predictions and 
    the true labels for each sample (image), resulting in one MSE value per image.
    """
    path_to_config = '/Users/janzdrazil/Desktop/new_age_image/PLANTAIGIT/configs/paramsEVAL.yaml'
    config = configloader.load_config(path_to_config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    dataset = EVALdataloader.CustomDataset(
        config=config,
        transform=transform,
        compute_stats=True 
    )

    print(f'Number of samples: {len(dataset)}')

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['dataloading']['batch_size'],
        shuffle=False,
        num_workers=config['dataloading']['num_workers'],
    )

    model = load_model_evaluation(config)
    model.to(device)
    model.eval()

    all_true_labels = []
    all_predictions = []


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()


            all_predictions.append(outputs_np)
            all_true_labels.append(labels_np)


    all_predictions = np.vstack(all_predictions)
    all_true_labels = np.vstack(all_true_labels)

    num_samples = all_predictions.shape[0]
    mse_scores = []
    for i in range(num_samples):
        mse = mean_squared_error(all_true_labels[i, :], all_predictions[i, :])
        mse_scores.append(mse)
        print(f'Sample {i+1}/{num_samples}: MSE = {mse:.4f}')

    average_mse = np.mean(mse_scores)
    print(f'Average MSE: {average_mse:.4f}')


if __name__ == "__main__":
    main()