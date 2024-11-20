import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import ODdataloader, configloader
from models.loadmodelOD import load_model_od


def main():
    """
    Main function to perform object detection on a dataset of images,
    visualize the detections, and save the output images with bounding boxes.
    """

    path_to_config = '/Users/janzdrazil/Desktop/new_age_image/PLANTAIGIT/configs/paramsOD.yaml'
    config = configloader.load_config(path_to_config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((1920, 2560)),
        transforms.ToTensor(),
    ])


    dataset = ODdataloader.CustomImageDataset(
        config=config,
        transform=transform,
    )

    print(f'Number of images: {len(dataset)}')

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['dataloading']['batch_size'],
        shuffle=False,
    )

    model = load_model_od(config)
    model.to(device)
    model.eval()

    output_directory = config['datasaving']['output_folder']
    os.makedirs(output_directory, exist_ok=True)

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch_idx, (images, image_names) in enumerate(data_loader):
            images = images.to(device)

            # Perform inference
            outputs = model(images)

            # Process each image in the batch
            for idx in range(len(images)):
                image = images[idx].cpu().numpy().transpose((1, 2, 0))
                output = outputs[idx]

                # Create a plot
                fig, ax = plt.subplots(1)
                ax.imshow(image)

                # Extract predictions
                boxes = output['boxes'].cpu()
                labels = output['labels'].cpu()
                scores = output['scores'].cpu()

                # Combine boxes, labels, and scores
                detections = list(zip(boxes, labels, scores))
                # Sort detections by score in descending order
                detections.sort(key=lambda x: x[2], reverse=True)

                # Select top N detections
                top_detections = detections[:24]

                # Draw bounding boxes and labels
                for box, label, score in top_detections:
                    x_min, y_min, x_max, y_max = box.tolist()
                    width = x_max - x_min
                    height = y_max - y_min

                    # Create a rectangle patch
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        width,
                        height,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Optional: Add labels and scores
                    # ax.text(
                    #     x_min,
                    #     y_min - 10,
                    #     f"{label.item()} ({score.item():.2f})",
                    #     color='yellow',
                    #     fontsize=8,
                    #     backgroundcolor='black'
                    # )

                ax.axis('off')

                # Save the image with detections
                image_name = os.path.splitext(image_names[idx])[0]
                output_path = os.path.join(output_directory, f"{image_name}_output.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

    print(f"Processing complete. Output images saved to: {output_directory}")


if __name__ == "__main__":
    main()
