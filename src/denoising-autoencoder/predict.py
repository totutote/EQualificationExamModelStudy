import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import utils
import matplotlib.pyplot as plt
import os
import numpy as np

def load_model(model_path):
    model = utils.DenoisingAutoencoder()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict(model, dataloader):
    output_dir = './outputs/denoising_autoencoder'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            noisy_images = utils.add_noise(images)
            denoised_images = model(noisy_images)

            # Save the first image in the batch
            original_img = images[0].permute(1, 2, 0).numpy()
            noisy_img = noisy_images[0].permute(1, 2, 0).numpy()
            denoised_img = denoised_images[0].permute(1, 2, 0).numpy()

            # Create a figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            # Plot original image
            axes[0].imshow(original_img)
            axes[0].set_title('Original')
            axes[0].axis('off')

            # Plot noisy image
            axes[1].imshow(noisy_img)
            axes[1].set_title('Noisy')
            axes[1].axis('off')

            # Plot denoised image
            axes[2].imshow(denoised_img)
            axes[2].set_title('Denoised')
            axes[2].axis('off')

            plt.tight_layout()  # Adjust layout to prevent overlapping titles
            plt.savefig(os.path.join(output_dir, f'comparison_{i}.png'))
            plt.clf()  # Clear the current figure

def main():
    model_path = 'denoising_autoencoder.pth'  # Update with your model path
    model = load_model(model_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predict(model, test_loader)

if __name__ == '__main__':
    main()