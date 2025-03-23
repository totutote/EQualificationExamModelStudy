import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ##transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = utils.DenoisingAutoencoder().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, _ in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
        images = images.to(device)
        # Add noise to images
        noisy_images = utils.add_noise(images)
        noisy_images = noisy_images.to(device)

        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            noisy_images = utils.add_noise(images).to(device)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            val_loss += loss.item()

    print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

# Save the trained model
torch.save(model.state_dict(), 'denoising_autoencoder.pth')