import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import torchvision

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 50
gn_input_dim = 100  # Dimension of the noise vector for the generator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")

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
# 生成器の初期化
generator = utils.Generator().to(device)
generator.apply(utils.waight_init)

# 識別器の初期化
discriminator = utils.Discriminator().to(device)
discriminator.apply(utils.waight_init)

criterion = nn.BCELoss()
optimizer_ds = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_gn = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

output_dir = './outputs/dcgan'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training loop
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    d_loss = 0.0
    g_loss = 0.0
    for i, (images, _) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')):
        real_images = images.to(device)
        # Add noise to images
        noisy_images = torch.randn(real_images.size(0), gn_input_dim, 1, 1, device=device)  # Random noise

        real_target = torch.full((real_images.size(0),), 1., device=device)
        fake_target = torch.full((real_images.size(0),), 0., device=device)

        # 識別器の学習
        discriminator.zero_grad()
        discriminator_real_output = discriminator(real_images)
        real_loss = criterion(discriminator_real_output, real_target)
        true_dsout_mean = discriminator_real_output.mean().item()

        fake_image = generator(noisy_images)
        discriminator_fake_output = discriminator(fake_image.detach())
        fake_loss = criterion(discriminator_fake_output, fake_target)
        false_dsout_mean = discriminator_fake_output.mean().item()
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_ds.step()

        # 生成器の学習
        generator.zero_grad()
        discriminator_output = discriminator(fake_image)
        g_loss = criterion(discriminator_output, real_target)
        g_loss.backward()
        fake_dsout_mean = discriminator_output.mean().item()
        optimizer_gn.step()

        fake_image = generator(noisy_images)

        # Save the last fake image of the epoch
        if (i+1) % 100 == 0:
            torchvision.utils.save_image(fake_image, os.path.join(output_dir, f'fake_images_step_{epoch+1}_{i+1}.png'))

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss/len(train_loader):.4f}, g_loss: {g_loss/len(train_loader):.4f}')

# Save the trained model
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')