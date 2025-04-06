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
ds_learning_rate = 0.00002
gn_learning_rate = 0.0002
num_epochs = 50
gn_input_dim = 150  # Dimension of the noise vector for the generator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ##transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 dataset (コメントアウト)
# train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Load Fashion-MNIST dataset
train_dataset = datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# データセットからチャンネル数を自動取得
sample_data, _ = next(iter(train_loader))
in_channels = sample_data.shape[1]  # 通常は3（RGB）または1（グレースケール）

# Initialize model, loss function, and optimizer
# 生成器の初期化
generator = utils.Generator(in_channels, input_dim=gn_input_dim).to(device)
generator.apply(utils.waight_init)

# 識別器の初期化
discriminator = utils.Discriminator(in_channels).to(device)
discriminator.apply(utils.waight_init)

criterion = nn.BCELoss()
optimizer_ds = optim.Adam(discriminator.parameters(), lr=ds_learning_rate, betas=(0.5, 0.999))
optimizer_gn = optim.Adam(generator.parameters(), lr=gn_learning_rate, betas=(0.5, 0.999))

output_dir = './outputs/dcgan'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training loop
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    total_d_loss = 0.0
    total_g_loss = 0.0
    total_real_loss = 0.0
    total_fake_loss = 0.0
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
        total_real_loss += real_loss.item()
        true_dsout_mean = discriminator_real_output.mean().item()

        fake_image = generator(noisy_images)
        discriminator_fake_output = discriminator(fake_image.detach())
        fake_loss = criterion(discriminator_fake_output, fake_target)
        total_fake_loss += fake_loss.item()
        false_dsout_mean = discriminator_fake_output.mean().item()
        d_loss = real_loss + fake_loss
        total_d_loss += d_loss.item()
        d_loss.backward()
        optimizer_ds.step()

        # 生成器の学習
        generator.zero_grad()
        discriminator_output = discriminator(fake_image)
        g_loss = criterion(discriminator_output, real_target)
        total_g_loss += g_loss.item()
        g_loss.backward()
        fake_dsout_mean = discriminator_output.mean().item()
        optimizer_gn.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], \
          d_loss: {total_d_loss/len(train_loader):.4f}, \
          real_loss: {total_real_loss/len(train_loader):.4f}, \
          fake_loss: {total_fake_loss/len(train_loader):.4f}, \
          g_loss: {total_g_loss/len(train_loader):.4f}')

    # Validation
    generator.eval()
    discriminator.eval()
    noisy_images = torch.randn(real_images.size(0), gn_input_dim, 1, 1, device=device)
    fake_image = generator(noisy_images)
    torchvision.utils.save_image(fake_image, os.path.join(output_dir, f'fake_images_step_{epoch+1}.png'))

# Save the trained model
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')