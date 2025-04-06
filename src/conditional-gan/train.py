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
label_num = 10  # Number of classes in the dataset
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
    ##transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
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
gn_out_channels =  in_channels  # Number of classes in the dataset
ds_in_channels =  in_channels + label_num  # Number of classes in the dataset

# Initialize model, loss function, and optimizer
# 生成器の初期化
generator = utils.Generator(gn_out_channels, input_dim=gn_input_dim + label_num).to(device)
generator.apply(utils.waight_init)

# 識別器の初期化
discriminator = utils.Discriminator(ds_in_channels).to(device)
discriminator.apply(utils.waight_init)

criterion = nn.BCELoss()
optimizer_ds = optim.Adam(discriminator.parameters(), lr=ds_learning_rate, betas=(0.5, 0.999))
optimizer_gn = optim.Adam(generator.parameters(), lr=gn_learning_rate, betas=(0.5, 0.999))

output_dir = './outputs/conditional_gan'
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
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')):
        real_images = images.to(device)
        real_labels = labels.to(device)
        loop_batch_size = real_images.size(0)

        real_image_label = utils.concat_img_label(real_images, real_labels, label_num, device)

        # Add noise to images
        noisy_images = torch.randn(loop_batch_size, gn_input_dim, 1, 1, device=device)  # Random noise

        fake_labels = torch.randint(label_num, (loop_batch_size,), dtype=torch.long, device=device)
        fake_noisy_image_and_label = utils.concat_noise_and_labels(noisy_images, fake_labels, label_num, device)

        real_target = torch.full((loop_batch_size,), 1., device=device)
        fake_target = torch.full((loop_batch_size,), 0., device=device)

        # 識別器の学習
        discriminator.zero_grad()
        discriminator_real_output = discriminator(real_image_label)
        real_loss = criterion(discriminator_real_output, real_target)
        total_real_loss += real_loss.item()
        #true_dsout_mean = discriminator_real_output.mean().item()

        fake_image = generator(fake_noisy_image_and_label)
        fake_image_label = utils.concat_img_label(fake_image, fake_labels, label_num, device)
        discriminator_fake_output = discriminator(fake_image_label.detach())
        fake_loss = criterion(discriminator_fake_output, fake_target)
        total_fake_loss += fake_loss.item()
        #false_dsout_mean = discriminator_fake_output.mean().item()
        d_loss = real_loss + fake_loss
        total_d_loss += d_loss.item()
        d_loss.backward()
        optimizer_ds.step()

        # 生成器の学習
        generator.zero_grad()
        discriminator_output = discriminator(fake_image_label)
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
    noisy_images = torch.randn(batch_size, gn_input_dim, 1, 1, device=device)
    fake_labels = torch.arange(label_num).repeat(batch_size // label_num + 1)[:batch_size].to(device)
    fake_noisy_image_and_label = utils.concat_noise_and_labels(noisy_images, fake_labels, label_num, device)
    fake_image = generator(fake_noisy_image_and_label)
    torchvision.utils.save_image(fake_image, os.path.join(output_dir, f'fake_images_step_{epoch+1}.png'))

# Save the trained model
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')