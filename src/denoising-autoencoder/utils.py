import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8
        #self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 8x8
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 2x2
        
        #self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 4x4
        #self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8x8
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  # 16x16
        self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # 32x32


    def forward(self, x):
        # Encoder
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        #x = torch.relu(self.conv3(x))
        #x = self.pool3(x)
        #x = torch.relu(self.conv4(x))
        
        # Decoder
        #x = torch.relu(self.deconv1(x))
        #x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))  # Output layer
        #x = self.deconv4(x)  # Output layer
        return x

def add_noise(images, noise_factor=0.2):
    noisy_images = images + noise_factor * torch.randn(*images.shape, device=images.device)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()