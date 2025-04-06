import torch
import torch.nn as nn

# 定数定義
DEFAULT_CHANNELS = 128  # デフォルトのチャネル数

class Discriminator(nn.Module):
    def __init__(self, img_ch, start_channels=DEFAULT_CHANNELS):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(img_ch, out_channels=start_channels, kernel_size=4, stride=2, padding=1)
        self.leak1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(start_channels, out_channels=start_channels*2, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(start_channels*2)
        self.leak2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(start_channels*2, out_channels=start_channels*4, kernel_size=3, stride=2, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(start_channels*4)
        self.leak3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(start_channels*4, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.sigmoid4 = nn.Sigmoid()

    def forward(self, x):
        x = self.leak1(self.conv1(x))
        x = self.leak2(self.batchnorm2(self.conv2(x)))
        x = self.leak3(self.batchnorm3(self.conv3(x)))
        x = self.sigmoid4(self.conv4(x))
        x = x.squeeze()
        return x

class Generator(nn.Module):
    def __init__(self, img_ch, input_dim, out_ch=DEFAULT_CHANNELS):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, out_channels=out_ch * 4, kernel_size=3, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(out_ch * 4)
        self.lelu1 = nn.ReLU(0.2)
        self.conv2 = nn.ConvTranspose2d(out_ch * 4, out_channels=out_ch * 2, kernel_size=3, stride=2, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(out_ch * 2)
        self.lelu2 = nn.ReLU(0.2)
        self.conv3 = nn.ConvTranspose2d(out_ch * 2, out_channels=out_ch, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_ch)
        self.lelu3 = nn.ReLU(0.2)
        self.conv4 = nn.ConvTranspose2d(out_ch, out_channels=img_ch, kernel_size=4, stride=2, padding=1)
        self.tanh4 = nn.Tanh()

    def forward(self, x):
        x = self.lelu1(self.conv1(x))
        x = self.lelu2(self.batchnorm2(self.conv2(x)))
        x = self.lelu3(self.batchnorm3(self.conv3(x)))
        x = self.tanh4(self.conv4(x))
        return x


def waight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)