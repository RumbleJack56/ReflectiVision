
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torchvision.transforms.functional as F

class EnhancedUNetDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(EnhancedUNetDiscriminator, self).__init__()

        # Encoder layers with more depth
        self.enc1 = self.down_block(in_channels, base_filters)
        self.enc2 = self.down_block(base_filters, base_filters * 2)
        self.enc3 = self.down_block(base_filters * 2, base_filters * 4)
        self.enc4 = self.down_block(base_filters * 4, base_filters * 8)
        self.enc5 = self.down_block(base_filters * 8, base_filters * 16)

        # Bottleneck with increased depth
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 16, base_filters * 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 32),
            nn.ReLU(True)
        )

        # Decoder layers
        self.dec5 = self.up_block(base_filters * 32, base_filters * 16)
        self.dec4 = self.up_block(base_filters * 16, base_filters * 8)
        self.dec3 = self.up_block(base_filters * 8, base_filters * 4)
        self.dec2 = self.up_block(base_filters * 4, base_filters * 2)
        self.dec1 = self.up_block(base_filters * 2, base_filters)

        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        bottleneck = self.bottleneck(enc5)

        dec5 = self.dec5(bottleneck) + enc5
        dec4 = self.dec4(dec5) + enc4
        dec3 = self.dec3(dec4) + enc3
        dec2 = self.dec2(dec3) + enc2
        dec1 = self.dec1(dec2) + enc1

        output = self.final_conv(dec1)
        output = FF.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
        
        return torch.sigmoid(output)

# Loss Functions

vgg = models.vgg16(pretrained=True).features[:8].eval().to("cuda")

# Freeze VGG parameters to prevent training on them
for param in vgg.parameters():
    param.requires_grad = False


def perceptual_loss(real, fake):
    # Check if input images are single-channel; if so, repeat to make them 3-channel
    if real.shape[1] == 1:
        real = real.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    if fake.shape[1] == 1:
        fake = fake.repeat(1, 3, 1, 1)  # Convert grayscale to RGB

    # Extract features from the VGG model
    with torch.no_grad():
        real_features = vgg(real)
        fake_features = vgg(fake)
    
    # Compute the Mean Squared Error between real and fake features
    return torch.mean((real_features - fake_features) ** 2)

def multi_scale_perceptual_loss(real, fake, scales=[0.25, 0.5, 1.0], weights=[0.5, 1.0, 2.0]):
    """
    Calculates perceptual loss at different scales with increasing weights.
    
    Parameters:
    - real (torch.Tensor): Real target images, expected to be 3-channel.
    - fake (torch.Tensor): Fake generated images, expected to be 3-channel.
    - scales (list of float): List of scaling factors to apply to the images (e.g., 0.25, 0.5, 1.0).
    - weights (list of float): List of weights corresponding to each scale, with higher weights for larger scales.

    Returns:
    - torch.Tensor: The total multi-scale perceptual loss.
    """
    total_loss = 0.0
    
    for scale, weight in zip(scales, weights):
        # Resize real and fake images to the current scale
        scaled_real = FF.interpolate(real, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_fake = FF.interpolate(fake, scale_factor=scale, mode='bilinear', align_corners=False)
        
        # Ensure inputs are 3-channel RGB
        if scaled_real.shape[1] == 1:
            scaled_real = scaled_real.repeat(1, 3, 1, 1)
        if scaled_fake.shape[1] == 1:
            scaled_fake = scaled_fake.repeat(1, 3, 1, 1)
        
        # Extract features using VGG model
        with torch.no_grad():
            real_features = vgg(scaled_real)
            fake_features = vgg(scaled_fake)
        
        # Calculate the perceptual loss at the current scale and apply the weight
        scale_loss = torch.mean((real_features - fake_features) ** 2)
        total_loss += weight * scale_loss
    
    return total_loss