import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Dual Convolutional Block with Batch Normalization for feature stabilization."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SatelliteCloudRemovalUNet(nn.Module):
    """
    U-Net Generative Architecture for Satellite Image Cloud Removal.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(SatelliteCloudRemovalUNet, self).__init__()
        
        # Encoder (Downsampling Stream)
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        
        # Decoder (Upsampling Stream)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(128, 64)
        
        # Final Projection Layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d1 = self.up1(x4)
        d1_fused = torch.cat([d1, x3], dim=1)
        d1_out = self.conv_up1(d1_fused)
        
        d2 = self.up2(d1_out)
        d2_fused = torch.cat([d2, x2], dim=1)
        d2_out = self.conv_up2(d2_fused)
        
        d3 = self.up3(d2_out)
        d3_fused = torch.cat([d3, x1], dim=1)
        d3_out = self.conv_up3(d3_fused)
        
        return self.outc(d3_out)