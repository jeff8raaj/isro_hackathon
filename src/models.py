import torch
import torch.nn as nn

class CloudRemovalGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CloudRemovalGenerator, self).__init__()
        
        # Encoder: Using sequential blocks to produce .0, .1, .2 keys
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.down4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        
        # Decoder
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        
        # Final
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        return self.final(torch.cat([u3, d1], 1))
