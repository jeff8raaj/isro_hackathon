import torch
import torch.nn as nn

class UNetDownsample(nn.Module):
    """Downsampling block: Convolution -> Batch Normalization -> LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super(UNetDownsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetUpsample(nn.Module):
    """Upsampling block: Transposed Convolution -> Batch Normalization -> Dropout -> ReLU"""
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(UNetUpsample, self).__init__()
        modules = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            modules.append(nn.Dropout(0.5))
        self.up = nn.Sequential(*modules)

    def forward(self, x):
        return self.up(x)

class CloudRemovalGenerator(nn.Module):
    """
    U-Net Generator Core Architecture.
    Takes cloudy multi-sensor stacks and reconstructs clean imagery.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(CloudRemovalGenerator, self).__init__()
        
        # Encoder (Downsampling path)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ) # Out: 64 x H/2 x W/2
        self.down2 = UNetDownsample(64, 128)  # Out: 128 x H/4 x W/4
        self.down3 = UNetDownsample(128, 256) # Out: 256 x H/8 x W/8
        self.down4 = UNetDownsample(256, 512) # Out: 512 x H/16 x W/16

        # Decoder (Upsampling path with Skip Connections)
        self.up1 = UNetUpsample(512, 256, use_dropout=True)  # Out: 256 x H/8 x W/8
        self.up2 = UNetUpsample(512, 128)                    # Input has doubled via skip connection concat
        self.up3 = UNetUpsample(256, 64)                     
        
        # Final reconstruction layer mapping back to original resolution and band count
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Scales output imagery cleanly to active color/reflectance spectral range
        )

    def forward(self, x):
        # Forward pass through encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Forward pass through decoder with explicit feature concatenation (Skip Connections)
        u1 = self.up1(d4)
        u1_conn = torch.cat([u1, d3], dim=1)
        
        u2 = self.up2(u1_conn)
        u2_conn = torch.cat([u2, d2], dim=1)
        
        u3 = self.up3(u2_conn)
        u3_conn = torch.cat([u3, d1], dim=1)
        
        return self.final(u3_conn)

if __name__ == "__main__":
    # Model execution shape simulation test
    print("Initializing Generator Architecture configurations...")
    model = CloudRemovalGenerator(in_channels=1, out_channels=1)
    
    # Simulating a dynamic network feed: [Batch Size, Channels, Height, Width]
    # Forcing a 256x256 patch size structure to verify down/up matching
    simulated_input = torch.randn(1, 1, 256, 256)
    
    try:
        output_tensor = model(simulated_input)
        print("\n=== SUCCESS: AI Model Initialization & Compiling ===")
        print(f"Input Matrix Profile : {simulated_input.shape}")
        print(f"Output Generated Profile: {output_tensor.shape} -> Match Verified!")
    except Exception as e:
        print(f"Network Compilation Failure: {e}")