import sys
import os
import torch
from PIL import Image
import numpy as np

# Force path resolution so imports work from terminal seamlessly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FIX: Import the exact class name matching models.py
from models import SatelliteCloudRemovalUNet
from metrics import calculate_psnr, calculate_ssim, calculate_rmse, calculate_sam

def run_reconstruction_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🛰️ Evaluation Engine Active. Target Hardware: {device}")

    # CONFIGURATION
    IS_RGB = True
    channels = 3 if IS_RGB else 1

    # PATHS - Verified dynamic workspace paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "src" in CURRENT_DIR else CURRENT_DIR
    
    checkpoint_path = os.path.join(PROJECT_ROOT, "data", "generator_checkpoint.pth")
    cloudy_test_path = os.path.join(PROJECT_ROOT, "data", "221666021", "BAND2.tif") # Baseline file pointer

    # FIX: Instantiate the correct U-Net structural name
    model = SatelliteCloudRemovalUNet(in_channels=channels, out_channels=channels).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint at: {checkpoint_path}")
        
     model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()
    print(f"✅ Trained checkpoint loaded cleanly from {checkpoint_path}")

if __name__ == "__main__":
    run_reconstruction_inference()