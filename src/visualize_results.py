import torch
import matplotlib.pyplot as plt
import os
import sys
from dataset_loader import LISS4CloudDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "src" in CURRENT_DIR else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import SatelliteCloudRemovalUNet

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteCloudRemovalUNet(in_channels=3, out_channels=3).to(device)
    
    checkpoint_path = os.path.join(PROJECT_ROOT, "data", "generator_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Missing weight checkpoint file at: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.eval()

    dataset = LISS4CloudDataset(dataset_type="RICE1")
    cloudy_img, clear_img = dataset[0]
    cloudy_input = cloudy_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = model(cloudy_input).squeeze(0).cpu()

    def denorm(t): 
        return torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(denorm(cloudy_img).permute(1, 2, 0))
    axes[0].set_title("Cloudy Input")
    axes[0].axis('off')
    
    axes[1].imshow(denorm(reconstructed).permute(1, 2, 0))
    axes[1].set_title("AI Reconstruction")
    axes[1].axis('off')
    
    axes[2].imshow(denorm(clear_img).permute(1, 2, 0))
    axes[2].set_title("Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    output_png = os.path.join(PROJECT_ROOT, "results_comparison.png")
    plt.savefig(output_png, dpi=300)
    print(f"✅ Validation asset saved to: '{output_png}'")

if __name__ == "__main__":
    visualize()