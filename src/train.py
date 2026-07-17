import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import LISS4CloudDataset
import os
import sys

# Setup relative tracking paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "src" in CURRENT_DIR else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the correct class name
from models import SatelliteCloudRemovalUNet

def train_framework():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Engine reporting for duty. Compute Target: {device}")

    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    EPOCHS = 10
    
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("Parsing dataset indexes...")
    train_dataset = LISS4CloudDataset(dataset_type="RICE1")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Initialize correct network structure
    model = SatelliteCloudRemovalUNet(in_channels=3, out_channels=3).to(device)
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    print("\n🏁 Beginning Model Optimization Loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for step, (cloudy_imgs, clear_imgs) in enumerate(train_loader):
            cloudy_imgs = cloudy_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            optimizer.zero_grad()
            reconstructed_imgs = model(cloudy_imgs)
            
            loss = criterion(reconstructed_imgs, clear_imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] | Step [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"✅ Epoch [{epoch}/{EPOCHS}] Complete. Avg Loss: {epoch_loss:.4f}")
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "generator_checkpoint.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    train_framework()