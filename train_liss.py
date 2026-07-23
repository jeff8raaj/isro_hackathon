import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import LISSDataset
# IMPORT YOUR MODEL CLASSES HERE
# from models import Pix2PixGenerator, Discriminator 

# --- Configuration ---
PATCH_DIR = "/home/jeffrin/isro_hackathon/data/patches/247109911"
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories
os.makedirs("outputs/liss_validation", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 1. Dataset & DataLoader
dataset = LISSDataset(patch_dir=PATCH_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 2. Model Setup
# netG = Pix2PixGenerator().to(DEVICE)
# optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002)
# criterion = nn.L1Loss()

print(f"Starting LISS Baseline training for {EPOCHS} epochs...")

# 3. Training Loop
for epoch in range(EPOCHS):
    epoch_dir = f"outputs/epoch_{epoch+1:03d}"
    os.makedirs(epoch_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        inputs = batch["input"].to(DEVICE)
        targets = batch["target"].to(DEVICE)

        # --- Training Logic ---
        # fake = netG(inputs)
        # loss = criterion(fake, targets)
        
        # optimizerG.zero_grad()
        # loss.backward()
        # optimizerG.step()

        if i % 10 == 0:
            # print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
            pass

        # --- Validation & Checkpointing ---
        if i == 0:
            # Save visual validation
            # save_image(inputs[0], f"{epoch_dir}/input.png", normalize=True)
            # save_image(fake[0], f"{epoch_dir}/prediction.png", normalize=True)
            # save_image(targets[0], f"{epoch_dir}/target.png", normalize=True)
            print(f"Epoch {epoch+1}: Validation images saved to {epoch_dir}")

    # Save checkpoint at end of epoch
    # torch.save(netG.state_dict(), f"checkpoints/liss_baseline_epoch_{epoch+1}.pth")
    print(f"Checkpoint for Epoch {epoch+1} saved.")

print("Training Complete.")
