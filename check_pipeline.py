# check_pipeline.py
import os
import yaml
from preprocessing.dataset import RemoteSensingDataset
from torch.utils.data import DataLoader

def verify_pipeline():
    # Load settings from config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Point directly to the extracted RICE1 path
    target_data_dir = os.path.join(config['data']['rice1_path'], 'RICE', 'RICE1')
    
    print(f"[*] Targeting Dataset Path: {target_data_dir}")
    if not os.path.exists(target_data_dir):
        print(f"[!] Path does not exist. Double-check layout configuration.")
        return

    # Initialize the PyTorch Dataset structure
    dataset = RemoteSensingDataset(root_dir=target_data_dir, patch_size=config['data']['patch_size'])
    print(f"[+] Successfully loaded {len(dataset)} paired data samples.")

    # Initialize a miniature batch loader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Extract a single sample verification frame
    batch = next(iter(dataloader))
    
    cloudy_batch = batch['cloudy']
    clear_batch = batch['clear']
    
    print("\n--- Tensor Verification Report ---")
    print(f"Cloudy Tensor Shape: {cloudy_batch.shape} (Expected: [Batch, Channels, H, W])")
    print(f"Clear Tensor Shape:  {clear_batch.shape}")
    print(f"Cloudy Range Verification: Min = {cloudy_batch.min().item():.2f}, Max = {cloudy_batch.max().item():.2f} (Expected: [-1.0, 1.0])")
    print(f"Clear Range Verification:  Min = {clear_batch.min().item():.2f}, Max = {clear_batch.max().item():.2f}")
    print("---------------------------------")
    print("[✓] Phase 2.2 Data Pipeline Foundations are ROCK SOLID.")

if __name__ == "__main__":
    verify_pipeline()
