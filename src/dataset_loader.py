import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

class SatellitePatchDataset(Dataset):
    def __init__(self, patch_dir):
        self.patch_dir = patch_dir
        # Find all numpy files inside your patch folder
        self.patch_paths = glob.glob(os.path.join(patch_dir, "*.npy"))
        print(f"Loaded Satellite Dataset: Found {len(self.patch_paths)} patches for processing.")

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        # 1. Load the binary matrix disk patch
        patch_path = self.patch_paths[idx]
        patch_data = np.load(patch_path)
        
        # 2. Convert to a standard floating-point Torch Tensor
        tensor_data = torch.from_numpy(patch_data).float()
        
        # 3. Permute layouts: Change from Channels-Last (H, W, C) to Channels-First (C, H, W)
        # This is strictly required by PyTorch Convolutional Layer operations!
        tensor_data = tensor_data.permute(2, 0, 1)
        
        # 4. Max-Value Min-Max Normalization scaling pixels to [0.0, 1.0] range
        if torch.max(tensor_data) > 0:
            tensor_data = tensor_data / 255.0
            
        return tensor_data

if __name__ == "__main__":
    # Point directly to your active dataset patch output path
    target_patches_dir = '/home/jeffrin/isro_hackathon/data/dataset_patches'
    
    try:
        # Initialize the dataset pipeline class
        dataset = SatellitePatchDataset(target_patches_dir)
        
        if len(dataset) > 0:
            # Create a PyTorch DataLoader to handle automatic batching and shuffling
            test_loader = DataLoader(dataset, collate_fn=None, batch_size=1, shuffle=True)
            
            # Fetch and profile a sample batch matrix configuration
            for sample_batch in test_loader:
                print("\n=== SUCCESS: PyTorch Pipeline Validation ===")
                print(f"Batch Tensor Shape Profile : {sample_batch.shape} -> [Batch, Channels, Height, Width]")
                print(f"Pixel Intensity Spectrum Range: [{torch.min(sample_batch):.2f} to {torch.max(sample_batch):.2f}]")
                break
        else:
            print("⚠️ No valid patches discovered. Run patch_maker.py first to populate your folder.")
            
    except Exception as e:
        print(f"Data Batching Pipeline Error: {e}")