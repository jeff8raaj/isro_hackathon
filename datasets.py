import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset

class LISSDataset(Dataset):
    def __init__(self, patch_dir, transform=None, normalise=True):
        self.patch_dir = patch_dir
        self.transform = transform
        self.normalise = normalise
        # Discover all patch GeoTIFFs
        self.patch_files = sorted([
            f for f in os.listdir(patch_dir) 
            if f.endswith('.tif')
        ])

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.patch_dir, self.patch_files[idx])

        with rasterio.open(file_path) as src:
            # Read all bands (C, H, W)
            image = src.read().astype(np.float32)
            # Store only the filename to avoid DataLoader serialization errors
            meta = {
                "filename": self.patch_files[idx]
            }

        # Normalize to [0, 1] range based on max value
        if self.normalise:
            max_val = image.max() if image.max() > 0 else 1.0
            image /= max_val

        # Convert to tensor and ensure (C, H, W)
        tensor = torch.from_numpy(image)

        # Return a dictionary that only contains tensors and strings
        return {
            "input": tensor,
            "target": tensor,
            "metadata": meta
        }
