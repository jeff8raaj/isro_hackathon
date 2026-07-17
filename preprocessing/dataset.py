
# preprocessing/dataset.py
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2

class RemoteSensingDataset(Dataset):
    def __init__(self, root_dir, patch_size=256):
        self.root_dir = root_dir
        self.patch_size = patch_size
        
        # Explicit mapping to RICE1 folder names: 'cloud' and 'label'
        self.cloudy_images = sorted(glob.glob(os.path.join(root_dir, 'cloud', '*.*')))
        self.clear_images = sorted(glob.glob(os.path.join(root_dir, 'label', '*.*')))

        if len(self.cloudy_images) == 0:
            available = os.listdir(root_dir) if os.path.exists(root_dir) else "Path Not Found"
            raise ValueError(f"Could not locate image pairs in {root_dir}. Available folders: {available}")

        assert len(self.cloudy_images) == len(self.clear_images), \
            f"Mismatch: Found {len(self.cloudy_images)} cloudy images and {len(self.clear_images)} clear images."

    def __len__(self):
        return len(self.cloudy_images)

    def __getitem__(self, idx):
        cloudy_path = self.cloudy_images[idx]
        clear_path = self.clear_images[idx]
        
        cloudy_img = cv2.imread(cloudy_path)
        clear_img = cv2.imread(clear_path)
        
        cloudy_img = cv2.cvtColor(cloudy_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        if cloudy_img.shape[0] != self.patch_size or cloudy_img.shape[1] != self.patch_size:
            cloudy_img = cv2.resize(cloudy_img, (self.patch_size, self.patch_size))
            clear_img = cv2.resize(clear_img, (self.patch_size, self.patch_size))
            
        # Normalize [0, 255] -> [0.0, 1.0]
        cloudy_img = cloudy_img.astype(np.float32) / 255.0
        clear_img = clear_img.astype(np.float32) / 255.0
        
        # Scale to [-1, 1] for cGAN
        cloudy_tensor = torch.from_numpy((cloudy_img * 2.0) - 1.0).permute(2, 0, 1)
        clear_tensor = torch.from_numpy((clear_img * 2.0) - 1.0).permute(2, 0, 1)
        
        return {
            'cloudy': cloudy_tensor,
            'clear': clear_tensor
        }
