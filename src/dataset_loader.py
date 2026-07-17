import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

class LISS4CloudDataset(Dataset):
    def __init__(self, base_dir="/home/jeffrin/isro_hackathon/data/datasets/RICE", dataset_type="RICE1", transform=None):
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.transform = transform
        
        # FIXED: Corrected paths to match your directory structure (cloud/ and label/)
        if dataset_type == "RICE1":
            self.cloudy_dir = os.path.join(base_dir, "RICE1", "cloud")
            self.clear_dir = os.path.join(base_dir, "RICE1", "label")
            self.use_masks = False
        elif dataset_type == "RICE2":
            self.cloudy_dir = os.path.join(base_dir, "RICE2", "cloud")
            self.clear_dir = os.path.join(base_dir, "RICE2", "label")
            self.mask_dir = os.path.join(base_dir, "RICE2", "mask")
            self.use_masks = True
        else:
            raise ValueError("Target dataset_type must be configured explicitly as 'RICE1' or 'RICE2'")

        if not os.path.exists(self.cloudy_dir) or not os.path.exists(self.clear_dir):
            raise FileNotFoundError(f"Missing directories: {self.cloudy_dir} or {self.clear_dir}")

        # Scan and align matching filename sequences
        self.filenames = sorted([f for f in os.listdir(self.cloudy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"📡 [DATA ENGINE INITIALIZED] Verified {len(self.filenames)} paired scenes mapped for {dataset_type}.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        target_file = self.filenames[idx]
        
        # 1. Load images using Pillow
        cloudy_img = Image.open(os.path.join(self.cloudy_dir, target_file)).convert("RGB")
        clear_img = Image.open(os.path.join(self.clear_dir, target_file)).convert("RGB")
        
        # 2. Handle Masks if necessary
        mask_tensor = torch.tensor(0.0)
        if self.use_masks:
            mask_img = Image.open(os.path.join(self.mask_dir, target_file)).convert("L")
            mask_np = np.array(mask_img, dtype=np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # 3. Vectorization and normalization
        if self.transform:
            cloudy_tensor = self.transform(cloudy_img)
            clear_tensor = self.transform(clear_img)
        else:
            cloudy_np = np.array(cloudy_img, dtype=np.float32)
            clear_np = np.array(clear_img, dtype=np.float32)
            
            # Reposition axes from (H, W, C) to (C, H, W)
            cloudy_tensor = torch.from_numpy(cloudy_np).permute(2, 0, 1)
            clear_tensor = torch.from_numpy(clear_np).permute(2, 0, 1)
            
            # Scale to [-1.0, 1.0]
            cloudy_tensor = (cloudy_tensor / 127.5) - 1.0
            clear_tensor = (clear_tensor / 127.5) - 1.0

        if self.use_masks:
            return cloudy_tensor, clear_tensor, mask_tensor
        return cloudy_tensor, clear_tensor

if __name__ == "__main__":
    print("Running data stream diagnostic smoke-test...")
    try:
        test_dataset = LISS4CloudDataset(dataset_type="RICE1")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
        
        # Fetch one batch to verify
        for cloudy_batch, clear_batch in test_loader:
            print("\n✅ === PIPELINE DIAGNOSTIC PASSED SUCCESSFULLY ===")
            print(f"Cloudy Input Batch Shape : {cloudy_batch.shape}")
            print(f"Clear Target Batch Shape : {clear_batch.shape}")
            break
    except Exception as error:
        print(f"❌ Data Loader validation execution failure: {error}")