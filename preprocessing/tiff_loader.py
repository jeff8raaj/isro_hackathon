# preprocessing/tiff_loader.py
import rasterio
import numpy as np
import torch

class GeoTIFFLoader:
    def __init__(self, normalise=True):
        self.normalise = normalise

    def load_image(self, file_path):
        """
        Reads a geospatial image file using rasterio.
        Maintains geo-spatial coordinate reference profile.
        """
        with rasterio.open(file_path) as src:
            meta = src.meta.copy()
            img_data = src.read([1, 2, 3]) # Extract first 3 primary bands
            
            img_data = np.moveaxis(img_data, 0, -1).astype(np.float32)
            
            if self.normalise:
                max_val = 65535.0 if img_data.max() > 255 else 255.0
                img_data /= max_val
                
        return img_data, meta

    def to_tensor(self, img_array):
        """
        Converts the processed numpy array to a PyTorch Tensor normalized to [-1, 1] for cGAN.
        """
        img_tensor = (img_array * 2.0) - 1.0
        return torch.from_numpy(img_tensor).permute(2, 0, 1)
