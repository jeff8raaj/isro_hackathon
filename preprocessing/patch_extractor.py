import numpy as np
import rasterio
from rasterio.windows import Window
import os

class SpatialPatchExtractor:
    def __init__(self, patch_size=256, stride=256):
        self.patch_size = patch_size
        self.stride = stride

    def extract_patches(self, img_array):
        """
        Slices a standard array input of format (H, W, C) into clean spatial sub-patches.
        Returns a list of arrays alongside tracking origins for reconstruction validation.
        """
        h, w, c = img_array.shape
        patches = []
        coordinates = []  # To trace spatial origins for inverted stitch stitching

        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img_array[y:y+self.patch_size, x:x+self.patch_size, :]
                patches.append(patch)
                coordinates.append((y, x))

        return np.array(patches), coordinates

    def reconstruct_from_patches(self, patches, coordinates, target_shape):
        """
        Stitches extracted model inference patches back into an intact spatial array.
        """
        h, w, c = target_shape
        reconstructed = np.zeros((h, w, c), dtype=np.float32)
        counts = np.zeros((h, w, c), dtype=np.float32)

        for patch, (y, x) in zip(patches, coordinates):
            reconstructed[y:y+self.patch_size, x:x+self.patch_size, :] += patch
            counts[y:y+self.patch_size, x:x+self.patch_size, :] += 1.0

        # Prevent divide-by-zero bounds on overlap edges
        return np.where(counts > 0, reconstructed / counts, reconstructed)

    def save_patches_as_geotiffs(self, patches, coordinates, original_meta, output_dir):
        """
        Takes the output of extract_patches and saves them as georeferenced TIFFs.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (patch, (y, x)) in enumerate(zip(patches, coordinates)):
            # Calculate transform for this specific patch
            window = Window(col_off=x, row_off=y, width=self.patch_size, height=self.patch_size)
            patch_transform = rasterio.windows.transform(window, original_meta['transform'])
            
            # Prepare metadata
            patch_meta = original_meta.copy()
            patch_meta.update({
                "height": self.patch_size,
                "width": self.patch_size,
                "transform": patch_transform,
                "dtype": 'float32'
            })
            
            # Save
            path = os.path.join(output_dir, f"patch_{i:06d}.tif")
            with rasterio.open(path, 'w', **patch_meta) as dst:
                # rasterio expects (bands, height, width)
                dst.write(patch.transpose(2, 0, 1))
