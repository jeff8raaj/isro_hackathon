# preprocessing/patch_extractor.py
import numpy as np

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
