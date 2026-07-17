import rasterio
import numpy as np
import os
import glob

# ... (rest of your imports) ...

class GeoTIFFLoader:
    def __init__(self, normalise=True):
        self.normalise = normalise

    def load_image(self, directory_path):
        """
        Loads all BAND*.tif files from a directory and stacks them.
        """
        # Find all BAND files and sort them to ensure correct order
        band_files = sorted(glob.glob(os.path.join(directory_path, "BAND*.tif")))
        
        if not band_files:
            raise FileNotFoundError(f"No BAND*.tif files found in {directory_path}")

        # Read the first band to get metadata and dimensions
        with rasterio.open(band_files[0]) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width
            # Update meta to reflect the total number of bands
            meta.update(count=len(band_files))

        # Initialize the stack
        stacked_bands = np.zeros((height, width, len(band_files)), dtype=np.float32)

        # Load each band
        for i, band_file in enumerate(band_files):
            with rasterio.open(band_file) as src:
                stacked_bands[:, :, i] = src.read(1)

        if self.normalise:
            max_val = 65535.0 if stacked_bands.max() > 255 else 255.0
            stacked_bands /= max_val

        return stacked_bands, meta
