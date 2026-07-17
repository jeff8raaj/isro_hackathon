import rasterio
import numpy as np
import os
import glob

def load_stacked_bands(directory_path):
    # Find all .tif files that look like bands (adjust pattern if needed)
    band_files = sorted(glob.glob(os.path.join(directory_path, "BAND*.tif")))
    
    if not band_files:
        raise FileNotFoundError("No BAND*.tif files found in the directory.")

    # Read the first band to get metadata (profile/transform)
    with rasterio.open(band_files[0]) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        
    # Prepare an empty array for all bands
    stacked_image = np.zeros((len(band_files), height, width), dtype=profile['dtype'])
    
    # Fill the array
    for i, band_file in enumerate(band_files):
        with rasterio.open(band_file) as src:
            stacked_image[i] = src.read(1) # Read first band of each file
            
    return {
        "image": stacked_image,
        "profile": profile,
        "transform": transform,
        "crs": crs,
        "band_files": band_files
    }

# Test it
if __name__ == "__main__":
    folder = "/home/jeffrin/isro_hackathon/data/raw_samples/247109911"
    data = load_stacked_bands(folder)
    print(f"✅ Successfully stacked {data['image'].shape[0]} bands.")
    print(f"Shape: {data['image'].shape}")
