import rasterio
from patchify import patchify
import numpy as np
import os

def generate_dataset_patches(image_path, output_dir, patch_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Opening aligned imagery stack: {image_path}")
    with rasterio.open(image_path) as src:
        # Read all bands/channels
        data = src.read()
        meta = src.meta.copy()
        
        # Convert layout to channels-last: (height, width, channels)
        data = np.moveaxis(data, 0, -1)
        
    h, w, c = data.shape
    print(f"Full dimensions: {w}x{h} with {c} channels.")
    
    # Check if the test raster is large enough to tile
    if h < patch_size or w < patch_size:
        print(f"⚠️ Image size ({w}x{h}) is smaller than patch size ({patch_size}x{patch_size}).")
        print("Falling back to slice at image's native resolution for testing...")
        patch_size = min(h, w)

    # Slice into patches
    print(f"Slicing matrix into {patch_size}x{patch_size} patches...")
    patches = patchify(data, (patch_size, patch_size, c), step=patch_size)
    
    patch_count = 0
    rows, cols, _, _, _, _ = patches.shape
    
    for r in range(rows):
        for col in range(cols):
            single_patch = patches[r, col, 0]
            
            # Skip empty or mostly black border patches (> 50% zero values)
            if np.count_nonzero(single_patch == 0) > (patch_size * patch_size * c * 0.5):
                continue
                
            output_filename = os.path.join(output_dir, f"patch_r{r}_c{col}.npy")
            # Save as binary NumPy files—loads instantly in PyTorch!
            np.save(output_filename, single_patch)
            patch_count += 1
            
    print(f"=== SUCCESS ===\nGenerated {patch_count} patches in: {output_dir}\n")

if __name__ == "__main__":
    # Test paths using your local files
    input_stack = '/home/jeffrin/isro_hackathon/data/sentinel_resampled.tif'
    output_folder = '/home/jeffrin/isro_hackathon/data/dataset_patches'
    
    try:
        generate_dataset_patches(input_stack, output_folder, patch_size=256)
    except Exception as e:
        print(f"Pipeline processing stopped: {e}")