import rasterio
import numpy as np
import cv2
import os

def generate_cloud_and_shadow_mask(image_path, output_mask_path, bright_threshold=180, dilation_iterations=2):
    print(f"Opening raster layer: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target raster not found at {image_path}")

    with rasterio.open(image_path) as src:
        band = src.read(1)
        meta = src.meta.copy()
        
        # 1. Cloud Slicing: Segment highly reflective pixels as 255 (White)
        print(f"Applying brightness filter at threshold level ({bright_threshold})...")
        _, cloud_mask = cv2.threshold(band, bright_threshold, 255, cv2.THRESH_BINARY)
        
        # 2. Shadow Dilation: Expand boundaries using a morphological structuring kernel
        print("Running spatial dilation loop to capture adjacent cloud shadow fields...")
        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(cloud_mask, kernel, iterations=dilation_iterations)
        
        # 3. Format metadata to cleanly save as an 8-bit single-channel image grid
        meta.update(dtype=rasterio.uint8, count=1)
        
        with rasterio.open(output_mask_path, 'w', **meta) as dst:
            dst.write(final_mask.astype(rasterio.uint8), 1)
            
    print(f"=== SUCCESS ===\nBinary matrix saved straight to: {output_mask_path}\n")

if __name__ == "__main__":
    # Pointing directly to your local verified test raster path
    input_raster = '/home/jeffrin/byte.tif'
    
    # Adjust this output path depending on where you decided to save the script
    output_raster = '/home/jeffrin/isro_hackathon/data/cloud_mask_test.tif'
    
    try:
        generate_cloud_and_shadow_mask(input_raster, output_raster)
    except Exception as e:
        print(f"Pipeline stopped with execution error: {e}")