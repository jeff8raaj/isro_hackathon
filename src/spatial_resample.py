import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import os

def match_resolutions(target_high_res_path, source_low_res_path, output_path):
    print("Reading target high-resolution geometry...")
    if not os.path.exists(target_high_res_path):
        raise FileNotFoundError(f"Target reference file missing: {target_high_res_path}")
        
    with rasterio.open(target_high_res_path) as ref_src:
        target_meta = ref_src.meta.copy()
        target_crs = ref_src.crs
        target_transform = ref_src.transform
        target_width = ref_src.width
        target_height = ref_src.height

    print(f"Opening low-res band: {source_low_res_path}")
    with rasterio.open(source_low_res_path) as src:
        # Create an in-memory Virtual Raster Template (VRT) to mathematically warp pixels
        with WarpedVRT(src, 
                       crs=target_crs, 
                       transform=target_transform, 
                       width=target_width, 
                       height=target_height, 
                       resampling=Resampling.bilinear) as vrt:
            
            # Align metadata parameters exactly to high-res target properties
            target_meta.update({
                'driver': 'GTiff',
                'height': target_height,
                'width': target_width,
                'crs': target_crs,
                'transform': target_transform,
                'dtype': vrt.dtypes[0],
                'count': 1
            })
            
            print(f"Writing warped matrix ({target_width}x{target_height}) -> {output_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, 'w', **target_meta) as dst:
                dst.write(vrt.read(1), 1)
                
    print("=== SUCCESS: Raster Resolution Harmonized ===\n")

if __name__ == "__main__":
    # Test paths utilizing your local test raster asset
    reference_raster = '/home/jeffrin/byte.tif'
    low_res_raster = '/home/jeffrin/byte.tif' 
    output_raster = '/home/jeffrin/isro_hackathon/data/sentinel_resampled.tif'
    
    try:
        match_resolutions(reference_raster, low_res_raster, output_raster)
    except Exception as e:
        print(f"Resampling Pipeline Error: {e}")