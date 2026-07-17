from preprocessing.liss_loader import GeoTIFFLoader
from preprocessing.patch_extractor import SpatialPatchExtractor

# 1. Load the scene
loader = GeoTIFFLoader(normalise=False)
scene_dir = "/home/jeffrin/isro_hackathon/data/raw_samples/247109911"
img_array, meta = loader.load_image(scene_dir)

# 2. Extract patches
extractor = SpatialPatchExtractor(patch_size=256, stride=256)
patches, coords = extractor.extract_patches(img_array)
print(f"Extracted {len(patches)} patches.")

# 3. Save as GeoTIFFs
output_dir = "/home/jeffrin/isro_hackathon/data/patches/247109911"
extractor.save_patches_as_geotiffs(patches, coords, meta, output_dir)
print(f"Patches successfully saved to {output_dir}")
