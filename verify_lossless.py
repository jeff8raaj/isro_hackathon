import numpy as np
from preprocessing.liss_loader import GeoTIFFLoader
from preprocessing.patch_extractor import SpatialPatchExtractor

# 1. Load the original scene
scene_dir = "/home/jeffrin/isro_hackathon/data/raw_samples/247109911"
loader = GeoTIFFLoader(normalise=False)
original_img, meta = loader.load_image(scene_dir)

# 2. Extract patches
patch_size = 256
extractor = SpatialPatchExtractor(patch_size=patch_size, stride=patch_size)
patches, coords = extractor.extract_patches(original_img)

# 3. Reconstruct
# Note: We only reconstruct up to the area covered by the non-overlapping patches
# to ensure a direct comparison.
h_limit = (original_img.shape[0] // patch_size) * patch_size
w_limit = (original_img.shape[1] // patch_size) * patch_size
original_trimmed = original_img[:h_limit, :w_limit, :]

reconstructed = extractor.reconstruct_from_patches(patches, coords, original_trimmed.shape)

# 4. Compare
difference = np.abs(original_trimmed - reconstructed)
max_diff = np.max(difference)

print(f"Original trimmed shape: {original_trimmed.shape}")
print(f"Reconstructed shape:    {reconstructed.shape}")
print(f"Max absolute difference: {max_diff}")

if max_diff < 1e-5:
    print("✅ Pipeline is lossless: Reconstruction matches original!")
else:
    print("❌ Pipeline error: Mismatch detected.")
