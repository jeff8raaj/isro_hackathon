import torch
import numpy as np
from models import CloudRemovalGenerator
import os

def run_inference(checkpoint_path, input_patch_path, output_patch_path):
    print("🔮 Initializing Evaluation & Inference Engine...")
    
    # 1. Setup processing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Reconstruct the model structure and load the saved weight parameters
    model = CloudRemovalGenerator(in_channels=1, out_channels=1).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint missing at: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # Set network flags to evaluation mode (freezes BatchNorm/Dropout scaling)
    print(f"Loaded trained checkpoint weights from: {checkpoint_path}")

    # 3. Load and preprocess a single target patch array
    if not os.path.exists(input_patch_path):
        raise FileNotFoundError(f"Input test patch missing at: {input_patch_path}")
        
    patch_data = np.load(input_patch_path)
    
    # Match layout conversions used in our dataloader: (H, W, C) -> (C, H, W) -> add Batch dimension
    tensor_data = torch.from_numpy(patch_data).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Scale pixel range to [0.0, 1.0]
    if torch.max(tensor_data) > 0:
        tensor_data = tensor_data / 255.0

    # 4. Execute the forward pass without tracking gradients (saves memory)
    print("Passing cloudy matrix through the U-Net generator...")
    with torch.no_grad():
        reconstructed_output = model(tensor_data)
        
    # 5. Post-process the output tensor back to a viewable NumPy array
    # Remove batch dimension -> move layout back to channels-last -> scale back to [0, 255]
    output_array = reconstructed_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_array = (output_array * 255.0).astype(np.uint8)
    
    # Save the clean predicted array matrix to disk
    os.makedirs(os.path.dirname(output_patch_path), exist_ok=True)
    np.save(output_patch_path, output_array)
    
    print("\n=== SUCCESS: Reconstruction Matrix Generated ===")
    print(f"Clean Predicted Array Saved To -> {output_patch_path}")
    print(f"Output Matrix Array Resolution : {output_array.shape}")

if __name__ == "__main__":
    # Path links matching your workspace profile
    checkpoint = '/home/jeffrin/isro_hackathon/data/generator_checkpoint.pth'
    sample_input_patch = '/home/jeffrin/isro_hackathon/data/dataset_patches/patch_r0_c0.npy'
    predicted_output_path = '/home/jeffrin/isro_hackathon/data/predicted_clean_patch.npy'
    
    try:
        run_inference(checkpoint, sample_input_patch, predicted_output_path)
    except Exception as e:
        print(f"Inference Engine Stopped: {e}")