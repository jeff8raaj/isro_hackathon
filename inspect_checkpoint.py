import torch
import os

def inspect():
    checkpoint_path = "data/generator_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state_dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    print(f"--- Printing first 40 keys of state_dict ---")
    keys = list(state.keys())
    for i in range(min(40, len(keys))):
        print(f"{i}: {keys[i]}")

if __name__ == "__main__":
    inspect()
