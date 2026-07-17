import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Import the verified classes
from src.models import CloudRemovalGenerator
from src.dataset_loader import LISS4CloudDataset

def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Instantiate and load the trained Generator
    try:
        generator = CloudRemovalGenerator().to(device)
        checkpoint = torch.load("data/generator_checkpoint.pth", map_location=device)
        
        # Get the state_dict
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Map mixed keys to match current class structure
        new_state_dict = {}
        for k, v in state_dict.items():
            # Map .conv.X and .up.X names to simple .X names
            # This aligns keys like 'down2.conv.1.weight' to 'down2.1.weight'
            new_key = k.replace(".conv.", ".").replace(".up.", ".")
            new_state_dict[new_key] = v

        # Load the mapped weights
        missing, unexpected = generator.load_state_dict(new_state_dict, strict=False)
        
        generator.eval()
        print("✅ Model loaded successfully (with mapping).")
        print(f"Missing keys ignored: {len(missing)}")
        print(f"Unexpected keys ignored: {len(unexpected)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Load the validation dataset
    try:
        val_dataset = LISS4CloudDataset(
            base_dir="/home/jeffrin/isro_hackathon/data/datasets/RICE", 
            dataset_type="RICE1"
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Create output directory
    output_dir = "outputs/validation_samples"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Generate and save 5 random validation triads
    print("\nGenerating validation sample triads...")
    count = 0
    with torch.no_grad():
        for i, (cloudy, ground_truth) in enumerate(val_loader):
            if count >= 5:
                break
            
            cloudy = cloudy.to(device)
            
            # Run model inference
            generated = generator(cloudy)
            
            # Concatenate horizontally: Cloudy | Generated | Ground Truth
            # Ensure ground_truth is also on the correct device
            ground_truth = ground_truth.to(device)
            triad = torch.cat([cloudy, generated, ground_truth], dim=3)
            
            # Save the triad image
            save_path = os.path.join(output_dir, f"triad_{count + 1}.png")
            save_image(triad, save_path, normalize=True, value_range=(-1.0, 1.0))
            print(f"Saved: {save_path}")
            count += 1

    print("\n🎉 Success! Validation samples saved to 'outputs/validation_samples/'.")

if __name__ == "__main__":
    main()
