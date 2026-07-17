import torch

def get_mapped_state_dict(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    model_state = model.state_dict()
    new_state_dict = {}
    
    # Simple mapping logic: 
    # Because your checkpoint has mixed naming (down1.0 vs down2.conv.0), 
    # we just need to ensure the target model keys exist.
    for k, v in state_dict.items():
        # This acts as a 'soft' mapping. If the key exists, it stays.
        # If it doesn't, it attempts to normalize the name.
        if k in model_state:
            new_state_dict[k] = v
        else:
            # Fallback for naming mismatches
            normalized_k = k.replace(".conv.", ".0.") 
            if normalized_k in model_state:
                new_state_dict[normalized_k] = v
    return new_state_dict
