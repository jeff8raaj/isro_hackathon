# training/metrics.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

class RemoteSensingMetrics:
    """
    High-precision evaluation engine operating strictly on float32 spatial mappings [0.0, 1.0].
    Preserves generator continuity by dynamically handling both batched and single images.
    """
    def __init__(self):
        pass

    @staticmethod
    def tensor_to_numpy_float32(tensor):
        """Converts tensors normalized to [-1, 1] to a list of channel-last float32 arrays in [0, 1]."""
        # If a single image (C, H, W) is passed, unsqueeze it to create a batch of 1 (1, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            
        # Rescale linearly to [0.0, 1.0] while keeping full float32 precision
        scaled = torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)
        np_imgs = scaled.detach().cpu().numpy().astype(np.float32)
        
        # Permute from (B, C, H, W) to a list of (H, W, C) arrays
        return [np.transpose(img, (1, 2, 0)) for img in np_imgs]

    def compute(self, target, prediction):
        """
        Computes the complete hackathon baseline suite across the provided inputs.
        Accepts raw network tensors and evaluates them without structural precision loss.
        """
        tgt_list = self.tensor_to_numpy_float32(target)
        pred_list = self.tensor_to_numpy_float32(prediction)
        
        batch_psnr = []
        batch_ssim = []
        batch_rmse = []
        
        for tgt_img, pred_img in zip(tgt_list, pred_list):
            # 1. Standard verified Peak Signal-to-Noise Ratio
            psnr_val = psnr_func(tgt_img, pred_img, data_range=1.0)
            batch_psnr.append(psnr_val)
            
            # 2. Multi-band structural similarity index
            ssim_val = ssim_func(tgt_img, pred_img, channel_axis=2, data_range=1.0)
            batch_ssim.append(ssim_val)
            
            # 3. Root Mean Squared Error (precision float tracking)
            rmse_val = np.sqrt(np.mean((pred_img - tgt_img) ** 2))
            batch_rmse.append(rmse_val)
            
        return {
            'psnr': float(np.mean(batch_psnr)),
            'ssim': float(np.mean(batch_ssim)),
            'rmse': float(np.mean(batch_rmse))
        }
