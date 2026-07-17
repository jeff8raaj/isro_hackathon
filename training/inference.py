# training/inference.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from training.metrics import RemoteSensingMetrics

class Pix2PixInference:
    """
    Handles runtime generation, performance auditing, and visual triad generation
    using the high-precision floating-point metrics engine.
    """
    def __init__(self, netG, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.netG = netG.to(self.device)
        self.netG.eval()
        self.metric_engine = RemoteSensingMetrics()

    def process_sample(self, cloudy_tensor, clear_tensor=None):
        """Runs evaluation over isolated tensor frames."""
        cloudy_tensor = cloudy_tensor.to(self.device)
        if cloudy_tensor.dim() == 3:
            cloudy_tensor = cloudy_tensor.unsqueeze(0)
            
        with torch.no_grad():
            fake_clear = self.netG(cloudy_tensor)
            
        metrics = None
        if clear_tensor is not None:
            if clear_tensor.dim() == 3:
                clear_tensor = clear_tensor.unsqueeze(0)
            clear_tensor = clear_tensor.to(self.device)
            metrics = self.metric_engine.compute(clear_tensor, fake_clear)
            
        return fake_clear, metrics

    def save_triad_plot(self, cloudy_tensor, clear_tensor, save_path):
        """Generates and persists a side-by-side validation comparison figure."""
        self.netG.eval()
        fake_clear, metrics = self.process_sample(cloudy_tensor, clear_tensor)
        
        # Convert tensors to printable [0, 1] numpy float32 formats for matplotlib display
        cloudy_img = self.metric_engine.tensor_to_numpy_float32(cloudy_tensor)[0]
        clear_img = self.metric_engine.tensor_to_numpy_float32(clear_tensor)[0]
        fake_img = self.metric_engine.tensor_to_numpy_float32(fake_clear)[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cloudy_img)
        axes[0].set_title("Input: Cloudy Patch")
        axes[0].axis('off')
        
        axes[1].imshow(fake_img)
        title_str = "Prediction: Regenerated"
        if metrics:
            title_str += f"\nPSNR: {metrics['psnr']:.2f}dB | SSIM: {metrics['ssim']:.3f} | RMSE: {metrics['rmse']:.4f}"
        axes[1].set_title(title_str)
        axes[1].axis('off')
        
        axes[2].imshow(clear_img)
        axes[2].set_title("Target: Clear Ground Truth")
        axes[2].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return metrics
