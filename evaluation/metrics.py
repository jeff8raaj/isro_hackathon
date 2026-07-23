import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


class RemoteSensingMetrics:

    @staticmethod
    def rmse(pred, target):
        return torch.sqrt(F.mse_loss(pred, target))

    @staticmethod
    def psnr(pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        # (C,H,W) -> (H,W,C)
        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

        return peak_signal_noise_ratio(
            target,
            pred,
            data_range=1.0
        )

    @staticmethod
    def ssim(pred, target):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        pred = np.transpose(pred, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))

        return structural_similarity(
            target,
            pred,
            channel_axis=2,
            data_range=1.0
        )

    @staticmethod
    def sam(pred, target):
        # pred and target shape: (C, H, W)
        dot_product = (pred * target).sum(dim=0)

        norm_pred = torch.norm(pred, dim=0)
        norm_target = torch.norm(target, dim=0)

        cos_theta = dot_product / (norm_pred * norm_target + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        return torch.acos(cos_theta).mean()
