# training/losses.py
import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """
    Stateless calculation engine for vanilla/PatchGAN loss profiles using BCEWithLogitsLoss.
    Automatically handles target tensor creation (all-ones or all-zeros) on the target device.
    """
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class HybridReconstructionLoss(nn.Module):
    """
    Combines PatchGAN adversarial mapping with L1 pixel constraint logic.
    Accepts raw prediction matrices and calculates independent value metrics.
    """
    def __init__(self, lambda_L1=100.0):
        super(HybridReconstructionLoss, self).__init__()
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_L1 = lambda_L1

    def calculate_discriminator_loss(self, pred_real, pred_fake):
        """Computes objective mapping loss for D"""
        loss_D_real = self.gan_loss(pred_real, target_is_real=True)
        loss_D_fake = self.gan_loss(pred_fake, target_is_real=False)
        return (loss_D_real + loss_D_fake) * 0.5

    def calculate_generator_loss(self, pred_fake, reconstructed, target_clear):
        """Computes objective mapping loss for G"""
        loss_G_GAN = self.gan_loss(pred_fake, target_is_real=True)
        loss_G_L1 = self.l1_loss(reconstructed, target_clear) * self.lambda_L1
        loss_total = loss_G_GAN + loss_G_L1
        return loss_total, loss_G_GAN, loss_G_L1
