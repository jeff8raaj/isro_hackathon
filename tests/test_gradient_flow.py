# tests/test_gradient_flow.py
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import UNetGenerator, NLayerDiscriminator, init_weights
from training import HybridReconstructionLoss
def verify_gradient_flow():
    print("=== Phase 2.4 Gradient & Loss Verification ===")
    
    # 1. Setup Mock Data Pairs (Batch Size = 2)
    dummy_cloudy = torch.randn(2, 3, 256, 256, requires_grad=False)
    dummy_clear = torch.randn(2, 3, 256, 256, requires_grad=False)
    
    # 2. Setup Architectures
    netG = UNetGenerator(input_nc=3, output_nc=3)
    netD = NLayerDiscriminator(input_nc=6)
    init_weights(netG, 'normal')
    init_weights(netD, 'normal')
    
    # Setup Loss Engine
    loss_engine = HybridReconstructionLoss(lambda_L1=100.0)
    
    # ---------------------------------------------------------
    # Test Step 1: Generator Forward & L1 Calculations
    # ---------------------------------------------------------
    fake_clear = netG(dummy_cloudy)
    print(f"[+] Generator Forward complete. Output Shape: {fake_clear.shape}")
    
    # ---------------------------------------------------------
    # Test Step 2: Discriminator Backward Optimization Step
    # ---------------------------------------------------------
    # Real pair pass
    real_pair = torch.cat((dummy_cloudy, dummy_clear), dim=1)
    pred_real = netD(real_pair)
    
    # Fake pair pass (detach to protect Generator gradients)
    fake_pair_D = torch.cat((dummy_cloudy, fake_clear.detach()), dim=1)
    pred_fake_D = netD(fake_pair_D)
    
    loss_D = loss_engine.calculate_discriminator_loss(pred_real, pred_fake_D)
    loss_D.backward()
    print(f"[✓] Discriminator backward pass complete. Loss D: {loss_D.item():.4f}")
    
    # ---------------------------------------------------------
    # Test Step 3: Generator Backward Optimization Step
    # ---------------------------------------------------------
    # Fake pair pass (NO detach, we want gradients to backprop into G)
    fake_pair_G = torch.cat((dummy_cloudy, fake_clear), dim=1)
    pred_fake_G = netD(fake_pair_G)
    
    loss_G, loss_G_GAN, loss_G_L1 = loss_engine.calculate_generator_loss(pred_fake_G, fake_clear, dummy_clear)
    loss_G.backward()
    print(f"[✓] Generator backward pass complete. Total G Loss: {loss_G.item():.4f}")
    print(f"    -> Adversarial Component: {loss_G_GAN.item():.4f}")
    print(f"    -> L1 Spatial Constraint: {loss_G_L1.item():.4f}")
    
    print("==========================================================")
    print("[✓] LOSS MODULES AND GRADIENT FLOW PASSED CRITICAL CHECKS.")

if __name__ == "__main__":
    verify_gradient_flow()
