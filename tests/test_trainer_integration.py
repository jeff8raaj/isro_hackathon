# tests/test_trainer_integration.py
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import Pix2PixTrainer

def run_trainer_test():
    print("=== Phase 2.5 Trainer Integration Verification ===")
    
    # Mock config dictionary replicating the exact architecture of your config.yaml
    mock_config = {
        'model': {'input_nc': 3, 'output_nc': 3, 'ngf': 64, 'ndf': 64},
        'training': {'lambda_L1': 100.0, 'lr_g': 0.0002, 'lr_d': 0.0002, 'beta1': 0.5, 'beta2': 0.999}
    }
    
    # 1. Instantiate the orchestrator trainer instance
    trainer = Pix2PixTrainer(config=mock_config, device='cpu')
    print("[✓] Pix2PixTrainer successfully instantiated with hyperparameter maps.")
    
    # 2. Simulate standard tensor batch tracking structures
    dummy_cloudy = torch.randn(2, 3, 256, 256)
    dummy_clear = torch.randn(2, 3, 256, 256)
    
    # 3. Execute optimization pass
    metrics = trainer.optimize_batch(dummy_cloudy, dummy_clear)
    
    print("\n--- Step Optimization Execution Report ---")
    for key, val in metrics.items():
        print(f"    -> {key}: {val:.4f}")
        
    print("------------------------------------------")
    print("[✓] BACKPROPAGATION OPTIMIZATION STEP INTEGRATION SUCCESSFUL.")

if __name__ == "__main__":
    run_trainer_test()
