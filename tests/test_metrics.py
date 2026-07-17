# tests/test_metrics.py
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import RemoteSensingMetrics

def verify_precision_metrics():
    print("=== Phase 2.6 Refactored Metrics Verification ===")
    dummy_target = torch.tanh(torch.randn(2, 3, 256, 256))
    dummy_pred = dummy_target + torch.randn(2, 3, 256, 256) * 0.02
    
    metric_engine = RemoteSensingMetrics()
    scores = metric_engine.compute(dummy_target, dummy_pred)
    
    print("\n--- High-Precision Structural Evaluation Output ---")
    print(f"    -> Batch Mean PSNR: {scores['psnr']:.2f} dB")
    print(f"    -> Batch Mean SSIM: {scores['ssim']:.4f}")
    print(f"    -> Batch Mean RMSE: {scores['rmse']:.4f}")
    print("----------------------------------------------------")
    
    assert scores['psnr'] > 0, "PSNR logic error!"
    assert 0 <= scores['ssim'] <= 1.0, "SSIM boundary calculation error!"
    assert scores['rmse'] >= 0, "RMSE logic error!"
    print("[✓] ALL HACKATHON METRICS EXECUTING SUCCESSFULLY.")

if __name__ == "__main__":
    verify_precision_metrics()
