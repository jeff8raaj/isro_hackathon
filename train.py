# train.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import RemoteSensingDataset
from training import Pix2PixTrainer, RemoteSensingMetrics, Pix2PixInference

def main():
    print("==================================================")
    print("      ISRO HACKATHON: PIX2PIX BASELINE RUNNER     ")
    print("==================================================\n")
    
    # 1. Load Configurations Safely
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration structure file at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("[+] Configuration properties parsed smoothly.")
    
    # Device configuration checking 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[+] Execution target architecture set to: {device.upper()}")
    
    # 2. Build Datasets & DataLoaders (RICE1 Baseline Mode)
    # Appending the exact internal directory structure verified via terminal ls checks
    target_data_path = os.path.join(config['data']['rice1_path'], "RICE", "RICE1")
    print(f"[*] Instantiating dataset matrices from path: {target_data_path}")
    
    # Instantiate using your exact verified signature (no split parameter)
    shared_dataset = RemoteSensingDataset(
        root_dir=target_data_path,
        patch_size=config['data']['patch_size']
    )
    
    # Utilizing the shared dataset for baseline validation cycles
    train_loader = DataLoader(
        shared_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        shared_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"[✓] Data setups initialized. Total Batches: {len(train_loader)}")
    
    # 3. Instantiate Operational Component Engines
    trainer = Pix2PixTrainer(config=config, device=device)
    metrics_engine = RemoteSensingMetrics()
    inference_engine = Pix2PixInference(trainer.netG, device=device)
    
    epochs = config['training']['epochs']
    print(f"\n[+] Booting training orchestration pipeline for {epochs} Epochs...")
    
    # 4. Master Epoch Training Loop Execution
    for epoch in range(1, epochs + 1):
        trainer.netG.train()
        trainer.netD.train()
        
        running_metrics = {'loss_D': 0.0, 'loss_G': 0.0, 'loss_G_GAN': 0.0, 'loss_G_L1': 0.0}
        
        for i, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                cloudy, clear = batch['cloudy'], batch['clear']
            else:
                cloudy, clear = batch
                
            step_losses = trainer.optimize_batch(cloudy, clear)
            
            for k, v in step_losses.items():
                running_metrics[k] += v
                
        # Calculate training epoch averages
        num_batches = len(train_loader)
        epoch_str = f"Epoch [{epoch}/{epochs}]"
        print(f"\n>>> {epoch_str} | Train Losses:")
        print(f"    -> D Adversarial: {running_metrics['loss_D']/num_batches:.4f}")
        print(f"    -> G Total Loss : {running_metrics['loss_G']/num_batches:.4f} (GAN: {running_metrics['loss_G_GAN']/num_batches:.4f}, L1: {running_metrics['loss_G_L1']/num_batches:.4f})")
        
        # 5. Operational Validation Check Phase
        trainer.netG.eval()
        val_scores = {'psnr': 0.0, 'ssim': 0.0, 'rmse': 0.0}
        
        with torch.no_grad():
            for val_batch in val_loader:
                if isinstance(val_batch, dict):
                    v_cloudy, v_clear = val_batch['cloudy'], val_batch['clear']
                else:
                    v_cloudy, v_clear = val_batch
                    
                v_cloudy = v_cloudy.to(device)
                v_clear = v_clear.to(device)
                
                v_fake = trainer.netG(v_cloudy)
                batch_scores = metrics_engine.compute(v_clear, v_fake)
                
                for metric_key in val_scores.keys():
                    val_scores[metric_key] += batch_scores[metric_key]
                    
        num_val_batches = len(val_loader)
        print(f"    Validation Metrics -> PSNR: {val_scores['psnr']/num_val_batches:.2f}dB | SSIM: {val_scores['ssim']/num_val_batches:.4f} | RMSE: {val_scores['rmse']/num_val_batches:.4f}")
        
        # 6. Periodic Visual Verification & Checkpoint Saving
        if epoch % 5 == 0 or epoch == 1:
            sample_batch = next(iter(val_loader))
            if isinstance(sample_batch, dict):
                s_cloudy, s_clear = sample_batch['cloudy'][0], sample_batch['clear'][0]
            else:
                s_cloudy, s_clear = sample_batch[0][0], sample_batch[1][0]
                
            triad_path = f"outputs/visuals/epoch_{epoch}_triad.png"
            inference_engine.save_triad_plot(s_cloudy, s_clear, triad_path)
            print(f"[✓] Validation visual triad generated: {triad_path}")
            
            trainer.save_checkpoint(epoch=epoch, directory="outputs/checkpoints")
            
    print("\n==================================================")
    print("[✓] RICE1 BASELINE DEVELOPMENT PASS COMPLETE.      ")
    print("==================================================")

if __name__ == "__main__":
    main()
