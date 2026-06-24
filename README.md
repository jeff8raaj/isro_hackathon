# GeoAI Satellite Cloud Removal Pipeline

An advanced remote sensing and deep learning pipeline designed to harmonize multi-sensor satellite imagery, mask atmospheric interference, and reconstruct obscured ground terrain using a Conditional Generative Adversarial Network (cGAN / U-Net) architecture.

## 👥 Team Roles & Responsibilities

* **Lead Data Engineer (ECE + AI):** Dataset Preprocessing, Image Enhancement Foundations, Automated Cloud/Shadow Masking, Repository Architecture.
* **AI Developer (Software/AI):** Model Scaling, Advanced Deep Learning Training Loops, Optimization Routines.
* **Validation & Interface Engineer (ECE Member 2):** Evaluation Metrics, Model Testing Suite, UI Dashboard Framework, Pitch Presentation.

---

## 📂 Project Architecture

```text
isro_hackathon/
│
├── data/                       # Local raw assets and generated tensor binaries
│   ├── dataset_patches/        # Cleanly sliced uniform image matrices (.npy)
│   ├── generator_checkpoint.pth # Trained model weight parameters
│   └── predicted_clean_patch.npy# Evaluated cloud-free reconstructed matrix
│
└── src/                        # Modular pipeline execution source code
    ├── cloud_masking.py        # CV brightness segmentation and shadow dilation
    ├── spatial_resample.py     # Bilinear interpolation geometric warping engine
    ├── patch_maker.py          # Automated sliding-window array slicing
    ├── dataset_loader.py       # Custom PyTorch Dataset and normalization class
    ├── models.py               # Symmetrical U-Net with skip-connection layers
    ├── train.py                # Adam optimization and MSE loss execution loop
    └── inference.py            # Checkpoint loading and scene reconstruction


    python3 src/cloud_masking.py
    