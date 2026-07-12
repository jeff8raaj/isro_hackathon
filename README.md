# Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery

**ISRO Bharatiya Antariksh Hackathon 2026**

A deep learning framework for automated cloud removal and surface reconstruction in high-resolution LISS-IV satellite imagery using Generative AI and multi-modal remote sensing data.

---

## Overview

Persistent cloud cover reduces the usability of optical satellite imagery for applications such as land use mapping, disaster monitoring, agriculture, and environmental assessment. This project develops a Generative AI-based framework that reconstructs cloud-covered regions while preserving spatial structures and spectral characteristics, producing analysis-ready imagery.

---

## Objectives

- Develop an automated cloud removal framework for LISS-IV imagery.
- Reconstruct cloud-covered regions using Generative AI.
- Preserve spatial and spectral information.
- Generate analysis-ready cloud-free imagery.
- Evaluate reconstruction quality using standard remote sensing metrics.
- Build a scalable workflow for operational deployment.

---

## System Architecture

```
Cloudy LISS-IV Image
        │
        ▼
Cloud Detection & Mask Generation
        │
        ▼
Image Preprocessing & Patch Extraction
        │
        ▼
Generative AI Model (cGAN/U-Net)
        │
        ▼
Multi-Modal Feature Fusion
(Sentinel-1, Sentinel-2, DEM, Temporal Images)
        │
        ▼
Cloud-Free Reconstruction
        │
        ▼
Quality Evaluation
(PSNR, SSIM, RMSE, SAM)
```

---

## Repository Structure

```
isro_hackathon/
│
├── data/
├── src/
├── models/
├── outputs/
├── notebooks/
├── requirements.txt
└── README.md


---

## Technology Stack

### Programming
- Python

### Deep Learning
- PyTorch
- cGAN
- U-Net
- Latent Diffusion Models (Future)
- Vision Transformers (Future)

### Geospatial
- GDAL
- Rasterio
- QGIS
- Google Earth Engine (Optional)

### Image Processing
- OpenCV
- NumPy
- Scikit-image
- Albumentations

---

## Dataset

### Primary Dataset

- LISS-IV Satellite Imagery (Bhoonidhi)

### Auxiliary Datasets

- Sentinel-1 SAR
- Sentinel-2 Optical Imagery
- DEM
- Temporal LISS-IV Imagery

---

## Evaluation Metrics

Model performance is evaluated using:

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Root Mean Square Error (RMSE)
- Spectral Angle Mapper (SAM)

---

## Current Implementation

The current baseline includes:

- Cloud mask generation
- Patch-based preprocessing
- cGAN/U-Net implementation
- Model training pipeline
- Inference pipeline
- Quantitative evaluation

---

## Future Enhancements

- Multi-modal data fusion
- Latent Diffusion Models
- Transformer-based feature fusion
- Temporal image-guided reconstruction
- Operational deployment pipeline

---

## Applications

- Disaster Management
- Agriculture
- Land Use/Land Cover Mapping
- Environmental Monitoring
- Urban Planning
- Infrastructure Assessment

---

## Team

**Team Name:** *Your Team Name*

- Sree Tharshan S
- Jeffrin S Raaj
- L Nibin Giovanni

---

## License

Developed for the ISRO Bharatiya Antariksh Hackathon 2026.

# ISRO Hackathon - Model Deployment & Visualization

This repository contains the trained generative model and visualization scripts for our project. Follow the steps below to set up the environment and verify the results.

## 📦 Project Assets
* `generator_checkpoint.pth` (24MB): The trained model weights.
* `results_comparison.png`: The visual evaluation output.

---

## 🚀 Quick Start Guide

Follow these steps in your terminal to set up the environment and run the visualization script:

### Step 1: Activate the Virtual Environment
Ensure you are in the project root directory, then run:
```bash
source venv/bin/activate