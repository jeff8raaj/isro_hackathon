import numpy as np
from math import log10

def calculate_psnr(target, prediction):
    """Computes Peak Signal-to-Noise Ratio between ground truth and prediction."""
    mse = np.mean((target.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Computes a streamlined Structural Similarity Index for baseline metrics."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return numerator / denominator

def calculate_rmse(target, prediction):
    """Computes Root Mean Squared Error."""
    mse = np.mean((target.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    return np.sqrt(mse)

def calculate_sam(target, prediction):
    """
    Computes Spectral Angle Mapper (SAM) in degrees.
    Treats pixels as vectors and computes the angle between them.
    """
    t_flat = target.astype(np.float64).flatten()
    p_flat = prediction.astype(np.float64).flatten()
    
    dot_product = np.dot(t_flat, p_flat)
    norm_t = np.linalg.norm(t_flat)
    norm_p = np.linalg.norm(p_flat)
    
    if norm_t == 0 or norm_p == 0:
        return 0.0
    
    cos_theta = dot_product / (norm_t * norm_p)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Clip boundaries to prevent nan errors
    
    # Return angle converted from radians to degrees
    return np.degrees(np.arccos(cos_theta))