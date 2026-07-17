"""
SSIM (Structural Similarity Index)
Range: -1 to 1 (closer to 1 = more structurally similar)
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def load_image(path):
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: " + path)
    return Image.open(path).convert("RGB")


def compute_ssim(ref, deg):
    ref = ref.astype(np.float32)
    deg = deg.astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    W  = 11

    def ssim_channel(r, d):
        mu_r   = uniform_filter(r, W)
        mu_d   = uniform_filter(d, W)
        mu_r2  = mu_r * mu_r
        mu_d2  = mu_d * mu_d
        mu_rd  = mu_r * mu_d
        sig_r2 = uniform_filter(r * r, W) - mu_r2
        sig_d2 = uniform_filter(d * d, W) - mu_d2
        sig_rd = uniform_filter(r * d, W) - mu_rd
        num    = (2 * mu_rd + C1) * (2 * sig_rd + C2)
        den    = (mu_r2 + mu_d2 + C1) * (sig_r2 + sig_d2 + C2)
        return float(np.mean(num / den))

    scores = [ssim_channel(ref[..., c], deg[..., c]) for c in range(3)]
    return float(np.mean(scores))


def get_path(prompt):
    while True:
        path = input(prompt).strip().strip('"').strip("'")
        if path == "":
            print("  Path cannot be empty. Try again.\n")
        elif not os.path.isfile(path):
            print("  File not found: " + path + "\n")
        else:
            return path


def main():
    print("\n" + "=" * 45)
    print("  SSIM — Structural Similarity Index")
    print("=" * 45)

    ref_path = get_path("  Enter REFERENCE image path: ")
    deg_path = get_path("  Enter DEGRADED  image path: ")

    ref_img = load_image(ref_path)
    deg_img = load_image(deg_path)

    if ref_img.size != deg_img.size:
        print("\n  ERROR: Images must be the same size.")
        print("  Reference:", ref_img.size, " Degraded:", deg_img.size)
        input("\n  Press Enter to exit.")
        return

    ref_arr = np.array(ref_img)
    deg_arr = np.array(deg_img)

    ssim = compute_ssim(ref_arr, deg_arr)

    print("\n  Result:")
    print("  SSIM = {:.6f}".format(ssim))
    if ssim >= 0.95:
        print("  Similarity: Very High (>= 0.95)")
    elif ssim >= 0.80:
        print("  Similarity: High (0.80-0.95)")
    elif ssim >= 0.60:
        print("  Similarity: Moderate (0.60-0.80)")
    else:
        print("  Similarity: Low (< 0.60)")

    input("\n  Press Enter to exit.")


main()
