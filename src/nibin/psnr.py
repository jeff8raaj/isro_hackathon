"""
PSNR (Peak Signal-to-Noise Ratio)
Higher value = better quality (less noise/distortion)
"""

import os
import math
import numpy as np
from PIL import Image


def load_image(path):
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: " + path)
    return Image.open(path).convert("RGB")


def compute_psnr(ref, deg):
    mse = np.mean((ref.astype(np.float32) - deg.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


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
    print("  PSNR — Peak Signal-to-Noise Ratio")
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

    psnr = compute_psnr(ref_arr, deg_arr)

    print("\n  Result:")
    if psnr == float("inf"):
        print("  PSNR = inf  (images are identical)")
    else:
        print("  PSNR = {:.4f} dB".format(psnr))
        if psnr >= 40:
            print("  Quality: Excellent (>= 40 dB)")
        elif psnr >= 30:
            print("  Quality: Good (30-40 dB)")
        elif psnr >= 20:
            print("  Quality: Acceptable (20-30 dB)")
        else:
            print("  Quality: Poor (< 20 dB)")

    input("\n  Press Enter to exit.")


main()
