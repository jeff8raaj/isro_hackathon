"""
SAM (Spectral Angle Mapper)
Lower value (radians) = better spectral/colour similarity
"""

import os
import numpy as np
from PIL import Image


def load_image(path):
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: " + path)
    return Image.open(path).convert("RGB")


def compute_sam(ref, deg):
    r = ref.astype(np.float32).reshape(-1, 3)
    d = deg.astype(np.float32).reshape(-1, 3)

    dot   = np.sum(r * d, axis=1)
    nr    = np.linalg.norm(r, axis=1)
    nd    = np.linalg.norm(d, axis=1)
    denom = nr * nd
    valid = denom > 1e-8

    cos_a = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return float(np.mean(np.arccos(cos_a)))


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
    print("  SAM — Spectral Angle Mapper")
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

    sam_rad = compute_sam(ref_arr, deg_arr)
    sam_deg = math_degrees = sam_rad * (180.0 / np.pi)

    print("\n  Result:")
    print("  SAM = {:.6f} radians".format(sam_rad))
    print("  SAM = {:.4f} degrees".format(sam_deg))
    print("  (Lower means colour/spectral content is more similar)")

    input("\n  Press Enter to exit.")


main()
