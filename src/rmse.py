"""
RMSE (Root Mean Square Error)
Lower value = better quality (less pixel deviation)
"""

import os
import numpy as np
from PIL import Image


def load_image(path):
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: " + path)
    return Image.open(path).convert("RGB")


def compute_rmse(ref, deg):
    ref = ref.astype(np.float32)
    deg = deg.astype(np.float32)
    return float(np.sqrt(np.mean((ref - deg) ** 2)))


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
    print("  RMSE — Root Mean Square Error")
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

    rmse = compute_rmse(ref_arr, deg_arr)

    print("\n  Result:")
    print("  RMSE = {:.4f}".format(rmse))
    print("  (Scale: 0-255, lower means images are closer)")

    input("\n  Press Enter to exit.")


main()
