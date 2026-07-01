"""
MASTER SCRIPT
Connects psnr.py, ssim.py, rmse.py, sam.py into one program.

Place this file in the SAME FOLDER as:
  psnr.py
  ssim.py
  rmse.py
  sam.py

It imports the compute_xxx() function from each file,
runs all four metrics on the same pair of images,
and also generates a combined result image.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Import the metric functions from each individual file ──
from psnr import compute_psnr
from ssim import compute_ssim
from rmse import compute_rmse
from sam import compute_sam


# ─────────────────────────────────────────
# Load Image
# ─────────────────────────────────────────
def load_image(path):
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: " + path)
    return Image.open(path).convert("RGB")


def get_path(prompt):
    while True:
        path = input(prompt).strip().strip('"').strip("'")
        if path == "":
            print("  Path cannot be empty. Try again.\n")
        elif not os.path.isfile(path):
            print("  File not found: " + path + "\n")
        else:
            return path


# ─────────────────────────────────────────
# Difference Heatmap
# ─────────────────────────────────────────
def make_heatmap(ref_arr, deg_arr):
    diff = np.abs(ref_arr.astype(np.float32) - deg_arr.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)
    norm = (diff_gray / diff_gray.max() * 255).astype(np.uint8) if diff_gray.max() > 0 else diff_gray.astype(np.uint8)
    h, w = norm.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap[..., 0] = norm
    heatmap[..., 1] = 255 - norm
    heatmap[..., 2] = 0
    return Image.fromarray(heatmap, "RGB")


def get_font(size=20):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ─────────────────────────────────────────
# Generate Combined Output Image
# ─────────────────────────────────────────
def generate_output(ref_img, deg_img, metrics, output_path):
    THUMB_W, THUMB_H = 400, 300
    PADDING, HEADER, FOOTER = 20, 50, 220

    ref_thumb = ref_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    deg_thumb = deg_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    ref_arr = np.array(ref_img.resize((THUMB_W, THUMB_H)), dtype=np.float32)
    deg_arr = np.array(deg_img.resize((THUMB_W, THUMB_H)), dtype=np.float32)
    heat = make_heatmap(ref_arr, deg_arr)

    total_w = THUMB_W * 3 + PADDING * 4
    total_h = HEADER + THUMB_H + PADDING * 2 + FOOTER

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    font_title = get_font(22)
    draw.text((PADDING, 12), "Image Quality Assessment — PSNR | SSIM | RMSE | SAM",
              fill=(255, 220, 50), font=font_title)

    y_img = HEADER + PADDING
    x1, x2, x3 = PADDING, PADDING * 2 + THUMB_W, PADDING * 3 + THUMB_W * 2
    canvas.paste(ref_thumb, (x1, y_img))
    canvas.paste(deg_thumb, (x2, y_img))
    canvas.paste(heat, (x3, y_img))

    font_label = get_font(18)
    lbl_y = y_img + THUMB_H + 6
    draw.text((x1 + 145, lbl_y), "REFERENCE", fill=(150, 220, 150), font=font_label)
    draw.text((x2 + 150, lbl_y), "DEGRADED", fill=(220, 150, 150), font=font_label)
    draw.text((x3 + 110, lbl_y), "DIFFERENCE HEATMAP", fill=(150, 180, 255), font=font_label)

    font_head, font_val, font_grade = get_font(20), get_font(18), get_font(16)
    mx = PADDING
    my = y_img + THUMB_H + 40
    draw.text((mx, my), "Metric Results:", fill=(255, 220, 50), font=font_head)
    my += 32

    psnr, ssim, rmse, sam = metrics["PSNR"], metrics["SSIM"], metrics["RMSE"], metrics["SAM"]

    psnr_str = "inf" if psnr == float("inf") else "{:.4f} dB".format(psnr)
    draw.text((mx, my), "PSNR  : " + psnr_str, fill=(255, 255, 255), font=font_val)
    my += 28
    draw.text((mx, my), "SSIM  : {:.6f}".format(ssim), fill=(255, 255, 255), font=font_val)
    my += 28
    draw.text((mx, my), "RMSE  : {:.4f}".format(rmse), fill=(255, 255, 255), font=font_val)
    my += 28
    draw.text((mx, my), "SAM   : {:.6f} rad".format(sam), fill=(255, 255, 255), font=font_val)

    canvas.save(output_path)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  MASTER PROGRAM — PSNR + SSIM + RMSE + SAM")
    print("  (imports psnr.py, ssim.py, rmse.py, sam.py)")
    print("=" * 55)

    ref_path = get_path("\n  Enter REFERENCE image path: ")
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

    print("\n  Running all 4 metrics ...")
    metrics = {
        "PSNR": compute_psnr(ref_arr, deg_arr),
        "SSIM": compute_ssim(ref_arr, deg_arr),
        "RMSE": compute_rmse(ref_arr, deg_arr),
        "SAM":  compute_sam(ref_arr, deg_arr),
    }

    ref_dir = os.path.dirname(os.path.abspath(ref_path))
    output_path = os.path.join(ref_dir, "quality_result.png")
    generate_output(ref_img, deg_img, metrics, output_path)

    print("\n  ── Results ──────────────────────────")
    print("  PSNR : " + ("inf" if metrics["PSNR"] == float("inf") else "{:.4f} dB".format(metrics["PSNR"])))
    print("  SSIM : {:.6f}".format(metrics["SSIM"]))
    print("  RMSE : {:.4f}".format(metrics["RMSE"]))
    print("  SAM  : {:.6f} rad".format(metrics["SAM"]))
    print("  ─────────────────────────────────────")
    print("\n  Output image saved to: " + output_path)

    input("\n  Press Enter to exit.")


main()
