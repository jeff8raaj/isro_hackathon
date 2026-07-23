import os
import torch
import csv

from models import UNetGenerator
from .evaluator import Evaluator
from datasets import LISSDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    # ---------------------------------
    # 1. Load trained Generator
    # ---------------------------------
    checkpoint_path = "outputs/checkpoints/checkpoint_epoch_100.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model = UNetGenerator(
        input_nc=3,
        output_nc=3,
        num_downs=8,
        ngf=64
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["netG_state_dict"]
    )

    model.eval()


    # ---------------------------------
    # 2. Dataset
    # ---------------------------------
    dataset = LISSDataset(
        patch_dir="data/patches/247109911"
    )

    evaluator = Evaluator()

    num_samples = len(dataset)


    total_metrics = {
        "PSNR": 0.0,
        "RMSE": 0.0,
        "SAM": 0.0,
        "SSIM": 0.0
    }

    all_scores = []


    print(f"\nEvaluating {num_samples} patches...\n")


    # ---------------------------------
    # 3. Evaluation Loop
    # ---------------------------------
    with torch.no_grad():

        for i in range(num_samples):

            sample = dataset[i]

            inputs = sample["input"].unsqueeze(0).to(DEVICE)
            target = sample["target"].unsqueeze(0).to(DEVICE)


            prediction = model(inputs)


            scores = evaluator.evaluate(
                prediction.squeeze(0),
                target.squeeze(0)
            )


            # Store individual patch metrics
            all_scores.append({
                "Patch": i + 1,
                "PSNR": scores["PSNR"],
                "SSIM": scores["SSIM"],
                "RMSE": scores["RMSE"],
                "SAM": scores["SAM"]
            })


            # Accumulate totals
            for key, value in scores.items():
                total_metrics[key] += value


            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{num_samples} patches")


    # ---------------------------------
    # 4. Print Average Results
    # ---------------------------------
    print("\n========== RESULTS ==========\n")


    for key in sorted(total_metrics.keys()):

        average = total_metrics[key] / num_samples

        print(
            f"{key:5s}: {average:.5f}"
        )


    print("\n=============================")


    # ---------------------------------
    # 5. Save CSV
    # ---------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )


    csv_path = "outputs/evaluation_metrics.csv"


    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Patch",
                "PSNR",
                "SSIM",
                "RMSE",
                "SAM"
            ]
        )

        writer.writeheader()
        writer.writerows(all_scores)


    print(
        f"\nSaved detailed metrics to {csv_path}"
    )



if __name__ == "__main__":
    main()
