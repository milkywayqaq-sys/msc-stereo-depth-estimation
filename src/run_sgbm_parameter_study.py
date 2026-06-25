import csv
import time
import cv2
import matplotlib.pyplot as plt

from utils import (
    list_kitti_image_ids,
    load_kitti_sample,
    create_sgbm,
    compute_disparity,
    compute_metrics,
    get_output_dirs,
    save_disparity_figure,
)


# ============================================================
# Experiment configuration
# ============================================================

NUM_IMAGES = 20

NUM_DISPARITIES = 128

BLOCK_SIZES = [3, 5, 7, 9, 11]

SAVE_EXAMPLE_IMAGES = True
EXAMPLE_IMAGE_ID = "000000"


# ============================================================
# Helper functions
# ============================================================

def compute_average(values):
    return sum(values) / len(values) if len(values) > 0 else float("nan")


def save_summary_plot(summary_rows, save_path):
    """
    Save plots showing how blockSize affects EPE, Bad-3px, runtime,
    and valid prediction ratio.
    """
    block_sizes = [row["block_size"] for row in summary_rows]
    avg_epe = [row["average_EPE"] for row in summary_rows]
    avg_bad3 = [row["average_Bad-3px (%)"] for row in summary_rows]
    avg_runtime = [row["average_Runtime (ms)"] for row in summary_rows]
    avg_valid = [row["average_Valid prediction ratio (%)"] for row in summary_rows]

    plt.figure(figsize=(10, 6))

    plt.plot(block_sizes, avg_epe, marker="o", label="Average EPE")
    plt.plot(block_sizes, avg_bad3, marker="o", label="Average Bad-3px (%)")
    plt.plot(block_sizes, avg_runtime, marker="o", label="Average Runtime (ms)")
    plt.plot(block_sizes, avg_valid, marker="o", label="Average Valid Ratio (%)")

    plt.xlabel("SGBM blockSize")
    plt.ylabel("Metric value")
    plt.title("SGBM Parameter Sensitivity: blockSize")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Main experiment
# ============================================================

def main():
    output_dirs = get_output_dirs()

    parameter_study_dir = output_dirs["figures"] / "sgbm_parameter_study"
    parameter_study_dir.mkdir(parents=True, exist_ok=True)

    image_ids = list_kitti_image_ids(num_images=NUM_IMAGES)

    print(f"Number of KITTI image pairs: {len(image_ids)}")
    print(f"Testing SGBM block sizes: {BLOCK_SIZES}")
    print(f"Fixed numDisparities: {NUM_DISPARITIES}")

    detailed_results = []
    summary_results = []

    for block_size in BLOCK_SIZES:
        print("\n" + "=" * 80)
        print(f"Running SGBM with blockSize = {block_size}")
        print("=" * 80)

        epe_list = []
        bad3_list = []
        runtime_list = []
        valid_ratio_list = []

        for image_id in image_ids:
            left, right, gt_disp = load_kitti_sample(image_id)

            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

            sgbm = create_sgbm(
                num_disparities=NUM_DISPARITIES,
                block_size=block_size
            )

            start_time = time.perf_counter()
            pred_disp = compute_disparity(sgbm, left_gray, right_gray)
            runtime_ms = (time.perf_counter() - start_time) * 1000.0

            epe, bad3, valid_ratio = compute_metrics(pred_disp, gt_disp)

            epe_list.append(epe)
            bad3_list.append(bad3)
            runtime_list.append(runtime_ms)
            valid_ratio_list.append(valid_ratio)

            detailed_results.append({
                "image_id": image_id,
                "method": "SGBM",
                "block_size": block_size,
                "num_disparities": NUM_DISPARITIES,
                "EPE": epe,
                "Bad-3px (%)": bad3,
                "Runtime (ms)": runtime_ms,
                "Valid prediction ratio (%)": valid_ratio
            })

            if SAVE_EXAMPLE_IMAGES and image_id == EXAMPLE_IMAGE_ID:
                save_disparity_figure(
                    pred_disp,
                    parameter_study_dir / f"{image_id}_sgbm_blockSize_{block_size}.png",
                    title=f"SGBM Disparity: blockSize={block_size}",
                    vmax=128
                )

            print(
                f"Image {image_id}, blockSize={block_size}: "
                f"EPE={epe:.3f}, Bad3={bad3:.2f}%, "
                f"Runtime={runtime_ms:.2f} ms, Valid={valid_ratio:.2f}%"
            )

        avg_epe = compute_average(epe_list)
        avg_bad3 = compute_average(bad3_list)
        avg_runtime = compute_average(runtime_list)
        avg_valid = compute_average(valid_ratio_list)

        summary_row = {
            "block_size": block_size,
            "num_disparities": NUM_DISPARITIES,
            "num_samples": len(image_ids),
            "average_EPE": avg_epe,
            "average_Bad-3px (%)": avg_bad3,
            "average_Runtime (ms)": avg_runtime,
            "average_Valid prediction ratio (%)": avg_valid
        }

        summary_results.append(summary_row)

        print(
            f"\nAverage for blockSize={block_size}: "
            f"EPE={avg_epe:.3f}, Bad3={avg_bad3:.2f}%, "
            f"Runtime={avg_runtime:.2f} ms, Valid={avg_valid:.2f}%"
        )

    # ========================================================
    # Save detailed CSV
    # ========================================================

    detailed_csv_path = output_dirs["metrics"] / "sgbm_blocksize_study_detailed.csv"

    detailed_fieldnames = [
        "image_id",
        "method",
        "block_size",
        "num_disparities",
        "EPE",
        "Bad-3px (%)",
        "Runtime (ms)",
        "Valid prediction ratio (%)"
    ]

    with open(detailed_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)

    # ========================================================
    # Save summary CSV
    # ========================================================

    summary_csv_path = output_dirs["summaries"] / "sgbm_blocksize_study_summary.csv"

    summary_fieldnames = [
        "block_size",
        "num_disparities",
        "num_samples",
        "average_EPE",
        "average_Bad-3px (%)",
        "average_Runtime (ms)",
        "average_Valid prediction ratio (%)"
    ]

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_results)

    # ========================================================
    # Save summary plot
    # ========================================================

    plot_path = parameter_study_dir / "sgbm_blocksize_summary_plot.png"
    save_summary_plot(summary_results, plot_path)

    print("\nSGBM parameter sensitivity experiment completed successfully.")
    print(f"Saved detailed results to: {detailed_csv_path}")
    print(f"Saved summary results to: {summary_csv_path}")
    print(f"Saved summary plot to: {plot_path}")
    print(f"Saved example disparity maps to: {parameter_study_dir}")


if __name__ == "__main__":
    main()