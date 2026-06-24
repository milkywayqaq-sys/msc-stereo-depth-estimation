import csv
import time
import cv2

from utils import (
    list_kitti_image_ids,
    load_kitti_sample,
    create_stereobm,
    create_sgbm,
    compute_disparity,
    compute_metrics,
    get_output_dirs,
    save_disparity_figure,
    save_comparison_figure,
)


# ============================================================
# Experiment configuration
# ============================================================

NUM_IMAGES = 20

BM_NUM_DISPARITIES = 128
BM_BLOCK_SIZE = 15

SGBM_NUM_DISPARITIES = 128
SGBM_BLOCK_SIZE = 5

SAVE_SELECTED_EXAMPLES = True
NUM_SELECTED_EXAMPLES = 5


# ============================================================
# Main experiment
# ============================================================

def main():
    output_dirs = get_output_dirs()

    image_ids = list_kitti_image_ids(num_images=NUM_IMAGES)

    print(f"Number of image pairs to process: {len(image_ids)}")
    print(f"StereoBM: numDisparities={BM_NUM_DISPARITIES}, blockSize={BM_BLOCK_SIZE}")
    print(f"SGBM: numDisparities={SGBM_NUM_DISPARITIES}, blockSize={SGBM_BLOCK_SIZE}")

    results = []

    for idx, image_id in enumerate(image_ids):
        left, right, gt_disp = load_kitti_sample(image_id)

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # -------------------------
        # StereoBM
        # -------------------------
        bm = create_stereobm(
            num_disparities=BM_NUM_DISPARITIES,
            block_size=BM_BLOCK_SIZE
        )

        start_time = time.perf_counter()
        bm_disp = compute_disparity(bm, left_gray, right_gray)
        bm_runtime = (time.perf_counter() - start_time) * 1000.0

        bm_epe, bm_bad3, bm_valid_ratio = compute_metrics(bm_disp, gt_disp)

        results.append({
            "image_id": image_id,
            "method": "StereoBM",
            "EPE": bm_epe,
            "Bad-3px (%)": bm_bad3,
            "Runtime (ms)": bm_runtime,
            "Valid prediction ratio (%)": bm_valid_ratio
        })

        # -------------------------
        # SGBM
        # -------------------------
        sgbm = create_sgbm(
            num_disparities=SGBM_NUM_DISPARITIES,
            block_size=SGBM_BLOCK_SIZE
        )

        start_time = time.perf_counter()
        sgbm_disp = compute_disparity(sgbm, left_gray, right_gray)
        sgbm_runtime = (time.perf_counter() - start_time) * 1000.0

        sgbm_epe, sgbm_bad3, sgbm_valid_ratio = compute_metrics(sgbm_disp, gt_disp)

        results.append({
            "image_id": image_id,
            "method": "SGBM",
            "EPE": sgbm_epe,
            "Bad-3px (%)": sgbm_bad3,
            "Runtime (ms)": sgbm_runtime,
            "Valid prediction ratio (%)": sgbm_valid_ratio
        })

        # -------------------------
        # Save disparity maps
        # -------------------------
        save_disparity_figure(
            bm_disp,
            output_dirs["bm_disparities"] / f"{image_id}_bm.png",
            title=f"StereoBM Disparity: {image_id}",
            vmax=128
        )

        save_disparity_figure(
            sgbm_disp,
            output_dirs["sgbm_disparities"] / f"{image_id}_sgbm.png",
            title=f"SGBM Disparity: {image_id}",
            vmax=128
        )

        comparison_path = output_dirs["figures"] / f"{image_id}_comparison.png"

        save_comparison_figure(
            left,
            gt_disp,
            bm_disp,
            sgbm_disp,
            comparison_path,
            vmax=128
        )

        if SAVE_SELECTED_EXAMPLES and idx < NUM_SELECTED_EXAMPLES:
            selected_path = output_dirs["selected_examples"] / f"{image_id}_comparison.png"

            save_comparison_figure(
                left,
                gt_disp,
                bm_disp,
                sgbm_disp,
                selected_path,
                vmax=128
            )

        print(
            f"Processed {image_id}: "
            f"BM EPE={bm_epe:.3f}, BM Bad3={bm_bad3:.2f}%, "
            f"BM Runtime={bm_runtime:.2f} ms | "
            f"SGBM EPE={sgbm_epe:.3f}, SGBM Bad3={sgbm_bad3:.2f}%, "
            f"SGBM Runtime={sgbm_runtime:.2f} ms"
        )

    # -------------------------
    # Save CSV results
    # -------------------------
    csv_path = output_dirs["metrics"] / "bm_sgbm_kitti_results.csv"

    fieldnames = [
        "image_id",
        "method",
        "EPE",
        "Bad-3px (%)",
        "Runtime (ms)",
        "Valid prediction ratio (%)"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nExperiment completed successfully.")
    print(f"Saved metrics to: {csv_path}")
    print(f"Saved comparison figures to: {output_dirs['figures']}")
    print(f"Saved selected examples to: {output_dirs['selected_examples']}")
    print(f"Saved BM disparity maps to: {output_dirs['bm_disparities']}")
    print(f"Saved SGBM disparity maps to: {output_dirs['sgbm_disparities']}")


if __name__ == "__main__":
    main()