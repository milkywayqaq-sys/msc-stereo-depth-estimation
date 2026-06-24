import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Paths
# =========================

root = Path.home() / "msc_stereo_depth"
kitti = root / "data" / "KITTI_2015" / "training"

left_dir = kitti / "image_2"
right_dir = kitti / "image_3"
gt_dir = kitti / "disp_occ_0"

figure_dir = root / "outputs" / "figures"
metric_dir = root / "outputs" / "metrics"
bm_dir = root / "outputs" / "disparities" / "bm"
sgbm_dir = root / "outputs" / "disparities" / "sgbm"

figure_dir.mkdir(parents=True, exist_ok=True)
metric_dir.mkdir(parents=True, exist_ok=True)
bm_dir.mkdir(parents=True, exist_ok=True)
sgbm_dir.mkdir(parents=True, exist_ok=True)


# =========================
# Helper functions
# =========================

def load_kitti_disparity(path):
    """
    KITTI ground-truth disparity is stored as uint16 PNG.
    Real disparity = raw value / 256.0
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read disparity file: {path}")
    return raw.astype(np.float32) / 256.0


def compute_metrics(pred, gt):
    """
    EPE is computed on pixels where both GT and prediction are valid.
    Bad-3px treats invalid predictions as bad pixels over valid GT regions.
    """
    gt_valid = gt > 0
    pred_valid = pred > 0

    common_valid = gt_valid & pred_valid

    if np.sum(common_valid) == 0:
        return np.nan, np.nan, 0.0

    error = np.abs(pred - gt)

    epe = np.mean(error[common_valid])

    bad_pixels = ((error > 3.0) | (~pred_valid)) & gt_valid
    bad3 = np.mean(bad_pixels[gt_valid]) * 100.0

    valid_ratio = np.sum(common_valid) / np.sum(gt_valid) * 100.0

    return epe, bad3, valid_ratio


def save_disparity_visualization(disp, save_path, vmax=128):
    """
    Save disparity map as a colored image.
    """
    disp_vis = np.copy(disp)
    disp_vis[disp_vis < 0] = 0
    disp_vis = np.clip(disp_vis, 0, vmax)

    plt.figure(figsize=(10, 4))
    plt.imshow(disp_vis, cmap="plasma", vmin=0, vmax=vmax)
    plt.colorbar(label="Disparity (pixels)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_comparison_figure(left, gt, bm_disp, sgbm_disp, save_path):
    """
    Save a comparison figure for one stereo pair.
    """
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    gt_vis = np.where(gt > 0, gt, np.nan)
    bm_vis = np.where(bm_disp > 0, bm_disp, np.nan)
    sgbm_vis = np.where(sgbm_disp > 0, sgbm_disp, np.nan)

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.imshow(left_rgb)
    plt.title("Left Image")
    plt.axis("off")

    plt.subplot(4, 1, 2)
    plt.imshow(gt_vis, cmap="plasma")
    plt.title("Ground Truth Disparity")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    plt.subplot(4, 1, 3)
    plt.imshow(bm_vis, cmap="plasma")
    plt.title("StereoBM Estimated Disparity")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    plt.subplot(4, 1, 4)
    plt.imshow(sgbm_vis, cmap="plasma")
    plt.title("SGBM Estimated Disparity")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# Main experiment
# =========================

left_files = sorted(left_dir.glob("*_10.png"))

# Start with only 5 image pairs first.
# After confirming everything works, you can change this to 20 or 50.
num_images = 5
left_files = left_files[:num_images]

results = []

print(f"Number of image pairs to process: {len(left_files)}")

for left_path in left_files:
    image_name = left_path.name
    image_id = image_name.replace("_10.png", "")

    right_path = right_dir / image_name
    gt_path = gt_dir / image_name

    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    gt = load_kitti_disparity(gt_path)

    if left is None:
        raise FileNotFoundError(f"Cannot read left image: {left_path}")
    if right is None:
        raise FileNotFoundError(f"Cannot read right image: {right_path}")

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # StereoBM
    # -------------------------
    bm = cv2.StereoBM_create(
        numDisparities=128,
        blockSize=15
    )

    start = time.perf_counter()
    bm_disp = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    bm_runtime = (time.perf_counter() - start) * 1000.0

    bm_epe, bm_bad3, bm_valid_ratio = compute_metrics(bm_disp, gt)

    results.append([
        image_id,
        "StereoBM",
        bm_epe,
        bm_bad3,
        bm_runtime,
        bm_valid_ratio
    ])

    # -------------------------
    # SGBM
    # -------------------------
    block_size = 5

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    start = time.perf_counter()
    sgbm_disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    sgbm_runtime = (time.perf_counter() - start) * 1000.0

    sgbm_epe, sgbm_bad3, sgbm_valid_ratio = compute_metrics(sgbm_disp, gt)

    results.append([
        image_id,
        "SGBM",
        sgbm_epe,
        sgbm_bad3,
        sgbm_runtime,
        sgbm_valid_ratio
    ])

    # -------------------------
    # Save visualisations
    # -------------------------
    save_disparity_visualization(
        bm_disp,
        bm_dir / f"{image_id}_bm.png"
    )

    save_disparity_visualization(
        sgbm_disp,
        sgbm_dir / f"{image_id}_sgbm.png"
    )

    save_comparison_figure(
        left,
        gt,
        bm_disp,
        sgbm_disp,
        figure_dir / f"{image_id}_comparison.png"
    )

    print(
        f"Processed {image_id}: "
        f"BM EPE={bm_epe:.3f}, BM Bad3={bm_bad3:.2f}%, "
        f"SGBM EPE={sgbm_epe:.3f}, SGBM Bad3={sgbm_bad3:.2f}%"
    )


# =========================
# Save CSV
# =========================

csv_path = metric_dir / "bm_sgbm_kitti_results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id",
        "method",
        "EPE",
        "Bad-3px (%)",
        "Runtime (ms)",
        "Valid prediction ratio (%)"
    ])
    writer.writerows(results)

print("\nExperiment completed successfully.")
print(f"Saved metrics to: {csv_path}")
print(f"Saved comparison figures to: {figure_dir}")
print(f"Saved BM disparity maps to: {bm_dir}")
print(f"Saved SGBM disparity maps to: {sgbm_dir}")
