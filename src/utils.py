from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Path utilities
# ============================================================

def get_project_root() -> Path:
    """
    Return the project root directory.
    This assumes utils.py is located inside the src/ folder.
    """
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_kitti_root() -> Path:
    """
    Return the KITTI_2015 dataset root.

    Priority:
    1. Environment variable KITTI_2015_ROOT
    2. Default local path: project_root/data/KITTI_2015

    Example expected structure:
    KITTI_2015/
        training/
            image_2/
            image_3/
            disp_occ_0/
    """
    import os

    env_path = os.environ.get("KITTI_2015_ROOT")

    if env_path is not None:
        return Path(env_path)

    return get_project_root() / "data" / "KITTI_2015"


def get_kitti_training_dir() -> Path:
    """
    Return KITTI training directory.
    """
    return get_kitti_root() / "training"


def get_output_dirs():
    """
    Return commonly used output directories.
    """
    root = get_project_root()

    dirs = {
        "figures": root / "outputs" / "figures",
        "selected_examples": root / "outputs" / "figures" / "selected_examples",
        "metrics": root / "outputs" / "metrics",
        "summaries": root / "outputs" / "summaries",
        "bm_disparities": root / "outputs" / "disparities" / "bm",
        "sgbm_disparities": root / "outputs" / "disparities" / "sgbm",
    }

    for d in dirs.values():
        ensure_dir(d)

    return dirs


# ============================================================
# KITTI loading utilities
# ============================================================

def list_kitti_image_ids(num_images=None):
    """
    List KITTI image IDs based on image_2/*_10.png.

    Returns:
        List of image IDs such as ["000000", "000001", ...]
    """
    training_dir = get_kitti_training_dir()
    left_dir = training_dir / "image_2"

    if not left_dir.exists():
        raise FileNotFoundError(
        f"Cannot find KITTI left image directory:\n{left_dir}\n\n"
        "Please download KITTI 2015 stereo dataset and place it as:\n"
        "data/KITTI_2015/training/image_2\n"
        "data/KITTI_2015/training/image_3\n"
        "data/KITTI_2015/training/disp_occ_0\n\n"
        "Alternatively, set KITTI_2015_ROOT to the folder that contains 'training'."
    )

    left_files = sorted(left_dir.glob("*_10.png"))
    image_ids = [p.name.replace("_10.png", "") for p in left_files]

    if num_images is not None:
        image_ids = image_ids[:num_images]

    return image_ids


def get_kitti_paths(image_id: str):
    """
    Return left image, right image, and ground-truth disparity paths for one KITTI image ID.
    """
    training_dir = get_kitti_training_dir()

    filename = f"{image_id}_10.png"

    left_path = training_dir / "image_2" / filename
    right_path = training_dir / "image_3" / filename
    gt_path = training_dir / "disp_occ_0" / filename

    return left_path, right_path, gt_path


def load_color_image(path: Path):
    """
    Load a color image using OpenCV.
    OpenCV loads images in BGR format.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return img


def load_gray_image(path: Path):
    """
    Load a grayscale image using OpenCV.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Cannot read grayscale image: {path}")

    return img


def load_kitti_disparity(path: Path):
    """
    Load KITTI ground-truth disparity.

    KITTI disparity PNG is stored as uint16.
    Real disparity value = raw_value / 256.0

    Invalid pixels are usually stored as 0.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if raw is None:
        raise FileNotFoundError(f"Cannot read disparity file: {path}")

    disp = raw.astype(np.float32) / 256.0

    return disp


def load_kitti_sample(image_id: str):
    """
    Load left image, right image, and ground-truth disparity for one KITTI sample.
    """
    left_path, right_path, gt_path = get_kitti_paths(image_id)

    left = load_color_image(left_path)
    right = load_color_image(right_path)
    gt_disp = load_kitti_disparity(gt_path)

    return left, right, gt_disp


# ============================================================
# Stereo method utilities
# ============================================================

def create_stereobm(num_disparities=128, block_size=15):
    """
    Create OpenCV StereoBM matcher.

    num_disparities must be divisible by 16.
    block_size must be an odd number.
    """
    return cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )


def create_sgbm(num_disparities=128, block_size=5):
    """
    Create OpenCV SGBM matcher.
    """
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
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


def compute_disparity(matcher, left_gray, right_gray):
    """
    Compute disparity map using an OpenCV stereo matcher.

    OpenCV returns fixed-point disparity scaled by 16.
    Real disparity = raw disparity / 16.0
    """
    disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disp


# ============================================================
# Evaluation utilities
# ============================================================

def compute_metrics(pred_disp, gt_disp):
    """
    Compute EPE, Bad-3px, and valid prediction ratio.

    EPE:
        Mean absolute disparity error over pixels where both GT and prediction are valid.

    Bad-3px:
        Percentage of GT-valid pixels where error > 3 pixels,
        treating invalid predictions as bad.

    Valid prediction ratio:
        Percentage of GT-valid pixels where prediction is also valid.
    """
    gt_valid = gt_disp > 0
    pred_valid = pred_disp > 0

    common_valid = gt_valid & pred_valid

    if np.sum(gt_valid) == 0:
        return np.nan, np.nan, 0.0

    if np.sum(common_valid) == 0:
        return np.nan, 100.0, 0.0

    error = np.abs(pred_disp - gt_disp)

    epe = np.mean(error[common_valid])

    bad_pixels = ((error > 3.0) | (~pred_valid)) & gt_valid
    bad3 = np.mean(bad_pixels[gt_valid]) * 100.0

    valid_ratio = np.sum(common_valid) / np.sum(gt_valid) * 100.0

    return epe, bad3, valid_ratio


# ============================================================
# Visualisation utilities
# ============================================================

def disparity_to_display(disp):
    """
    Convert disparity to display format.
    Invalid disparity values are set to NaN.
    """
    return np.where(disp > 0, disp, np.nan)


def save_disparity_figure(disp, save_path: Path, title="Disparity", vmax=128):
    """
    Save a single disparity map figure.
    """
    ensure_dir(save_path.parent)

    disp_vis = disparity_to_display(disp)

    plt.figure(figsize=(10, 4))
    plt.imshow(disp_vis, cmap="plasma", vmin=0, vmax=vmax)
    plt.title(title)
    plt.colorbar(label="Disparity")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_comparison_figure(left, gt_disp, bm_disp, sgbm_disp, save_path: Path, vmax=128):
    """
    Save comparison figure:
    left image, ground truth, StereoBM, and SGBM.
    """
    ensure_dir(save_path.parent)

    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    gt_vis = disparity_to_display(gt_disp)
    bm_vis = disparity_to_display(bm_disp)
    sgbm_vis = disparity_to_display(sgbm_disp)

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.imshow(left_rgb)
    plt.title("Left Image")
    plt.axis("off")

    plt.subplot(4, 1, 2)
    plt.imshow(gt_vis, cmap="plasma", vmin=0, vmax=vmax)
    plt.title("Ground Truth Disparity")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    plt.subplot(4, 1, 3)
    plt.imshow(bm_vis, cmap="plasma", vmin=0, vmax=vmax)
    plt.title("StereoBM Estimated Disparity")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    plt.subplot(4, 1, 4)
    plt.imshow(sgbm_vis, cmap="plasma", vmin=0, vmax=vmax)
    plt.title("SGBM Estimated Disparity")
    plt.colorbar(label="Disparity")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":
    print("Project root:", get_project_root())
    print("KITTI root:", get_kitti_root())

    image_ids = list_kitti_image_ids(num_images=5)
    print("First KITTI image IDs:", image_ids)

    left, right, gt = load_kitti_sample(image_ids[0])
    print("Left image shape:", left.shape)
    print("Right image shape:", right.shape)
    print("GT disparity shape:", gt.shape)
    print("GT valid pixels:", np.sum(gt > 0))

    print("utils.py self-test completed successfully.")