import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
root = Path.home() / "msc_stereo_depth"
kitti = root / "data" / "KITTI_2015" / "training"

left_path = kitti / "image_2" / "000000_10.png"
right_path = kitti / "image_3" / "000000_10.png"
disp_path = kitti / "disp_occ_0" / "000000_10.png"

output_dir = root / "outputs" / "kitti_check"
output_dir.mkdir(parents=True, exist_ok=True)

# Read images
left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

# KITTI disparity is stored as uint16 PNG.
# Real disparity value = stored value / 256.0
disp_raw = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
disp = disp_raw.astype(np.float32) / 256.0

if left is None:
    raise FileNotFoundError(f"Cannot read left image: {left_path}")
if right is None:
    raise FileNotFoundError(f"Cannot read right image: {right_path}")
if disp_raw is None:
    raise FileNotFoundError(f"Cannot read disparity image: {disp_path}")

print("Left image shape:", left.shape)
print("Right image shape:", right.shape)
print("GT disparity shape:", disp.shape)
print("GT disparity min:", np.min(disp))
print("GT disparity max:", np.max(disp))
print("Valid GT pixels:", np.sum(disp > 0))

# Convert BGR to RGB for matplotlib
left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

# Visualize GT disparity
valid_disp = np.where(disp > 0, disp, np.nan)

plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.imshow(left_rgb)
plt.title("KITTI Left Image: image_2/000000_10.png")
plt.axis("off")

plt.subplot(3, 1, 2)
plt.imshow(right_rgb)
plt.title("KITTI Right Image: image_3/000000_10.png")
plt.axis("off")

plt.subplot(3, 1, 3)
plt.imshow(valid_disp, cmap="plasma")
plt.title("KITTI Ground Truth Disparity: disp_occ_0/000000_10.png")
plt.colorbar(label="Disparity (pixels)")
plt.axis("off")

plt.tight_layout()
save_path = output_dir / "kitti_sample_check.png"
plt.savefig(save_path, dpi=150)
plt.close()

print(f"Saved visualization to: {save_path}")
print("KITTI reading test completed successfully.")

