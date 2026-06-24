import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Create two simple synthetic grayscale images
h, w = 240, 320
left = np.zeros((h, w), dtype=np.uint8)
right = np.zeros((h, w), dtype=np.uint8)

# Draw a white rectangle in the left image
cv2.rectangle(left, (100, 80), (180, 160), 255, -1)

# Draw the same rectangle shifted left in the right image
# This creates a synthetic disparity of about 10 pixels
cv2.rectangle(right, (90, 80), (170, 160), 255, -1)

# StereoBM
bm = cv2.StereoBM_create(
    numDisparities=64,
    blockSize=15
)
disp_bm = bm.compute(left, right).astype(np.float32) / 16.0

# StereoSGBM
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=5,
    P1=8 * 1 * 5 ** 2,
    P2=32 * 1 * 5 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disp_sgbm = sgbm.compute(left, right).astype(np.float32) / 16.0

print("StereoBM disparity shape:", disp_bm.shape)
print("StereoSGBM disparity shape:", disp_sgbm.shape)

bm_valid = disp_bm[disp_bm > 0]
sgbm_valid = disp_sgbm[disp_sgbm > 0]

if len(bm_valid) > 0:
    print("StereoBM valid disparity mean:", bm_valid.mean())
else:
    print("StereoBM produced no valid positive disparity.")

if len(sgbm_valid) > 0:
    print("StereoSGBM valid disparity mean:", sgbm_valid.mean())
else:
    print("StereoSGBM produced no valid positive disparity.")

print("OpenCV stereo test completed successfully.")


