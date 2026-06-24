from utils import (
    list_kitti_image_ids,
    load_kitti_sample,
    get_output_dirs,
    save_comparison_figure,
    save_disparity_figure,
)


def main():
    image_ids = list_kitti_image_ids(num_images=1)
    image_id = image_ids[0]

    left, right, gt_disp = load_kitti_sample(image_id)

    output_dirs = get_output_dirs()
    save_path = output_dirs["figures"] / f"{image_id}_kitti_check.png"

    # For check_kitti, we only visualise GT disparity.
    save_disparity_figure(
        gt_disp,
        save_path,
        title=f"KITTI Ground Truth Disparity: {image_id}",
        vmax=128
    )

    print("KITTI sample loaded successfully.")
    print(f"Image ID: {image_id}")
    print(f"Left image shape: {left.shape}")
    print(f"Right image shape: {right.shape}")
    print(f"GT disparity shape: {gt_disp.shape}")
    print(f"Valid GT pixels: {(gt_disp > 0).sum()}")
    print(f"Saved GT disparity visualisation to: {save_path}")
    print("check_kitti.py completed successfully.")


if __name__ == "__main__":
    main()