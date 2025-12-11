"""
Generate SAM masks from CAM outputs
Uses the CAM (Class Activation Map) from your trained model to prompt SAM for precise segmentation

Usage:
    python cam_to_sam_mask.py --image ./test-input/cat1.jpg --cam_dict ./results/cat1_cam_dict.npy --output_dir ./sam_masks
"""

import os
import os.path as osp
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("WARNING: segment-anything not installed. Install with:")
    print("  pip install git+https://github.com/facebookresearch/segment-anything.git")


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, type=str,
                        help="Path to input image")
    parser.add_argument("--cam_dict", required=True, type=str,
                        help="Path to CAM dictionary (.npy file)")
    parser.add_argument("--output_dir", default="./sam_masks", type=str,
                        help="Output directory for SAM masks")

    # SAM model
    parser.add_argument("--sam_checkpoint", default="./pretrained/sam_vit_b_01ec64.pth", type=str,
                        help="Path to SAM checkpoint")
    parser.add_argument("--sam_model", default="vit_b", type=str,
                        choices=["vit_b", "vit_l", "vit_h"],
                        help="SAM model type")

    # CAM processing
    parser.add_argument("--cam_threshold", default=0.5, type=float,
                        help="Threshold for CAM to select point prompts")
    parser.add_argument("--num_points", default=5, type=int,
                        help="Maximum number of point prompts per class")
    parser.add_argument("--min_distance", default=20, type=int,
                        help="Minimum distance between point prompts")

    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        type=str, help="Device to run SAM on")

    return parser.parse_args()


def extract_points_from_cam(cam, threshold=0.5, num_points=5, min_distance=20):
    """
    Extract point prompts from CAM

    Args:
        cam: 2D numpy array (H, W) with activation values
        threshold: Threshold for considering a point
        num_points: Maximum number of points to extract
        min_distance: Minimum distance between points

    Returns:
        points: (N, 2) array of (x, y) coordinates
    """
    # Find local maxima
    cam_filtered = ndi.maximum_filter(cam, size=3, mode='constant')
    peaks = peak_local_max(cam_filtered, min_distance=min_distance, threshold_abs=threshold)

    # Sort by activation value (highest first)
    if len(peaks) > 0:
        peak_values = cam[peaks[:, 0], peaks[:, 1]]
        sorted_indices = np.argsort(peak_values)[::-1]
        peaks = peaks[sorted_indices]

        # Limit number of points
        peaks = peaks[:num_points]

        # Convert to (x, y) format (SAM expects x, y, not row, col)
        points = np.flip(peaks, axis=1)  # (row, col) -> (x, y)

        return points
    else:
        return np.array([])


def visualize_results(image, cam, mask, points, class_name, output_path):
    """Create visualization of CAM, points, and SAM mask"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # CAM heatmap
    axes[1].imshow(image)
    axes[1].imshow(cam, alpha=0.5, cmap='jet')
    axes[1].set_title(f'CAM ({class_name})')
    axes[1].axis('off')

    # Points on image
    axes[2].imshow(image)
    if len(points) > 0:
        axes[2].scatter(points[:, 0], points[:, 1], c='red', s=100, marker='*', edgecolors='white', linewidths=2)
    axes[2].set_title(f'Prompts ({len(points)} points)')
    axes[2].axis('off')

    # SAM mask
    axes[3].imshow(image)
    if mask is not None:
        axes[3].imshow(mask, alpha=0.5, cmap='jet')
    axes[3].set_title('SAM Segmentation')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = get_arguments()

    if not HAS_SAM:
        print("\nERROR: segment-anything is not installed!")
        print("Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return

    # VOC class names
    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    print("="*80)
    print("CAM to SAM Mask Generation")
    print("="*80)
    print(f"Image: {args.image}")
    print(f"CAM dict: {args.cam_dict}")
    print(f"Output: {args.output_dir}")
    print(f"SAM model: {args.sam_model}")
    print(f"Device: {args.device}")
    print()

    # Create output directory for this image
    image_name = osp.splitext(osp.basename(args.image))[0]
    image_output_dir = osp.join(args.output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    print(f"Output directory: {image_output_dir}")

    # Load image
    print("\nLoading image...")
    image = Image.open(args.image).convert('RGB')
    image_np = np.array(image)
    H, W = image_np.shape[:2]
    print(f"  Image size: {W}x{H}")

    # Load CAM dictionary
    print("\nLoading CAM dictionary...")
    cam_dict = np.load(args.cam_dict, allow_pickle=True).item()
    print(f"  Found {len(cam_dict)} classes: {[categories[k] for k in cam_dict.keys()]}")

    # Load SAM model
    print(f"\nLoading SAM model ({args.sam_model})...")
    if not osp.exists(args.sam_checkpoint):
        print(f"\nERROR: SAM checkpoint not found at {args.sam_checkpoint}")
        print("Download it from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        return

    sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # Set image for SAM
    predictor.set_image(image_np)
    print("  SAM model loaded and image encoded")

    # Process each class
    print("\n" + "="*80)
    print("Generating SAM masks...")
    print("="*80)

    all_masks = {}

    for class_idx, cam in cam_dict.items():
        class_name = categories[class_idx]
        print(f"\nProcessing class {class_idx} ({class_name})...")

        # Resize CAM to image size if needed
        if cam.shape != (H, W):
            cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            cam_resized = cam

        # Extract point prompts from CAM
        points = extract_points_from_cam(
            cam_resized,
            threshold=args.cam_threshold,
            num_points=args.num_points,
            min_distance=args.min_distance
        )

        print(f"  Extracted {len(points)} point prompts from CAM")

        if len(points) == 0:
            print(f"  WARNING: No points found above threshold {args.cam_threshold}")
            print(f"  Try lowering --cam_threshold")
            continue

        # Use SAM to generate mask
        print(f"  Running SAM with {len(points)} points...")

        # Prepare prompts for SAM
        point_coords = points
        point_labels = np.ones(len(points))  # All positive prompts

        # Generate mask
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True  # Generate 3 masks
        )

        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]

        print(f"  Generated mask with score: {best_score:.3f}")

        # Save results
        all_masks[class_idx] = best_mask

        # Save mask as image
        mask_path = osp.join(image_output_dir, f"mask_{class_name}.png")
        Image.fromarray((best_mask * 255).astype(np.uint8)).save(mask_path)
        print(f"  Saved mask: {mask_path}")

        # Save visualization
        vis_path = osp.join(image_output_dir, f"vis_{class_name}.png")
        visualize_results(image_np, cam_resized, best_mask, points, class_name, vis_path)
        print(f"  Saved visualization: {vis_path}")

    # Save combined masks
    print("\n" + "-"*80)
    print("Saving combined results...")

    # Create semantic segmentation map
    seg_map = np.zeros((H, W), dtype=np.uint8)
    for class_idx, mask in all_masks.items():
        seg_map[mask] = class_idx + 1  # +1 because 0 is background

    seg_path = osp.join(image_output_dir, "segmentation.png")
    Image.fromarray(seg_map).save(seg_path)
    print(f"  Saved segmentation map: {seg_path}")

    # Save as colored visualization
    # Create color map (VOC palette)
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # background
    for i in range(1, 21):
        colormap[i] = [
            (i * 7) % 256,
            (i * 13) % 256,
            (i * 19) % 256
        ]

    seg_colored = colormap[seg_map]
    seg_colored_path = osp.join(image_output_dir, "segmentation_color.png")
    Image.fromarray(seg_colored).save(seg_colored_path)
    print(f"  Saved colored segmentation: {seg_colored_path}")

    # Create overlay
    overlay = cv2.addWeighted(image_np, 0.6, seg_colored, 0.4, 0)
    overlay_path = osp.join(image_output_dir, "overlay.png")
    Image.fromarray(overlay).save(overlay_path)
    print(f"  Saved overlay: {overlay_path}")

    print("\n" + "="*80)
    print("✓ Done!")
    print("="*80)
    print(f"Generated SAM masks for {len(all_masks)} classes")
    print(f"Results saved to: {image_output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
