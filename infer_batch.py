"""
Batch Inference Script for S2C Model
Run inference on multiple images in a folder

Usage: python infer_batch.py --image_dir <dir> --checkpoint <path> --output_dir <dir>
"""

import os
import os.path as osp
import argparse
import glob
from tqdm import tqdm

# Import single image inference function
from infer_single_image import (
    load_image, infer_single_image, visualize_and_save,
    InferenceModel
)
import torch

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--image_dir", required=True, type=str,
                        help="Directory containing images")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="./inference_output_batch", type=str,
                        help="Directory to save outputs")

    # Image extensions to process
    parser.add_argument("--extensions", default=["jpg", "jpeg", "png", "bmp"],
                        nargs='+', type=str,
                        help="Image file extensions to process")

    # Model parameters
    parser.add_argument("--C", default=20, type=int)
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--th_multi", default=0.5, type=float)
    parser.add_argument("--W", default=[1.0, 1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--T", default=1, type=float)

    # Class labels
    parser.add_argument("--class_indices", default="", type=str,
                        help="Comma-separated class indices (leave empty for all)")

    # Multi-scale inference
    parser.add_argument("--scales", default=[0.5, 1.0, 1.5, 2.0], nargs='+', type=float)

    # GPU
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--debug", action='store_true')

    # Dummy arguments
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--sstart", default=2, type=int)

    return parser.parse_args()


def find_images(image_dir, extensions):
    """Find all images in directory with given extensions"""
    image_paths = []

    for ext in extensions:
        # Case insensitive search
        pattern1 = osp.join(image_dir, f"*.{ext}")
        pattern2 = osp.join(image_dir, f"*.{ext.upper()}")

        image_paths.extend(glob.glob(pattern1))
        image_paths.extend(glob.glob(pattern2))

    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))

    return image_paths


def main():
    args = get_arguments()

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    print("="*80)
    print("S2C Batch Inference")
    print("="*80)
    print(f"Image directory: {args.image_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")

    # Find images
    print("\nSearching for images...")
    image_paths = find_images(args.image_dir, args.extensions)

    if len(image_paths) == 0:
        print(f"\nERROR: No images found in {args.image_dir}")
        print(f"Looking for extensions: {args.extensions}")
        return

    print(f"Found {len(image_paths)} images")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse class indices
    if args.class_indices:
        class_indices = [int(x) for x in args.class_indices.split(',')]
        print(f"\nClasses to detect: {[categories[i] for i in class_indices]}")
    else:
        print("\nDetecting all classes")
        class_indices = list(range(20))

    # Create label tensor
    label = torch.zeros(1, args.C)
    for idx in class_indices:
        label[0, idx] = 1

    # Load model
    print("\n" + "-"*80)
    print("Loading model...")
    print("-"*80)

    model = InferenceModel(C=args.C, D=args.D)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location='cpu')

    # Handle DataParallel 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.net_main.load_state_dict(new_state_dict)

    # Move to GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.net_main = model.net_main.to(device)
    model.net_main.eval()

    print(f"Model loaded on {device}")

    # Process images
    print("\n" + "-"*80)
    print("Processing images...")
    print("-"*80)

    successful = 0
    failed = 0

    for img_path in tqdm(image_paths, desc="Inference"):
        try:
            # Load image
            img_list, original_size = load_image(img_path, scales=args.scales)

            # Run inference
            cam_dict = infer_single_image(model, img_list, label, original_size)

            # Save results
            img_name = osp.splitext(osp.basename(img_path))[0]
            img_output_dir = osp.join(args.output_dir, img_name)
            visualize_and_save(img_path, cam_dict, img_output_dir, categories)

            successful += 1

        except Exception as e:
            print(f"\nERROR processing {img_path}: {str(e)}")
            failed += 1
            continue

    # Summary
    print("\n" + "="*80)
    print("Batch Inference Complete!")
    print("="*80)
    print(f"Successfully processed: {successful}/{len(image_paths)} images")
    if failed > 0:
        print(f"Failed: {failed}/{len(image_paths)} images")
    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
