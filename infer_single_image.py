"""
Single Image Inference Script for S2C Model
Usage: python infer_single_image.py --image_path <path_to_image> --checkpoint <path_to_checkpoint> --output_dir <output_directory>
"""

import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import model and utilities
# Note: We only import the network, not the full model class to avoid torch_scatter dependency
from networks import resnet38d
from tools.imutils import denorm, cam_on_image
from torchvision import transforms

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--image_path", required=True, type=str,
                        help="Path to input image")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to model checkpoint (e.g., ./experiments/exp_name/ckpt/015net_main.pth)")
    parser.add_argument("--output_dir", default="./inference_output", type=str,
                        help="Directory to save output CAMs")

    # Model parameters (must match training configuration)
    parser.add_argument("--C", default=20, type=int, help="Number of classes")
    parser.add_argument("--D", default=256, type=int, help="Feature dimension")
    parser.add_argument("--th_multi", default=0.5, type=float)
    parser.add_argument("--W", default=[1.0, 1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--T", default=1, type=float)

    # Class labels (provide as comma-separated indices, e.g., "7,12" for cat and dog)
    parser.add_argument("--class_indices", default="", type=str,
                        help="Comma-separated class indices present in the image (e.g., '7,12' for cat,dog). If empty, will generate CAMs for all classes.")

    # Multi-scale inference
    parser.add_argument("--scales", default=[0.5, 1.0, 1.5, 2.0], nargs='+', type=float,
                        help="Scales for multi-scale inference")

    # GPU
    parser.add_argument("--gpu", default=0, type=int, help="GPU device ID")
    parser.add_argument("--debug", action='store_true')

    # Dummy arguments for model compatibility
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--sstart", default=2, type=int)

    return parser.parse_args()


# Simple model wrapper for inference only (no torch_scatter needed)
class InferenceModel:
    """Lightweight model wrapper for inference without training dependencies"""
    def __init__(self, C=20, D=256):
        self.net_main = resnet38d.Net_CAM(C=C, D=D)
        self.C = C
        self.D = D

    def eval(self):
        """Set model to evaluation mode"""
        self.net_main.eval()
        return self


def load_image(image_path, scales=[0.5, 1.0, 1.5, 2.0]):
    """
    Load and preprocess image for inference
    Returns: list of tensors at different scales + flipped versions (8 total)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    # Normalization (ImageNet stats)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_list = []

    for scale in scales:
        # Calculate new size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))

        # Resize
        img_scaled = img.resize(new_size, Image.BICUBIC)

        # To tensor and normalize
        img_tensor = transforms.ToTensor()(img_scaled)
        img_tensor = normalize(img_tensor)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        img_list.append(img_tensor)

        # Also add horizontally flipped version
        img_flipped = transforms.functional.hflip(img_scaled)
        img_tensor_flip = transforms.ToTensor()(img_flipped)
        img_tensor_flip = normalize(img_tensor_flip)
        img_tensor_flip = img_tensor_flip.unsqueeze(0)

        img_list.append(img_tensor_flip)

    return img_list, original_size


def infer_single_image(model, img_list, label, original_size):
    """
    Perform multi-scale inference on a single image

    Args:
        model: loaded model
        img_list: list of 8 image tensors (4 scales × 2 for flip)
        label: class label tensor (1, C)
        original_size: (W, H) original image size

    Returns:
        cam_dict: dictionary of CAMs for each class
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = original_size[1], original_size[0]
    C = label.shape[1]

    model.eval()

    with torch.no_grad():
        cam_list = []

        for i, img in enumerate(img_list):
            img = img.to(device)

            # Forward pass
            out = model.net_main(img)
            cam = out['cam']

            # Interpolate to original size
            cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)[0]
            cam = F.relu(cam)

            # Multiply by label
            cam = cam.cpu().numpy()
            cam *= label.cpu().view(C, 1, 1).numpy()

            # Flip back if it was flipped
            if i % 2 == 1:
                cam = np.flip(cam, axis=-1)

            cam_list.append(cam)

        # Aggregate multi-scale CAMs
        cam = np.sum(cam_list, axis=0)
        cam_max = np.max(cam, (1, 2), keepdims=True)
        norm_cam = cam / (cam_max + 1e-5)

        # Create CAM dictionary
        cam_dict = {}
        for i in range(C):
            if label[0, i] > 1e-5:
                cam_dict[i] = norm_cam[i]

    return cam_dict


def visualize_and_save(image_path, cam_dict, output_dir, categories):
    """Save CAM visualizations"""
    # Get image name without extension
    img_name = osp.splitext(osp.basename(image_path))[0]

    # Create output directory for this image
    image_output_dir = osp.join(output_dir, img_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Load original image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).transpose(2, 0, 1) / 255.0  # (C, H, W)

    print(f"\nSaving CAM visualizations to: {image_output_dir}")

    for cls_idx, cam in cam_dict.items():
        # Overlay CAM on image
        cam_img = cam_on_image(img_np, cam)

        # Save
        out_path = osp.join(image_output_dir, f"cam_{categories[cls_idx]}.png")
        plt.imsave(out_path, np.transpose(cam_img, (1, 2, 0)))
        print(f"  Saved: {out_path}")

    # Save raw CAM dictionary
    cam_dict_path = osp.join(image_output_dir, "cam_dict.npy")
    np.save(cam_dict_path, cam_dict)
    print(f"  Saved CAM dictionary: {cam_dict_path}")


def main():
    args = get_arguments()

    # VOC categories
    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    print("="*80)
    print("S2C Single Image Inference")
    print("="*80)
    print(f"Image: {args.image_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")

    # Check if checkpoint exists
    if not osp.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found at {args.checkpoint}")
        print("\nMake sure you copied the trained model checkpoint to this device.")
        print("Expected format: ./experiments/[exp_name]/ckpt/[epoch]net_main.pth")
        return

    # Check if image exists
    if not osp.exists(args.image_path):
        print(f"\nERROR: Image not found at {args.image_path}")
        return

    # Parse class indices
    if args.class_indices:
        class_indices = [int(x) for x in args.class_indices.split(',')]
        print(f"\nClasses to detect: {[categories[i] for i in class_indices]}")
    else:
        print("\nNo class indices specified. Will generate CAMs for all classes with high activation.")
        class_indices = list(range(20))  # All classes

    # Create label tensor
    label = torch.zeros(1, args.C)
    for idx in class_indices:
        label[0, idx] = 1

    print("\n" + "-"*80)
    print("Step 1: Loading model...")
    print("-"*80)

    # Initialize lightweight inference model (no torch_scatter needed)
    model = InferenceModel(C=args.C, D=args.D)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu')

    # Handle DataParallel 'module.' prefix if present
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v

    model.net_main.load_state_dict(new_state_dict)

    # Move to GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model.net_main = model.net_main.to(device)
    model.net_main.eval()

    print(f"Model loaded successfully on {device}")

    print("\n" + "-"*80)
    print("Step 2: Loading and preprocessing image...")
    print("-"*80)

    # Load image
    img_list, original_size = load_image(args.image_path, scales=args.scales)
    print(f"Image size: {original_size[0]}x{original_size[1]}")
    print(f"Using {len(img_list)} scales for multi-scale inference: {args.scales}")

    print("\n" + "-"*80)
    print("Step 3: Running inference...")
    print("-"*80)

    # Run inference
    cam_dict = infer_single_image(model, img_list, label, original_size)

    print(f"Generated CAMs for {len(cam_dict)} classes")

    print("\n" + "-"*80)
    print("Step 4: Saving results...")
    print("-"*80)

    # Visualize and save
    visualize_and_save(args.image_path, cam_dict, args.output_dir, categories)

    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
