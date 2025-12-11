"""
Quick verification script to check if everything is ready for inference
Run this on the new device before attempting inference
"""

import os
import os.path as osp
import sys

def check_file(path, name, required=True):
    """Check if a file exists"""
    exists = osp.exists(path)
    status = "✓" if exists else ("✗ MISSING" if required else "○")

    if exists:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {status} {name}: {path} ({size_mb:.1f} MB)")
    else:
        print(f"  {status} {name}: {path}")
        if required:
            print(f"      WARNING: This file is required!")

    return exists

def check_import(module_name):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
        return True
    except ImportError:
        print(f"  ✗ {module_name} - NOT INSTALLED")
        return False

def main():
    print("="*80)
    print("S2C Inference Setup Checker")
    print("="*80)
    print()

    all_good = True

    # Check Python version
    print("1. Checking Python version...")
    py_version = sys.version_info
    print(f"   Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("   ✗ WARNING: Python 3.8+ is recommended")
        all_good = False
    else:
        print("   ✓ Python version OK")
    print()

    # Check required Python packages
    print("2. Checking Python packages...")
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'PIL',
        'cv2',
        'matplotlib',
        'scipy',
        'skimage',
    ]

    optional_packages = [
        'segment_anything',
        'torch_scatter',
    ]

    for package in required_packages:
        if not check_import(package):
            all_good = False

    print("   Optional packages (for training only):")
    for package in optional_packages:
        check_import(package)
    print()

    # Check source code
    print("3. Checking source code files...")
    source_files = [
        './models/model_s2c.py',
        './networks/resnet38d.py',
        './tools/imutils.py',
        './tools/utils.py',
        './infer_single_image.py',
    ]

    for file in source_files:
        if not check_file(file, osp.basename(file), required=True):
            all_good = False
    print()

    # Check pretrained weights
    print("4. Checking pretrained weights...")
    pretrained_files = [
        ('./pretrained/resnet_38d.params', 'ResNet38 ImageNet weights'),
        ('./pretrained/sam_vit_b_01ec64.pth', 'SAM ViT-B weights'),
    ]

    for file, name in pretrained_files:
        if not check_file(file, name, required=True):
            all_good = False
            print(f"      Download from: https://github.com/[your-repo]/README.md")
    print()

    # Check for trained checkpoints
    print("5. Checking for trained model checkpoints...")
    exp_dir = './experiments'

    if not osp.exists(exp_dir):
        print(f"  ✗ Experiments directory not found: {exp_dir}")
        print(f"      Create it and place your trained checkpoint inside:")
        print(f"      ./experiments/[your_exp_name]/ckpt/[epoch]net_main.pth")
        all_good = False
    else:
        # Search for checkpoint files
        found_checkpoints = []
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                if file.endswith('net_main.pth'):
                    checkpoint_path = osp.join(root, file)
                    found_checkpoints.append(checkpoint_path)

        if found_checkpoints:
            print(f"  ✓ Found {len(found_checkpoints)} checkpoint(s):")
            for ckpt in found_checkpoints:
                size_mb = os.path.getsize(ckpt) / (1024 * 1024)
                print(f"    - {ckpt} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ No checkpoint files found in {exp_dir}")
            print(f"      Copy your trained model to:")
            print(f"      ./experiments/[your_exp_name]/ckpt/[epoch]net_main.pth")
            all_good = False
    print()

    # GPU Check
    print("6. Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ○ No GPU found - will use CPU (slower)")
    except:
        print(f"  ✗ Cannot check GPU status")
    print()

    # Final summary
    print("="*80)
    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("You're ready to run inference. Example command:")
        print()
        if found_checkpoints:
            example_ckpt = found_checkpoints[0]
            print(f"python infer_single_image.py \\")
            print(f"    --image_path /path/to/your/image.jpg \\")
            print(f"    --checkpoint {example_ckpt} \\")
            print(f"    --output_dir ./inference_output")
        else:
            print(f"python infer_single_image.py \\")
            print(f"    --image_path /path/to/your/image.jpg \\")
            print(f"    --checkpoint ./experiments/[exp_name]/ckpt/015net_main.pth \\")
            print(f"    --output_dir ./inference_output")
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Please fix the issues above before running inference.")
        print("See INFERENCE_GUIDE.md for detailed setup instructions.")
    print("="*80)

if __name__ == '__main__':
    main()
