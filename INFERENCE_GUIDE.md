# Single Image Inference Guide for S2C

This guide explains how to export a trained S2C model from one device and use it for inference on another device without retraining.

---

## Overview

After training the S2C model on one device, you can transfer just the trained weights to another device and run inference on individual images without needing the full training dataset or retraining the model.

---

## Step 1: Export from Training Device

### Files to Copy

After training completes, you need to copy the following files from the **training device**:

#### 1. Trained Model Checkpoint (Required)
```
./experiments/[your_exp_name]/ckpt/[epoch]net_main.pth
```

**Example:** If your experiment name was `241211_s2c_myexp` and you want to use the model from epoch 15:
```
./experiments/241211_s2c_myexp/ckpt/015net_main.pth
```

The checkpoint file is typically 200-300MB in size.

#### 2. Pretrained Weights (Required on new device)
```
./pretrained/resnet_38d.params          (~170MB)
./pretrained/sam_vit_b_01ec64.pth       (~375MB)
```

**Note:** These are the ImageNet pretrained ResNet38 and SAM ViT-B weights. If the new device already has these in the `./pretrained/` folder, you don't need to copy them again.

#### 3. Inference Script (New)
```
./infer_single_image.py
```

This is the single-image inference script created in this repository.

### How to Copy

**Option A: Using USB/External Drive**
```bash
# On training device, create a transfer folder
mkdir -p transfer/ckpt
mkdir -p transfer/pretrained

# Copy trained model
cp ./experiments/[your_exp_name]/ckpt/[epoch]net_main.pth transfer/ckpt/

# Copy pretrained weights (if needed on new device)
cp ./pretrained/resnet_38d.params transfer/pretrained/
cp ./pretrained/sam_vit_b_01ec64.pth transfer/pretrained/

# Copy inference script
cp ./infer_single_image.py transfer/

# Copy to USB drive
cp -r transfer /path/to/usb/
```

**Option B: Using SCP (network transfer)**
```bash
# From training device to new device
scp -r ./experiments/[your_exp_name]/ckpt/[epoch]net_main.pth user@new-device:/path/to/S2C/experiments/[your_exp_name]/ckpt/
scp ./infer_single_image.py user@new-device:/path/to/S2C/
```

---

## Step 2: Setup on New Device

### Directory Structure

On the **new device**, set up the following structure:

```
S2C/
├── models/              # Source code (from repo)
├── networks/            # Source code (from repo)
├── tools/               # Source code (from repo)
├── voc12/               # Source code (from repo)
├── pretrained/          # Pretrained weights
│   ├── resnet_38d.params
│   └── sam_vit_b_01ec64.pth
├── experiments/
│   └── [your_exp_name]/
│       └── ckpt/
│           └── 015net_main.pth    # Your trained model
├── inference_output/    # Will be created automatically
└── infer_single_image.py
```

### Prerequisites

Make sure the new device has:

1. **Python 3.8+** with the required packages:
   ```bash
   pip install torch torchvision
   pip install torch-scatter
   pip install opencv-python pillow matplotlib numpy scipy scikit-image
   pip install segment-anything
   ```

   **Note:** If `torch-scatter` installation fails, install it with your specific CUDA version:
   ```bash
   # Check your versions first
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

   # Then install for your CUDA version (example for CUDA 11.8):
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```

2. **Source Code**: Clone or copy the S2C repository
   ```bash
   git clone [repository-url]
   cd S2C
   ```

3. **Pretrained Weights**: Place in `./pretrained/` folder

4. **Trained Checkpoint**: Place in `./experiments/[exp_name]/ckpt/`

---

## Step 3: Run Inference

### Basic Usage

```bash
python infer_single_image.py \
    --image_path /path/to/your/image.jpg \
    --checkpoint ./experiments/[your_exp_name]/ckpt/015net_main.pth \
    --output_dir ./inference_output
```

### With Class Labels (Recommended)

If you know which classes are in the image, specify them for better results:

```bash
python infer_single_image.py \
    --image_path ./test_images/cat.jpg \
    --checkpoint ./experiments/241211_s2c_myexp/ckpt/015net_main.pth \
    --class_indices "7,12" \
    --output_dir ./results
```

**Class indices for VOC 2012:**
```
0: aeroplane     10: diningtable
1: bicycle       11: dog
2: bird          12: horse
3: boat          13: motorbike
4: bottle        14: person
5: bus           15: pottedplant
6: car           16: sheep
7: cat           17: sofa
8: chair         18: train
9: cow           19: tvmonitor
```

### Advanced Options

```bash
python infer_single_image.py \
    --image_path ./image.jpg \
    --checkpoint ./experiments/myexp/ckpt/015net_main.pth \
    --output_dir ./results \
    --class_indices "7,14" \
    --scales 0.5 1.0 1.5 2.0 \
    --gpu 0
```

**Parameters:**
- `--image_path`: Path to input image (required)
- `--checkpoint`: Path to trained model checkpoint (required)
- `--output_dir`: Where to save results (default: `./inference_output`)
- `--class_indices`: Comma-separated class IDs (e.g., "7,12" for cat,dog)
- `--scales`: Multi-scale inference scales (default: 0.5, 1.0, 1.5, 2.0)
- `--gpu`: GPU device ID (default: 0)
- `--C`: Number of classes (default: 20 for VOC)
- `--D`: Feature dimension (default: 256)

---

## Step 4: Understanding the Output

### Output Files

After running inference, you'll find in the output directory:

1. **CAM Visualizations** (PNG images):
   ```
   image_name_cam_cat.png
   image_name_cam_person.png
   ...
   ```
   These show the Class Activation Maps overlaid on the original image.

2. **CAM Dictionary** (NumPy file):
   ```
   image_name_cam_dict.npy
   ```
   Raw CAM data that can be loaded for further processing:
   ```python
   import numpy as np
   cam_dict = np.load('image_name_cam_dict.npy', allow_pickle=True).item()
   ```

### Example Output

```
================================================================================
S2C Single Image Inference
================================================================================
Image: ./test_images/2007_000129.jpg
Checkpoint: ./experiments/241211_s2c_myexp/ckpt/015net_main.pth
Output directory: ./inference_output

Classes to detect: ['cat', 'person']

--------------------------------------------------------------------------------
Step 1: Loading model...
--------------------------------------------------------------------------------
Loading checkpoint from: ./experiments/241211_s2c_myexp/ckpt/015net_main.pth
Model loaded successfully on cuda:0

--------------------------------------------------------------------------------
Step 2: Loading and preprocessing image...
--------------------------------------------------------------------------------
Image size: 500x375
Using 8 scales for multi-scale inference: [0.5, 1.0, 1.5, 2.0]

--------------------------------------------------------------------------------
Step 3: Running inference...
--------------------------------------------------------------------------------
Generated CAMs for 2 classes

--------------------------------------------------------------------------------
Step 4: Saving results...
--------------------------------------------------------------------------------

Saving CAM visualizations to: ./inference_output
  Saved: ./inference_output/2007_000129_cam_cat.png
  Saved: ./inference_output/2007_000129_cam_person.png
  Saved CAM dictionary: ./inference_output/2007_000129_cam_dict.npy

================================================================================
Inference completed successfully!
================================================================================
```

---

## Troubleshooting

### Error: "Checkpoint not found"

**Problem:** The script can't find the model checkpoint file.

**Solution:**
```bash
# Check if the file exists
ls -lh ./experiments/[your_exp_name]/ckpt/

# Make sure the path matches exactly
# Epoch 15 checkpoint should be named: 015net_main.pth
```

### Error: "RuntimeError: Error(s) in loading state_dict"

**Problem:** Model architecture mismatch.

**Solution:** Make sure the `--C` and `--D` parameters match your training configuration:
```bash
python infer_single_image.py \
    --image_path ./image.jpg \
    --checkpoint ./ckpt/015net_main.pth \
    --C 20 \
    --D 256
```

### Error: "FileNotFoundError: resnet_38d.params"

**Problem:** Pretrained weights not found.

**Solution:**
```bash
# Download and place pretrained weights
mkdir -p pretrained
cd pretrained

# Download ResNet38
wget https://drive.google.com/file/d/1fpb4vah3e-Ynx4cv5upUcqnpJFY_FTja/view?usp=sharing

# Download SAM ViT-B
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Out of Memory Error

**Problem:** GPU runs out of memory during inference.

**Solution:** The inference script uses multi-scale testing which can be memory-intensive. Try:
```bash
# Use fewer scales
python infer_single_image.py \
    --image_path ./image.jpg \
    --checkpoint ./ckpt/015net_main.pth \
    --scales 1.0
```

Or run on CPU:
```bash
# This will automatically use CPU if no GPU is available
CUDA_VISIBLE_DEVICES="" python infer_single_image.py \
    --image_path ./image.jpg \
    --checkpoint ./ckpt/015net_main.pth
```

---

## Notes

1. **No Training Data Needed**: The inference script doesn't require the VOC2012 dataset or any training data.

2. **Model-Only Transfer**: You only need the `.pth` checkpoint file, not the entire experiment directory.

3. **Multi-Scale Inference**: The script uses 4 scales (0.5x, 1.0x, 1.5x, 2.0x) by default, same as training. This takes more time but produces better results.

4. **Class Labels**: While you can run without specifying `--class_indices`, results are better when you specify which classes to look for.

5. **Checkpoint Naming**: The checkpoint files follow the naming pattern `[epoch]net_main.pth` where epoch is zero-padded to 3 digits (e.g., `015net_main.pth` for epoch 15).

---

## Quick Reference

### Minimum Required Files for New Device

```
✓ Source code (models/, networks/, tools/, voc12/)
✓ infer_single_image.py
✓ ./pretrained/resnet_38d.params
✓ ./pretrained/sam_vit_b_01ec64.pth
✓ ./experiments/[exp_name]/ckpt/[epoch]net_main.pth
```

### One-Line Inference Command

```bash
python infer_single_image.py --image_path IMAGE.jpg --checkpoint experiments/EXPNAME/ckpt/015net_main.pth --class_indices "7,14"
```

---

## Additional Information

For more details about the S2C model and training, see:
- [README.md](README.md) - Main project documentation
- [train.py](train.py) - Training script
- [evaluation.py](evaluation.py) - Evaluation on VOC dataset

For issues or questions about inference, check the inference script source code at [infer_single_image.py](infer_single_image.py).
