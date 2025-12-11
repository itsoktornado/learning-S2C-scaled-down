# Running Inference Without torch-scatter

## Problem

You may encounter errors when installing `torch-scatter`:
```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'torch-scatter' when getting requirements to build wheel
```

## Solution

**Good news**: `torch-scatter` is **NOT needed for inference**! It's only used during training for the SSC (SAM-Segment Contrasting) loss.

The inference scripts have been updated to work **without** `torch-scatter`.

---

## Updated Installation (No torch-scatter Required)

### Minimal Installation for Inference

```bash
# Install core dependencies (no torch-scatter!)
pip install torch torchvision
pip install opencv-python pillow matplotlib
pip install numpy scipy scikit-image
pip install tqdm

# Segment Anything is technically optional too, but install if you can
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or use the requirements file:
```bash
pip install -r requirements_inference.txt
```

---

## What Changed?

The inference scripts now use a lightweight `InferenceModel` class that:
- ✅ Only loads the ResNet38 CAM network
- ✅ Doesn't import the full training model
- ✅ Avoids all `torch-scatter` dependencies
- ✅ Works exactly the same for generating CAMs

### Before (required torch-scatter):
```python
from models.model_s2c import model_WSSS  # Imports torch_scatter
model = model_WSSS(args)
```

### After (no torch-scatter needed):
```python
from networks import resnet38d  # Direct network import
model = InferenceModel(C=20, D=256)
```

---

## Verify Your Setup

Run the verification script:
```bash
python check_inference_setup.py
```

You should see:
```
2. Checking Python packages...
  ✓ torch
  ✓ torchvision
  ✓ numpy
  ✓ PIL
  ✓ cv2
  ✓ matplotlib
  ✓ scipy
  ✓ skimage
   Optional packages (for training only):
  ✗ segment_anything - NOT INSTALLED
  ✗ torch_scatter - NOT INSTALLED
```

The optional packages being missing is **OK** for inference!

---

## Run Inference

Everything works normally:

```bash
# Single image
python infer_single_image.py \
    --image_path /path/to/image.jpg \
    --checkpoint ./experiments/myexp/ckpt/015net_main.pth \
    --output_dir ./results

# Batch inference
python infer_batch.py \
    --image_dir /path/to/images/ \
    --checkpoint ./experiments/myexp/ckpt/015net_main.pth \
    --output_dir ./results_batch
```

---

## Why This Works

Looking at the training code in [model_s2c.py:334](d:\Projects\S2C\models\model_s2c.py#L334):

```python
# torch_scatter is ONLY used here during training:
pt = torch_scatter.scatter_mean(feat_main_.detach(), index_)  # SSC loss
```

During **inference**, the model:
1. Only runs forward pass to get CAMs
2. Doesn't compute SSC loss
3. Doesn't need scatter operations
4. Doesn't need torch_scatter at all!

---

## If You Still Want to Install torch-scatter (Optional)

If you plan to train the model or want the full environment:

### Method 1: Let PyTorch Build It
```bash
# Make sure PyTorch is installed first
pip install torch torchvision

# Then try torch-scatter
pip install torch-scatter
```

### Method 2: Use Pre-built Wheels
```bash
# Check your PyTorch version
python -c "import torch; print(torch.__version__)"

# Install matching version (example for PyTorch 2.1.0 + CPU)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# For CUDA 11.8:
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### Method 3: Install from Conda (Easier)
```bash
conda install pytorch-scatter -c pyg
```

---

## Summary

| Package | Training | Inference |
|---------|----------|-----------|
| torch | ✅ Required | ✅ Required |
| torchvision | ✅ Required | ✅ Required |
| **torch-scatter** | ✅ Required | ❌ **Not needed!** |
| segment-anything | ✅ Required | ⚠️ Imported but not used |
| opencv-python | ✅ Required | ✅ Required |
| numpy, scipy, etc. | ✅ Required | ✅ Required |

**For inference only, you can skip torch-scatter entirely!**

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch_scatter'"

If you see this error when running inference, make sure you're using the **updated** inference scripts:
- [infer_single_image.py](infer_single_image.py) - Updated to avoid torch_scatter
- [infer_batch.py](infer_batch.py) - Updated to avoid torch_scatter

### "ImportError: cannot import name 'model_WSSS'"

This is expected! The updated scripts don't import `model_WSSS` anymore. They use `InferenceModel` instead.

### Still Having Issues?

1. Make sure you have the latest version of the inference scripts
2. Check that PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
3. Verify the checkpoint file exists and is accessible
4. Run `python check_inference_setup.py` to diagnose issues

---

## Questions?

- For inference setup: See [QUICK_START_INFERENCE.md](QUICK_START_INFERENCE.md)
- For detailed guide: See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)
- For training (requires torch-scatter): See [README.md](README.md)
