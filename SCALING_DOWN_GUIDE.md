# S2C Scaling Down Guide for Single RTX 4090 GPU

This guide provides comprehensive strategies to adapt the S2C (SAM to CAMs) project for training on a single RTX 4090 GPU (24GB VRAM) instead of the original 4-GPU setup.

## Table of Contents
- [Hardware Bottleneck Analysis](#hardware-bottleneck-analysis)
- [Tier 1: Critical Reductions](#tier-1-critical-reductions-implement-first)
- [Tier 2: Moderate Reductions](#tier-2-moderate-reductions)
- [Tier 3: Dataset & Training Optimizations](#tier-3-dataset--training-optimizations)
- [Tier 4: Evaluation Optimizations](#tier-4-evaluation-optimizations)
- [Recommended Configuration](#recommended-configuration-for-rtx-4090-24gb)
- [Summary Table](#summary-table)

---

## Hardware Bottleneck Analysis

The original setup requires **4 GPUs** with the following computational bottlenecks:

1. **SAM ViT-H Model** - Massive Vision Transformer (632M parameters)
2. **ResNet38d Backbone** - Deep feature extractor
3. **Multi-scale inference** - 4 scales (0.5x, 1.0x, 1.5x, 2.0x) during CPM phase
4. **Batch size 8** - High memory footprint
5. **Multi-scale + flip evaluation** - 8 forward passes per image
6. **Dataset size** - 10,582 training images (train_aug.txt)

---

## Tier 1: Critical Reductions (Implement First)

### Option 1A: Switch to Lighter SAM Model

**Location**: `models/model_s2c.py:98-99`

**Impact**: ~3-4x memory reduction for SAM alone

**Current Code**:
```python
# Line 98-99
sam_path = './pretrained/sam_vit_h.pth'
self.net_sam = sam_model_registry['vit_h'](checkpoint=sam_path)
```

**Modified Code**:
```python
# Option 1: ViT-B (Recommended for maximum memory savings)
sam_path = './pretrained/sam_vit_b.pth'
self.net_sam = sam_model_registry['vit_b'](checkpoint=sam_path)

# Option 2: ViT-L (Better quality, moderate savings)
sam_path = './pretrained/sam_vit_l.pth'
self.net_sam = sam_model_registry['vit_l'](checkpoint=sam_path)
```

**SAM Model Comparison**:
- **ViT-H (current)**: 632M params, ~2.5GB VRAM per image
- **ViT-L**: 308M params, ~1.2GB VRAM per image (50% reduction)
- **ViT-B**: 89M params, ~400MB VRAM per image (84% reduction)

**Download Links**:
- ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

**Trade-off**: Slightly less precise segmentation masks, but should still work well for WSSS

---

### Option 1B: Reduce Batch Size

**Location**: `train.py:38`

**Impact**: Linear memory reduction (50-75% reduction if going from 8 to 2-4)

**Current Code**:
```python
# Line 38
parser.add_argument("--batch_size", default=8, type=int)
```

**Modified Code**:
```python
# Conservative option
parser.add_argument("--batch_size", default=4, type=int)

# Maximum memory savings
parser.add_argument("--batch_size", default=2, type=int)
```

**Command Line Usage**:
```bash
# Override default without code changes
python train.py --name exp_scaled --model s2c --batch_size 2
```

**Trade-off**:
- 2-4x longer training time
- Potentially less stable gradients (smaller batch statistics)
- May need to adjust learning rate accordingly

---

### Option 1C: Reduce Input Resolution

**Location**: `train.py:43-44`

**Impact**: ~50% memory reduction with moderate quality impact

**Current Code**:
```python
# Line 43-44
parser.add_argument("--resize", default=[256, 512], nargs='+', type=float)
parser.add_argument("--crop", default=256, type=int)
```

**Modified Code**:
```python
# Option 1: Moderate reduction
parser.add_argument("--resize", default=[224, 448], nargs='+', type=float)
parser.add_argument("--crop", default=224, type=int)

# Option 2: Maximum reduction
parser.add_argument("--resize", default=[192, 384], nargs='+', type=float)
parser.add_argument("--crop", default=192, type=int)
```

**Command Line Usage**:
```bash
python train.py --name exp --model s2c --resize 224 448 --crop 224
```

**Trade-off**: Smaller spatial context, may reduce CAM quality by 1-2% mIoU

---

## Tier 2: Moderate Reductions

### Option 2A: Reduce Multi-Scale Inference Scales

**Location**: `models/model_s2c.py:200-205`

**Impact**: 25-50% memory & computation reduction during CPM phase

**Current Code**:
```python
# Line 200-205 (4 scales)
img_05 = F.interpolate(self.img, scale_factor=0.5, mode='bilinear', align_corners=True)
img_10 = self.img
img_15 = F.interpolate(self.img, scale_factor=1.5, mode='bilinear', align_corners=True)
img_20 = F.interpolate(self.img, scale_factor=2.0, mode='bilinear', align_corners=True)

img_ms = [img_05, img_10, img_15, img_20]
```

**Modified Code - Option 1 (3 scales)**:
```python
# Remove smallest scale
img_10 = self.img
img_15 = F.interpolate(self.img, scale_factor=1.5, mode='bilinear', align_corners=True)
img_20 = F.interpolate(self.img, scale_factor=2.0, mode='bilinear', align_corners=True)

img_ms = [img_10, img_15, img_20]
```

**Modified Code - Option 2 (2 scales)**:
```python
# Keep base and one upscale
img_10 = self.img
img_15 = F.interpolate(self.img, scale_factor=1.5, mode='bilinear', align_corners=True)

img_ms = [img_10, img_15]
```

**Trade-off**: Less robust multi-scale CAMs, but still effective (minimal impact if using 3 scales)

---

### Option 2B: Reduce SAM Image Resolution

**Location**: `models/model_s2c.py:77` and `models/model_s2c.py:223`

**Impact**: ~40% memory reduction for SAM processing

**Current Code**:
```python
# Line 77
self.size_sam = 1024

# Line 223 - SAM input is resized to this dimension
img_sam = F.interpolate(denorm(self.img)*255, (self.size_sam, self.size_sam), mode='bilinear', align_corners=True)
```

**Modified Code**:
```python
# Option 1: Moderate reduction
self.size_sam = 768

# Option 2: Maximum reduction
self.size_sam = 512
```

**Trade-off**: SAM operates on lower resolution, may miss fine details but generally acceptable

---

### Option 2C: Reduce Feature Dimension

**Location**: `train.py:49`

**Impact**: ~20-30% memory reduction

**Current Code**:
```python
# Line 49
parser.add_argument("--D", default=256, type=int)
```

**Modified Code**:
```python
# Option 1: Moderate reduction
parser.add_argument("--D", default=192, type=int)

# Option 2: Larger reduction
parser.add_argument("--D", default=128, type=int)
```

**Command Line Usage**:
```bash
python train.py --name exp --model s2c --D 192
```

**Trade-off**: Lower-dimensional features may reduce representation capacity slightly

---

### Option 2D: Use Gradient Checkpointing

**Location**: Add to `models/model_s2c.py`

**Impact**: 30-40% memory reduction with 10-20% training slowdown

**Implementation**:

Add import at the top of `models/model_s2c.py`:
```python
import torch.utils.checkpoint as checkpoint
```

Modify the forward pass in the `update()` method around line 309:
```python
# Current (line 309)
out_main = self.net_main(self.img)

# Replace with:
out_main = checkpoint.checkpoint(self.net_main, self.img)
```

For SAM encoder (around line 228):
```python
# Current (line 228-230)
features_sam = self.net_sam(run_encoder_only=True,
                            transformed_image=img_sam,
                            original_image_size=(H,W))

# Replace with:
features_sam = checkpoint.checkpoint(
    self.net_sam,
    run_encoder_only=True,
    transformed_image=img_sam,
    original_image_size=(H,W)
)
```

**Trade-off**: Trades compute for memory (recomputes activations during backward pass)

---

## Tier 3: Dataset & Training Optimizations

### Option 3A: Use Smaller Training Set

**Location**: `train.py:35`

**Impact**: Faster epochs, less diverse training data

**Current Code**:
```python
# Line 35 (10,582 images)
parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
```

**Modified Code**:
```python
# Use base training set (1,464 images)
parser.add_argument("--train_list", default="voc12/train.txt", type=str)
```

**Command Line Usage**:
```bash
python train.py --name exp --model s2c --train_list voc12/train.txt
```

**Trade-off**: May reduce final performance by 2-4% mIoU, much faster experimentation

---

### Option 3B: Reduce Number of Workers

**Location**: `train.py:37` and `train.py:109`

**Impact**: Lower CPU/RAM usage, minimal GPU impact

**Current Code**:
```python
# Line 37
parser.add_argument("--num_workers", default=8, type=int)

# Line 109
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, pin_memory=True, drop_last=True,
                               num_workers=4)
```

**Modified Code**:
```python
# Line 37
parser.add_argument("--num_workers", default=2, type=int)

# Line 109
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, pin_memory=True, drop_last=True,
                               num_workers=2)
```

**Trade-off**: Data loading may bottleneck training slightly if CPU is weak

---

### Option 3C: Delay CPM Start

**Location**: `train.py:58`

**Impact**: Reduces memory-intensive SAM usage early in training

**Current Code**:
```python
# Line 58 (CPM starts at epoch 2)
parser.add_argument("--sstart", default=2, type=int)
```

**Modified Code**:
```python
# Delay CPM to epoch 5
parser.add_argument("--sstart", default=5, type=int)

# Or delay to epoch 10
parser.add_argument("--sstart", default=10, type=int)
```

**Command Line Usage**:
```bash
python train.py --name exp --model s2c --sstart 5
```

**Trade-off**: Slower convergence to high-quality CAMs, but useful for initial experiments

---

### Option 3D: Use Mixed Precision Training (FP16)

**Location**: Add to `models/model_s2c.py`

**Impact**: ~50% memory reduction, 1.5-2x speedup with minimal quality loss

**Implementation**:

Add imports at the top of `models/model_s2c.py`:
```python
from torch.cuda.amp import autocast, GradScaler
```

In `train_setup()` method (after line 155):
```python
# Add after line 155
self.scaler = GradScaler()
self.logger.info('Mixed precision training enabled (FP16)')
```

Modify the `update()` method to wrap forward and backward:
```python
def update(self, epo, iter):
    # ... existing code ...

    self.net_main.train()
    self.opt_main.zero_grad()

    loss = 0

    # Wrap forward pass with autocast
    with autocast():
        out_main = self.net_main(self.img)
        feat_main = out_main['feat']
        cam_main = out_main['cam']
        pred_main = out_main['pred']

        # ... rest of forward pass ...

        # All loss computations
        self.loss_cls = self.W[0] * self.bce(pred_main, self.label)
        loss += self.loss_cls

        # ... other losses ...

    # Replace loss.backward() with scaled backward
    self.scaler.scale(loss).backward()
    self.scaler.step(self.opt_main)
    self.scaler.update()
```

**Trade-off**: Minimal quality loss with FP16, widely adopted in modern training

---

## Tier 4: Evaluation Optimizations

### Option 4A: Reduce Evaluation MSF Scales

**Location**: `models/model_s2c.py:380-395` and dataset creation

**Impact**: 2-4x faster evaluation

**Current**: Uses 8 passes (4 scales × 2 orientations: normal + flip)

**Modification**: Reduce to 4 passes (2 scales × 2 orientations)

This requires modifying the multi-scale dataset creation in `tools/utils.py` or using fewer scales during evaluation.

**Trade-off**: Slightly less robust evaluation metrics, but faster validation

---

### Option 4B: Skip CRF During Training

**Location**: `train.py:69` and command line

**Impact**: Faster validation epochs

**Current Code**:
```python
# Line 69
parser.add_argument("--crf", action='store_true')
```

**Usage**: Don't add the `--crf` flag during training:
```bash
# CRF disabled (default)
python train.py --name exp --model s2c

# CRF enabled (slower)
python train.py --name exp --model s2c --crf
```

**Trade-off**: CRF refinement only needed for final evaluation, not during training

---

## Recommended Configuration for RTX 4090 (24GB)

Here's the recommended combination for optimal performance/quality trade-off on a single RTX 4090:

### Command Line Configuration

```bash
python train.py \
  --name s2c_4090_scaled \
  --model s2c \
  --batch_size 4 \
  --crop 224 \
  --resize 224 448 \
  --D 192 \
  --sstart 3 \
  --max_epochs 40 \
  --lr 0.02
```

### Required Code Modifications

1. **Switch to SAM ViT-L** (Option 1A)
   - Edit `models/model_s2c.py:98-99`
   - Download SAM ViT-L weights to `./pretrained/sam_vit_l.pth`

2. **Reduce SAM resolution to 768** (Option 2B)
   - Edit `models/model_s2c.py:77`
   - Change `self.size_sam = 1024` to `self.size_sam = 768`

3. **Use 3 scales for MS-CAM** (Option 2A)
   - Edit `models/model_s2c.py:200-205`
   - Remove the 0.5x scale, keep 1.0x, 1.5x, 2.0x

4. **Implement FP16 training** (Option 3D) - Highly Recommended
   - Follow implementation in Option 3D
   - Adds automatic mixed precision for 50% memory savings

### Expected Performance

- **Memory usage**: ~18-20GB VRAM (fits comfortably in 24GB)
- **Training time**: ~12-16 hours for 40 epochs (vs ~6-8 hours on 4 GPUs)
- **Expected mIoU**: 63-66% (vs ~68% in paper)
- **Quality trade-off**: Acceptable for learning, research, and most applications

### Memory Budget Breakdown

| Component | Memory Usage (GB) |
|-----------|-------------------|
| ResNet38d (batch=4, 224x224) | ~4-5 GB |
| SAM ViT-L (768x768) | ~3-4 GB |
| Multi-scale processing (3x) | ~4-5 GB |
| Activations & gradients | ~4-5 GB |
| CUDA overhead | ~1-2 GB |
| **Total** | **~18-20 GB** |

---

## Summary Table

| Option | Location | Memory Saved | Performance Impact | Difficulty | Priority |
|--------|----------|--------------|-------------------|-----------|----------|
| **SAM ViT-B** | `model_s2c.py:98` | 80% (SAM only) | Moderate (-2-3% mIoU) | Easy | **HIGH** |
| **SAM ViT-L** | `model_s2c.py:98` | 50% (SAM only) | Low (-1% mIoU) | Easy | **HIGH** |
| **Batch size 2-4** | `train.py:38` | 50-75% | Low (slower only) | Easy | **HIGH** |
| **FP16 Training** | `model_s2c.py` | 50% | Minimal | Medium | **HIGH** |
| **Reduce crop 224** | `train.py:44` | 30-40% | Moderate (-1-2% mIoU) | Easy | **MEDIUM** |
| **2-3 MS scales** | `model_s2c.py:200` | 25-50% | Low-Moderate | Easy | **MEDIUM** |
| **SAM res 768** | `model_s2c.py:77` | 40% | Low | Easy | **MEDIUM** |
| **SAM res 512** | `model_s2c.py:77` | 60% | Moderate | Easy | **MEDIUM** |
| **Feature dim 192** | `train.py:49` | 20-25% | Low | Easy | **MEDIUM** |
| **Feature dim 128** | `train.py:49` | 30-40% | Moderate | Easy | **LOW** |
| **train.txt only** | `train.py:35` | N/A (time) | Moderate (-2-4% mIoU) | Easy | **LOW** |
| **Gradient checkpointing** | `model_s2c.py` | 30-40% | Low (20% slower) | Medium | **MEDIUM** |
| **Delay CPM start** | `train.py:58` | N/A (early epochs) | Low | Easy | **LOW** |
| **Reduce workers** | `train.py:37,109` | N/A (CPU/RAM) | Minimal | Easy | **LOW** |

---

## Implementation Checklist

### Minimal Changes (Easiest to Implement)
- [ ] Download SAM ViT-L weights
- [ ] Change SAM model to ViT-L in `model_s2c.py:98-99`
- [ ] Run with `--batch_size 4 --crop 224 --resize 224 448`

### Recommended Changes
- [ ] Set SAM resolution to 768 in `model_s2c.py:77`
- [ ] Reduce MS-CAM to 3 scales in `model_s2c.py:200-205`
- [ ] Implement FP16 training (Option 3D)
- [ ] Set feature dimension to 192 with `--D 192`

### Optional Optimizations
- [ ] Implement gradient checkpointing
- [ ] Use smaller training set for faster experiments
- [ ] Delay CPM start to epoch 5
- [ ] Reduce evaluation MSF scales

---

## Troubleshooting

### Out of Memory (OOM) Errors

If you still encounter OOM errors:

1. Further reduce batch size to 2 or 1
2. Switch to SAM ViT-B instead of ViT-L
3. Reduce crop size to 192
4. Implement gradient checkpointing
5. Monitor memory with: `nvidia-smi -l 1`

### Training Instability

If training becomes unstable with smaller batch sizes:

1. Reduce learning rate proportionally: `--lr 0.01` (for batch_size=2)
2. Increase warmup period
3. Use gradient clipping in the optimizer

### Slow Training

If training is too slow:

1. Ensure FP16 training is enabled
2. Check data loading isn't bottlenecking (increase `num_workers` if CPU allows)
3. Use `train.txt` instead of `train_aug.txt` for faster experimentation
4. Profile with PyTorch profiler to identify bottlenecks

---

## Additional Resources

- Original Paper: [CVPR 2024 - From SAM to CAMs](https://arxiv.org/abs/2310.00966)
- SAM Repository: https://github.com/facebookresearch/segment-anything
- PyTorch AMP Guide: https://pytorch.org/docs/stable/amp.html
- Memory Optimization Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

**Last Updated**: 2025-12-05
**GPU Target**: NVIDIA RTX 4090 (24GB VRAM)
**Original Setup**: 4x GPUs (likely V100 or A100)
