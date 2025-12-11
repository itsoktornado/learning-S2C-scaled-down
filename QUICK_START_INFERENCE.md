# Quick Start: Inference on a New Device

This is a quick reference guide for running inference with a trained S2C model on a different device.

---

## ⚡ Quick Setup (5 Steps)

### 1. Copy Files from Training Device

Copy these files to a USB drive or transfer via network:

```bash
# The trained model checkpoint (most important!)
./experiments/[your_exp_name]/ckpt/015net_main.pth

# Pretrained weights (if new device doesn't have them)
./pretrained/resnet_38d.params
./pretrained/sam_vit_b_01ec64.pth
```

### 2. Setup on New Device

```bash
# Clone the S2C repository
git clone [repo-url]
cd S2C

# Install dependencies (torch-scatter NOT needed for inference!)
pip install torch torchvision opencv-python pillow matplotlib numpy scipy scikit-image tqdm

# Optional (can be skipped for inference):
pip install git+https://github.com/facebookresearch/segment-anything.git

# Create directories
mkdir -p pretrained
mkdir -p experiments/[your_exp_name]/ckpt
```

### 3. Place Files

```bash
# Place pretrained weights
cp /path/to/transfer/resnet_38d.params ./pretrained/
cp /path/to/transfer/sam_vit_b_01ec64.pth ./pretrained/

# Place your trained model
cp /path/to/transfer/015net_main.pth ./experiments/[your_exp_name]/ckpt/
```

### 4. Verify Setup

```bash
python check_inference_setup.py
```

This will check if everything is ready. Fix any missing files it reports.

### 5. Run Inference!

```bash
# Single image
python infer_single_image.py \
    --image_path /path/to/image.jpg \
    --checkpoint ./experiments/[your_exp_name]/ckpt/015net_main.pth \
    --output_dir ./results

# Batch inference (whole folder)
python infer_batch.py \
    --image_dir /path/to/images/ \
    --checkpoint ./experiments/[your_exp_name]/ckpt/015net_main.pth \
    --output_dir ./results_batch
```

---

## 📋 Common Use Cases

### Example 1: Test a single cat image

```bash
python infer_single_image.py \
    --image_path ./test_images/cat.jpg \
    --checkpoint ./experiments/myexp/ckpt/015net_main.pth \
    --class_indices "7" \
    --output_dir ./cat_results
```

### Example 2: Process multiple images with people and cars

```bash
python infer_batch.py \
    --image_dir ./test_images/ \
    --checkpoint ./experiments/myexp/ckpt/015net_main.pth \
    --class_indices "6,14" \
    --output_dir ./results
```

### Example 3: Let model detect all classes automatically

```bash
python infer_single_image.py \
    --image_path ./image.jpg \
    --checkpoint ./experiments/myexp/ckpt/015net_main.pth \
    --output_dir ./results
```

---

## 🎯 VOC Class IDs Quick Reference

```
0=aeroplane   5=bus        10=diningtable  15=pottedplant
1=bicycle     6=car        11=dog          16=sheep
2=bird        7=cat        12=horse        17=sofa
3=boat        8=chair      13=motorbike    18=train
4=bottle      9=cow        14=person       19=tvmonitor
```

**Example:** For an image with a person (14) riding a horse (12):
```bash
--class_indices "12,14"
```

---

## 📁 Expected File Structure on New Device

```
S2C/
├── pretrained/
│   ├── resnet_38d.params              ← Required
│   └── sam_vit_b_01ec64.pth           ← Required
├── experiments/
│   └── [your_exp_name]/
│       └── ckpt/
│           └── 015net_main.pth        ← Your trained model
├── infer_single_image.py              ← New script
├── infer_batch.py                     ← New script
├── check_inference_setup.py           ← New script
└── [source code files...]
```

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Checkpoint not found" | Check path: `./experiments/[exp_name]/ckpt/015net_main.pth` |
| "resnet_38d.params not found" | Download and place in `./pretrained/` |
| "Out of memory" | Use `--scales 1.0` for single-scale inference |
| "Module not found" | Run `pip install [missing-package]` |
| Model load error | Ensure `--C 20 --D 256` match training config |

---

## 🔍 What You Get

After inference, the output directory contains:

```
./results/
├── image_name_cam_cat.png           # CAM visualization overlaid on image
├── image_name_cam_person.png        # One per detected class
└── image_name_cam_dict.npy          # Raw CAM data for further processing
```

---

## 📖 More Information

- **Detailed Guide**: See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for comprehensive documentation
- **Training**: See [README.md](README.md) for training instructions
- **Paper**: CVPR 2024 - "From SAM to CAMs"

---

## 💡 Tips

1. **Specify classes** when you know what's in the image for better results:
   ```bash
   --class_indices "7,14"  # cat and person
   ```

2. **Use batch inference** for multiple images - it's more efficient:
   ```bash
   python infer_batch.py --image_dir ./images/ --checkpoint ...
   ```

3. **Save GPU memory** by using single-scale inference:
   ```bash
   --scales 1.0
   ```

4. **CPU inference** works but is slower:
   ```bash
   CUDA_VISIBLE_DEVICES="" python infer_single_image.py ...
   ```

---

## ✅ Checklist Before First Run

- [ ] Source code cloned/copied
- [ ] Dependencies installed (`pip install ...`)
- [ ] Pretrained weights in `./pretrained/`
- [ ] Trained checkpoint in `./experiments/[exp_name]/ckpt/`
- [ ] Ran `python check_inference_setup.py` successfully
- [ ] Test image ready

**Ready?** Run your first inference! 🚀

```bash
python infer_single_image.py \
    --image_path YOUR_IMAGE.jpg \
    --checkpoint ./experiments/EXPNAME/ckpt/015net_main.pth
```
