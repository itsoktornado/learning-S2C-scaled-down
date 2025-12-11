#!/bin/bash
# Complete dependency installation script for S2C inference

echo "=================================================="
echo "Installing S2C Inference Dependencies"
echo "=================================================="
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found. Please install Python and pip first."
    exit 1
fi

echo "Step 1: Installing core PyTorch packages..."
pip install torch torchvision

echo ""
echo "Step 2: Installing torch-scatter..."
# Try simple install first
if pip install torch-scatter; then
    echo "✓ torch-scatter installed successfully"
else
    echo "⚠ Simple installation failed. Checking PyTorch/CUDA versions..."

    # Get PyTorch and CUDA versions
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)

    echo "PyTorch version: $TORCH_VERSION"
    echo "CUDA version: $CUDA_VERSION"
    echo ""
    echo "Please install torch-scatter manually for your configuration:"
    echo "Visit: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    echo ""
    echo "Example commands:"
    echo "  For CUDA 11.8:"
    echo "    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
    echo "  For CPU only:"
    echo "    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html"
    echo ""
    read -p "Press Enter after installing torch-scatter manually, or Ctrl+C to exit..."
fi

echo ""
echo "Step 3: Installing image processing libraries..."
pip install opencv-python pillow matplotlib

echo ""
echo "Step 4: Installing scientific computing libraries..."
pip install numpy scipy scikit-image

echo ""
echo "Step 5: Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Run verification:"
echo "  python check_inference_setup.py"
echo ""
