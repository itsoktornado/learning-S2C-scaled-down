@echo off
REM Complete dependency installation script for S2C inference (Windows)

echo ==================================================
echo Installing S2C Inference Dependencies
echo ==================================================
echo.

echo Step 1: Installing core PyTorch packages...
pip install torch torchvision
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 2: Installing torch-scatter...
pip install torch-scatter
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Simple installation failed. Checking PyTorch/CUDA versions...
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
    echo.
    echo Please install torch-scatter manually for your configuration:
    echo Visit: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
    echo.
    echo Example commands:
    echo   For CUDA 11.8:
    echo     pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    echo   For CPU only:
    echo     pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    echo.
    pause
)

echo.
echo Step 3: Installing image processing libraries...
pip install opencv-python pillow matplotlib

echo.
echo Step 4: Installing scientific computing libraries...
pip install numpy scipy scikit-image

echo.
echo Step 5: Installing Segment Anything...
pip install git+https://github.com/facebookresearch/segment-anything.git

echo.
echo ==================================================
echo Installation Complete!
echo ==================================================
echo.
echo Run verification:
echo   python check_inference_setup.py
echo.
pause
