#!/bin/bash

# Set environment name and Python version
ENV_NAME="map4d"
PYTHON_VERSION="3.10.14"
CUDA_VERSION="11.8"
PYTORCH_VERSION="2.0.1" # Adjust if a newer compatible version is required

echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION, CUDA $CUDA_VERSION, and PyTorch $PYTORCH_VERSION."

# Create the Conda environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
source activate $ENV_NAME

# Install PyTorch and CUDA toolkit
conda install -y pytorch=$PYTORCH_VERSION torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
pip install -r requirements.txt
# Verify the installation
echo "Verifying installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

echo "Environment '$ENV_NAME' created successfully!"
