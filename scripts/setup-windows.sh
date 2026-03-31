#!/usr/bin/env bash
# F1 Design AI - Windows Setup Script
# Prerequisites: Python 3.11+, Git, Docker Desktop, Blender 4.x, NVIDIA GPU + drivers
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  F1 Design AI - Windows Setup"
echo "=========================================="
echo ""

# ---- Python venv ----
if [ -d ".venv" ]; then
    info "Virtual environment already exists."
else
    info "Creating virtual environment..."
    python -m venv .venv
fi

info "Activating virtual environment..."
source .venv/Scripts/activate

# ---- Pip upgrade ----
info "Upgrading pip..."
python -m pip install --upgrade pip

# ---- PyTorch with CUDA ----
info "Installing PyTorch with CUDA..."
pip install torch --index-url https://download.pytorch.org/whl/cu126

# ---- Core dependencies ----
info "Installing core dependencies..."
pip install numpy scipy matplotlib meshio pyvista numpy-stl \
    scikit-learn gpytorch botorch pyyaml pandas tqdm usd-core

# ---- Optional: NVIDIA Modulus (requires Python <3.13) ----
if python -c "import sys; exit(0 if sys.version_info < (3,13) else 1)" 2>/dev/null; then
    info "Installing NVIDIA Modulus..."
    pip install nvidia-modulus || warn "nvidia-modulus install failed (optional)"
else
    warn "Skipping nvidia-modulus (requires Python <3.13, you have $(python --version))"
fi

# ---- Check tools ----
echo ""
echo "=========================================="
echo "  Setup Summary"
echo "=========================================="
echo ""

check() {
    if "$@" &>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} $1"
    else
        echo -e "  ${RED}[--]${NC} $1 (not found)"
    fi
}

check python --version
check pip --version
check git --version
check docker --version
check nvidia-smi

# Blender check
BLENDER_EXE="/c/Program Files/Blender Foundation/Blender 4.3/blender.exe"
if [ -f "$BLENDER_EXE" ]; then
    echo -e "  ${GREEN}[OK]${NC} Blender ($BLENDER_EXE)"
else
    echo -e "  ${YELLOW}[--]${NC} Blender not found at expected path"
    echo "       Download from https://www.blender.org/download/"
fi

# OpenFOAM Docker
if docker image inspect openfoam/openfoam-default:2406 &>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} OpenFOAM Docker image"
else
    echo -e "  ${YELLOW}[--]${NC} OpenFOAM Docker image not pulled yet"
    echo "       Run: docker pull openfoam/openfoam-default:2406"
fi

# PyTorch CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_DEV=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "  ${GREEN}[OK]${NC} PyTorch CUDA ($CUDA_DEV)"
else
    echo -e "  ${YELLOW}[--]${NC} PyTorch CUDA not available"
fi

echo ""
info "Setup complete! Activate the environment with:"
echo "  source .venv/Scripts/activate"
echo ""
info "Quick start:"
echo "  python scripts/run_pipeline.py --estimate-only"
echo "  blender --background --python blender/f1_car_generator.py"
echo ""
