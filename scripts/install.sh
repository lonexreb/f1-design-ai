#!/usr/bin/env bash
# F1 Design AI - Tool Installation Script (macOS)
# Installs: Blender, OpenFOAM, ParaView, FreeCAD, Python deps
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    error "This script is for macOS. For Linux, install packages via your distro's package manager."
    echo "  Ubuntu/Debian: sudo apt install openfoam blender paraview freecad"
    echo "  Fedora:        sudo dnf install openfoam blender paraview freecad"
    exit 1
fi

# Check Homebrew
if ! command -v brew &>/dev/null; then
    error "Homebrew not found. Install it first:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

info "Updating Homebrew..."
brew update

# ---- Blender ----
if command -v blender &>/dev/null || [ -d "/Applications/Blender.app" ]; then
    info "Blender already installed: $(blender --version 2>/dev/null | head -1 || echo 'found in /Applications')"
else
    info "Installing Blender..."
    brew install --cask blender
    info "Blender installed."
fi

# ---- OpenFOAM ----
if command -v simpleFoam &>/dev/null; then
    info "OpenFOAM already installed."
else
    info "Installing OpenFOAM via Homebrew..."
    # OpenFOAM is available via a tap
    brew tap OpenFOAM/OpenFOAM 2>/dev/null || true
    if brew install openfoam2406 2>/dev/null; then
        info "OpenFOAM installed via Homebrew."
    else
        warn "Homebrew OpenFOAM install failed. Trying alternative methods..."
        echo ""
        echo "Manual install options:"
        echo "  1. Docker (recommended for macOS):"
        echo "     docker pull openfoam/openfoam-dev"
        echo "     docker run -it -v \$(pwd):/data openfoam/openfoam-dev"
        echo ""
        echo "  2. Compile from source:"
        echo "     git clone https://develop.openfoam.com/Development/openfoam.git"
        echo "     cd openfoam && source etc/bashrc && ./Allwmake -j"
        echo ""
        echo "  3. Use OpenFOAM via Docker wrapper (easiest):"
        echo "     We'll create a wrapper script for you."

        # Create Docker wrapper
        WRAPPER_DIR="$(cd "$(dirname "$0")/.." && pwd)/scripts"
        cat > "$WRAPPER_DIR/openfoam-docker.sh" << 'DOCKER_EOF'
#!/usr/bin/env bash
# OpenFOAM Docker wrapper - runs OpenFOAM commands in a container
# Usage: ./openfoam-docker.sh <command> [args...]
# Example: ./openfoam-docker.sh simpleFoam
#          ./openfoam-docker.sh blockMesh

CASE_DIR="${OPENFOAM_CASE_DIR:-$(pwd)}"
OPENFOAM_IMAGE="${OPENFOAM_IMAGE:-openfoam/openfoam-dev}"

if ! command -v docker &>/dev/null; then
    echo "Error: Docker not found. Install Docker Desktop first."
    exit 1
fi

# Pull image if not present
if ! docker image inspect "$OPENFOAM_IMAGE" &>/dev/null; then
    echo "Pulling OpenFOAM Docker image..."
    docker pull "$OPENFOAM_IMAGE"
fi

docker run --rm \
    -v "$CASE_DIR":/data \
    -w /data \
    "$OPENFOAM_IMAGE" \
    "$@"
DOCKER_EOF
        chmod +x "$WRAPPER_DIR/openfoam-docker.sh"
        info "Created Docker wrapper at scripts/openfoam-docker.sh"
    fi
fi

# ---- ParaView ----
if command -v paraview &>/dev/null || [ -d "/Applications/ParaView.app" ]; then
    info "ParaView already installed."
else
    info "Installing ParaView..."
    brew install --cask paraview
    info "ParaView installed."
fi

# ---- FreeCAD ----
if command -v freecad &>/dev/null || [ -d "/Applications/FreeCAD.app" ]; then
    info "FreeCAD already installed."
else
    info "Installing FreeCAD..."
    brew install --cask freecad
    info "FreeCAD installed."
fi

# ---- Python dependencies ----
info "Installing Python dependencies..."
pip3 install --user numpy scipy matplotlib meshio pyvista stl 2>/dev/null || \
pip3 install numpy scipy matplotlib meshio pyvista stl

# ---- BlenderFOAM (optional) ----
BLENDERFOAM_DIR="$(cd "$(dirname "$0")/.." && pwd)/vendor/BlenderFOAM"
if [ -d "$BLENDERFOAM_DIR" ]; then
    info "BlenderFOAM already cloned."
else
    info "Cloning BlenderFOAM..."
    mkdir -p "$(dirname "$BLENDERFOAM_DIR")"
    git clone https://github.com/nathanrooy/BlenderFOAM.git "$BLENDERFOAM_DIR" 2>/dev/null || \
        warn "Could not clone BlenderFOAM. You can clone it manually later."
fi

# ---- Summary ----
echo ""
echo "=========================================="
echo "  F1 Design AI - Installation Summary"
echo "=========================================="
echo ""

check_tool() {
    if command -v "$1" &>/dev/null || [ -d "$2" 2>/dev/null ]; then
        echo -e "  ${GREEN}[OK]${NC} $3"
    else
        echo -e "  ${RED}[--]${NC} $3 (not found - install manually)"
    fi
}

check_tool blender "/Applications/Blender.app" "Blender"
check_tool simpleFoam "" "OpenFOAM"
check_tool paraview "/Applications/ParaView.app" "ParaView"
check_tool freecad "/Applications/FreeCAD.app" "FreeCAD"
check_tool python3 "" "Python 3"

echo ""
info "Next steps:"
echo "  1. Generate F1 car geometry:  blender --background --python blender/f1_car_generator.py"
echo "  2. Run CFD simulation:        cd openfoam/f1_baseline && blockMesh && simpleFoam"
echo "  3. Visualize results:         paraview openfoam/f1_baseline/postProcessing/"
echo ""
