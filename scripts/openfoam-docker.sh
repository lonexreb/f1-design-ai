#!/usr/bin/env bash
# OpenFOAM Docker wrapper - runs OpenFOAM commands in a container
# Usage: ./openfoam-docker.sh <command> [args...]
# Example: ./openfoam-docker.sh simpleFoam
#          ./openfoam-docker.sh blockMesh

CASE_DIR="${OPENFOAM_CASE_DIR:-$(pwd)}"
OPENFOAM_IMAGE="${OPENFOAM_IMAGE:-microfluidica/openfoam:2406}"

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
    --platform linux/amd64 \
    -v "$CASE_DIR":/data \
    -w /data \
    "$OPENFOAM_IMAGE" \
    "$@"
