# F1 Car Aerodynamic Design - AI-Assisted Pipeline

An open-source, AI-assisted workflow for designing efficient Formula 1 cars using
Blender, OpenFOAM, ParaView, and OpenClaw as the AI orchestration layer.

Baseline reference: **Red Bull RB19/RB20** design philosophy.

## Architecture

```
OpenClaw (AI Assistant)
    |
    +-- Research: Literature review (arXiv, ResearchGate)
    +-- Design: Blender Python API for parametric geometry
    +-- Simulation: OpenFOAM CFD (simpleFoam + k-omega SST)
    +-- Analysis: ParaView post-processing + optimization
    +-- Materials: Composite layup & material selection (Elmer FEM)
```

## Directory Structure

```
f1-design-ai/
├── scripts/           # Installation & automation scripts
│   ├── install.sh     # Install all required tools (macOS)
│   └── run_pipeline.py  # End-to-end Blender -> OpenFOAM -> ParaView
├── blender/           # Blender Python scripts
│   ├── f1_car_generator.py   # Parametric F1 car geometry
│   └── export_stl.py         # Export meshes for OpenFOAM
├── openfoam/          # OpenFOAM case directories
│   └── f1_baseline/   # Baseline CFD case
│       ├── 0/         # Initial/boundary conditions
│       ├── constant/  # Physical properties & mesh
│       └── system/    # Solver settings
├── postprocessing/    # ParaView scripts & visualization
├── results/           # Simulation outputs & comparisons
└── docs/              # Research papers, notes, references
```

## Quick Start

```bash
# 1. Install tools
chmod +x scripts/install.sh
./scripts/install.sh

# 2. Generate baseline F1 car geometry
blender --background --python blender/f1_car_generator.py

# 3. Run CFD simulation
cd openfoam/f1_baseline
blockMesh
snappyHexMesh -overwrite
simpleFoam

# 4. Post-process results
paraview openfoam/f1_baseline/postProcessing/

# 5. Run full pipeline with optimization
python3 scripts/run_pipeline.py --sweeps ride_height,wing_angle
```

## Tool Stack

| Tool | Version | Role |
|------|---------|------|
| Blender | 4.x | 3D parametric modeling via Python API |
| OpenFOAM | v2406+ | CFD simulation (simpleFoam, k-omega SST) |
| ParaView | 5.12+ | Flow visualization & post-processing |
| FreeCAD + CfdOF | 0.22+ | Alternative parametric design path |
| Python | 3.11+ | Pipeline orchestration |
| OpenClaw | latest | AI assistant for research & automation |

## Key Research Papers

- Ravelli & Savini (2018) - F1 CFD with OpenFOAM (5.7% drag error)
- AI Design Agents (2025) - Multi-agent Blender+OpenFOAM framework
- DrivAerNet++ (NeurIPS 2024) - 8,150 car designs + CFD dataset
- F1 Front Wing PINNs - Neural network CFD surrogate (R^2=0.968)

## Red Bull RB19/RB20 Design Concepts

- Ground effect Venturi tunnels (primary downforce)
- Floor strakes + edge vortex generators (aero seal)
- Anti-dive/anti-squat suspension (consistent ride height)
- Full-width front wing (conditions flow for entire car)
- Aggressive sidepod undercut with turning vanes
