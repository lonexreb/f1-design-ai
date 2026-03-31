# F1 Car Aerodynamic Design - AI-Assisted Pipeline

An open-source, AI-assisted workflow for designing efficient Formula 1 cars using
parametric geometry (Blender), multi-backend CFD simulation, and ML surrogate models.

Baseline reference: **Red Bull RB19/RB20** design philosophy, FIA 2024 regulations.

## Architecture

```
Parameter --> Blender --> STL --> Backend Router --> Force Extraction --> Results JSON
Selection     Geometry    Export      |                                       |
                                     +---> Omniverse Flow (GPU CFD)          |
                                     +---> OpenFOAM (CPU CFD)                +---> ML Training
                                     +---> Modulus PINN (~1ms)               |     (Autoresearch)
                                     +---> ML Surrogate (GP/MLP)             |
                                     +---> Empirical Estimate                +---> Active Learning
```

## Directory Structure

```
f1-design-ai/
├── blender/
│   └── f1_car_generator.py       # Parametric F1 car geometry (Blender 4.x/5.x)
├── scripts/
│   ├── run_pipeline.py           # Pipeline orchestrator (5 backends + ML + active learning)
│   ├── omniverse_sim.py          # NVIDIA Omniverse Flow GPU CFD backend
│   ├── convert_to_usd.py         # STL -> USD conversion for Omniverse
│   ├── modulus_surrogate.py      # NVIDIA Modulus PINN surrogate (F1AeroNet)
│   ├── openfoam-docker.sh        # OpenFOAM Docker wrapper (Apple Silicon compatible)
│   ├── install.sh                # macOS tool installer (Homebrew-based)
│   └── setup-windows.sh          # Windows setup (venv, PyTorch CUDA, deps)
├── ml/                           # ML surrogate pipeline
│   ├── surrogate.py              # 4 model types: Linear, MLP, GP, PINN
│   ├── data_prep.py              # Data loading, normalization, train/test split
│   ├── experiment.py             # Experiment runner + model comparison
│   ├── autoresearch_config.py    # Karpathy's autoresearch pattern
│   ├── config.py                 # Config loader (reads config.yaml, typed access, caching)
│   ├── active_learning.py        # GP uncertainty-driven parameter proposal
│   └── models/                   # Trained model artifacts (.pkl, .pt)
├── openfoam/
│   └── f1_baseline/              # Complete OpenFOAM case (simpleFoam, k-omega SST)
│       ├── 0/                    # Initial/boundary conditions
│       ├── constant/             # Physical properties & mesh
│       └── system/               # Solver settings
├── autoresearch/                 # Autoresearch experiment tracking
│   ├── seed_paper.md             # Research problem description
│   ├── run.py                    # Experiment entry point
│   └── experiment_log.json       # Results from 11 experiments
├── results/
│   └── sweep_all.json            # 32 empirical parameter sweep results
├── config.yaml                   # Central configuration (backends, physics, ML, paths)
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Architecture & conventions (source of truth)
├── DESIGN.md                     # Architecture decisions & rationale
├── EXPERIMENT.md                 # Parameter sweep results & ML experiment tracking
├── RESEARCH.md                   # Literature references & research directions
└── NEXT-TO-DO.md                 # Prioritized backlog
```

## Quick Start

```bash
# 1. Install tools
# macOS:
chmod +x scripts/install.sh && ./scripts/install.sh
# Windows (Git Bash / WSL):
chmod +x scripts/setup-windows.sh && ./scripts/setup-windows.sh
# Or manually:
pip install -r requirements.txt

# 2. Generate baseline F1 car geometry
blender --background --python blender/f1_car_generator.py

# 3. Run pipeline (choose a backend)
python3 scripts/run_pipeline.py --estimate-only              # Empirical (instant, no tools needed)
python3 scripts/run_pipeline.py --backend openfoam           # OpenFOAM CPU CFD
python3 scripts/run_pipeline.py --backend omniverse          # NVIDIA Omniverse GPU CFD
python3 scripts/run_pipeline.py --backend surrogate          # Best trained ML model

# 4. Parameter sweeps
python3 scripts/run_pipeline.py --sweeps all --estimate-only

# 5. Train ML surrogates (Karpathy's autoresearch pattern)
python3 scripts/run_pipeline.py --train-surrogate

# 6. Active learning (propose + simulate new data points)
python3 scripts/run_pipeline.py --active-learn 5 --backend omniverse

# OpenFOAM via Docker (if not installed locally)
./scripts/openfoam-docker.sh blockMesh
./scripts/openfoam-docker.sh simpleFoam
```

## Simulation Backends

The pipeline supports 5 backends with automatic fallback:

| Priority | Backend | What it does |
|---|---|---|
| 1 | Omniverse Flow | NVIDIA GPU-accelerated CFD |
| 2 | OpenFOAM | CPU CFD (simpleFoam, k-omega SST, Docker fallback) |
| 3 | Modulus PINN | Physics-informed neural network (~1ms inference) |
| 4 | ML Surrogate | Best trained model from ml/models/ |
| 5 | Empirical | Correlation-based estimates (always available) |

## ML Surrogate Models

| Model | Strengths | Test R² (32pts) |
|---|---|---|
| **PINN** | Physics constraints, best generalization | **0.85-0.92** |
| GP | Uncertainty estimates for active learning | 0.51-0.99 (uneven) |
| MLP | Nonlinear, PyTorch + CUDA/MPS | 0.02-0.93 |
| Linear | Fast, interpretable | -1.83-0.76 |

PINN is the best overall surrogate at small dataset sizes. See [EXPERIMENT.md](EXPERIMENT.md) for full results.

## Parametric Design Space

| Variable | Default | Range | Unit |
|---|---|---|---|
| ride_height | 0.030 | 0.020 - 0.050 | meters |
| front_wing_angle | 14.0 | 10 - 20 | degrees |
| rear_wing_angle | 16.0 | 10 - 22 | degrees |
| diffuser_angle | 12.0 | 6 - 18 | degrees |
| sidepod_undercut | 0.15 | 0.08 - 0.20 | meters |

## Tool Stack

| Tool | Role |
|---|---|
| Blender 4.x/5.x | Parametric geometry via Python API |
| OpenFOAM v2406+ | CFD simulation (simpleFoam, k-omega SST) |
| NVIDIA Omniverse | GPU CFD via Flow extension |
| NVIDIA Modulus | Physics-informed neural network surrogates |
| PyTorch | ML model training (CUDA/MPS/CPU) |
| GPyTorch / scikit-learn | Gaussian Process surrogates |
| Docker | OpenFOAM containerized fallback |

## Key Research References

- Ravelli & Savini (2018) — F1 CFD with OpenFOAM (5.7% drag error vs wind tunnel)
- DrivAerNet++ (NeurIPS 2024) — 8,150 car designs + CFD dataset
- F1 Front Wing PINNs — Neural network CFD surrogate (R²=0.968)
- Karpathy's Autoresearch (2026) — Autonomous ML research pattern
- AI Design Agents (ArXiv 2025) — Multi-agent aero design framework

## Documentation

| File | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Architecture, commands, conventions (source of truth) |
| [DESIGN.md](DESIGN.md) | Architecture decisions with rationale |
| [EXPERIMENT.md](EXPERIMENT.md) | Parameter sweep results, ML experiment tracking |
| [RESEARCH.md](RESEARCH.md) | Literature references, research directions |
| [NEXT-TO-DO.md](NEXT-TO-DO.md) | Prioritized backlog (P1-P7) |

## Current Status

- 32 empirical data points (no CFD validation yet)
- 4 trained ML surrogates (PINN best at MSE=0.0038)
- 11 autoresearch experiments completed
- OpenFOAM case configured with coarse mesh for first run
- Pipeline end-to-end validation in progress
