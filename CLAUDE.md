# F1 Aerodynamic Design Pipeline

AI-assisted F1 car aerodynamic design with multi-backend simulation and ML surrogate models. Baseline: Red Bull RB19/RB20, FIA 2024 regulations.

## Architecture

```
blender/f1_car_generator.py       Parametric F1 car geometry (monocoque, wings, floor, diffuser, sidepods, halo, wheels)
scripts/run_pipeline.py           Pipeline orchestrator with 5 simulation backends + ML training + active learning
scripts/omniverse_sim.py          NVIDIA Omniverse Flow GPU CFD backend
scripts/convert_to_usd.py         STL -> USD conversion for Omniverse (wind tunnel scene)
scripts/modulus_surrogate.py      NVIDIA Modulus PINN surrogate (F1AeroNet, physics-constrained)
scripts/install.sh                macOS tool installer (Homebrew-based)
config.yaml                       Central configuration (backends, physics, ML hyperparameters, paths)
ml/                               ML surrogate pipeline:
  ml/surrogate.py                   4 model types: Linear, MLP, GP (with uncertainty), PINN
  ml/data_prep.py                   Data loading, normalization, train/test split
  ml/experiment.py                  Experiment runner + model comparison
  ml/autoresearch_config.py         Karpathy's autoresearch pattern (seed paper + experiment loop)
  ml/active_learning.py             GP uncertainty-driven parameter proposal + simulation
openfoam/f1_baseline/             Complete OpenFOAM case (simpleFoam, k-omega SST, 300 km/h, Re~31M)
results/sweep_all.json            32 empirical parameter sweep results (no CFD validation yet)
```

## Simulation Backends (fallback chain)

| Priority | Backend | Command | What it does |
|---|---|---|---|
| 1 | omniverse | `--backend omniverse` | NVIDIA Omniverse Flow GPU CFD |
| 2 | openfoam | `--backend openfoam` | OpenFOAM simpleFoam CPU CFD (default) |
| 3 | modulus | `--backend modulus` | NVIDIA Modulus PINN (~1ms inference, auto-trains on first use) |
| 4 | surrogate | `--backend surrogate` | Best available ML model from ml/models/ (GP > MLP > PINN > Linear) |
| 5 | estimate | `--backend estimate` | Empirical correlations (always available, instant) |

## Key Commands

```bash
# Generate F1 car geometry
blender --background --python blender/f1_car_generator.py -- --ride-height 0.030 --rear-wing-angle 18

# Run pipeline with different backends
python3 scripts/run_pipeline.py                              # Default: OpenFOAM
python3 scripts/run_pipeline.py --backend omniverse          # GPU CFD via Omniverse
python3 scripts/run_pipeline.py --backend modulus            # PINN surrogate
python3 scripts/run_pipeline.py --backend surrogate          # Best ML model
python3 scripts/run_pipeline.py --estimate-only              # Empirical only

# Parameter sweeps
python3 scripts/run_pipeline.py --sweeps all --estimate-only
python3 scripts/run_pipeline.py --sweeps ride_height --backend modulus

# ML surrogate training (Karpathy's autoresearch pattern)
python3 scripts/run_pipeline.py --train-surrogate

# Active learning (propose + simulate N new data points)
python3 scripts/run_pipeline.py --active-learn 5 --backend omniverse

# Run ML experiments directly
python3 -m ml.experiment --model gp --data results/sweep_all.json
python3 -m ml.autoresearch_config setup
python3 -m ml.autoresearch_config run
python3 -m ml.active_learning --iterations 3

# Manual OpenFOAM
cd openfoam/f1_baseline && blockMesh && snappyHexMesh -overwrite && simpleFoam

# Install tools (macOS)
chmod +x scripts/install.sh && ./scripts/install.sh
pip install -r requirements.txt
```

## Parametric Variables

| Variable | Default | Range | Unit |
|---|---|---|---|
| ride_height | 0.030 | 0.020 - 0.050 | meters |
| front_wing_angle | 14.0 | 10 - 20 | degrees |
| rear_wing_angle | 16.0 | 10 - 22 | degrees |
| diffuser_angle | 12.0 | 6 - 18 | degrees |
| sidepod_undercut | 0.15 | 0.08 - 0.20 | meters |

Bounds defined in both `config.yaml` (truth source) and `ml/data_prep.py` (hard-coded duplicate).

## ML Surrogate Models

| Model | Class | Strengths | Use case |
|---|---|---|---|
| Linear | `LinearSurrogate` | Fast, interpretable, polynomial features | Baseline comparison |
| MLP | `MLPSurrogate` | Captures nonlinearity, PyTorch + CUDA | General purpose |
| GP | `GPSurrogate` | Uncertainty estimates, GPyTorch/sklearn dual backend | Active learning |
| PINN | `ModulusSurrogate` | Physics constraints (Cd>0, Cl<0, range bounds) | **Best overall** (R² 0.85-0.92 at 32pts) |

## Code Conventions

- Python dataclasses for structured data (`SimulationParams`, `SimulationResult`)
- Blender script uses `argparse` with `--` separator for Blender CLI arg passthrough
- All dimensions in SI units (meters, degrees, m/s, kg/m^3)
- FIA reference constants at module level
- ML models follow base class interface: `fit()`, `predict()`, optional `uncertainty()`, `save()`, `load()`
- Results stored as JSON arrays; experiment logs as JSONL
- Physics violations tracked: Cd<0, Cl>0, Cd>2.0, Cl<-8.0

## What's NOT Implemented Yet

- **No CFD runs completed** - all 32 results in sweep_all.json are empirical estimates
- **OpenClaw/NemoClaw AI orchestration** - referenced in README but no code exists
- **config.yaml is disconnected** - ml/ modules use hard-coded values, not config.yaml
- **Autoresearch ran but limited** - 11 experiments completed (see EXPERIMENT.md); manual loop with 7 hard-coded HP experiments; no LLM-driven hypothesis generation yet
- **Active learning + autoresearch disconnected** - parallel workflows that should be unified
- **Warp LBM solver** - placeholder in omniverse_sim.py (returns None)
- **README.md is significantly outdated** - references non-existent files (export_stl.py, postprocessing/, docs/), describes OpenClaw as if implemented, architecture diagram shows only OpenClaw orchestration (not the actual multi-backend pipeline), directory structure is missing ml/, omniverse/, config.yaml, and all documentation files
- No tests, no CI/CD, no web UI
- No multi-variable interaction sweeps

## Related Docs

- [DESIGN.md](DESIGN.md) - Architecture decisions and aerodynamic rationale
- [EXPERIMENT.md](EXPERIMENT.md) - Parameter sweep results and ML experiment tracking
- [RESEARCH.md](RESEARCH.md) - Literature references, autoresearch, OpenClaw research
- [NEXT-TO-DO.md](NEXT-TO-DO.md) - Prioritized backlog
