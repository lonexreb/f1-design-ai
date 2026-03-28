# F1 Aerodynamic Design Pipeline

AI-assisted F1 car aerodynamic design: Blender (parametric geometry) -> OpenFOAM (CFD) -> ParaView (post-processing). Baseline: Red Bull RB19/RB20, FIA 2024 regulations.

## Architecture

```
blender/f1_car_generator.py    Parametric F1 car geometry (monocoque, wings, floor, diffuser, sidepods, halo, wheels)
scripts/run_pipeline.py        Pipeline orchestrator (Blender -> OpenFOAM -> results extraction + empirical fallback)
scripts/install.sh             macOS tool installer (Homebrew-based)
openfoam/f1_baseline/          Complete OpenFOAM case (simpleFoam, k-omega SST, 300 km/h, Re~31M)
results/sweep_all.json         32 empirical parameter sweep results (no CFD validation yet)
```

## Key Commands

```bash
# Generate F1 car geometry
blender --background --python blender/f1_car_generator.py -- --ride-height 0.030 --rear-wing-angle 18

# Run full pipeline (Blender + OpenFOAM)
python3 scripts/run_pipeline.py

# Run parameter sweep (empirical estimates only)
python3 scripts/run_pipeline.py --sweeps all --estimate-only

# Run single-variable sweep
python3 scripts/run_pipeline.py --sweeps ride_height

# Run with Docker OpenFOAM
python3 scripts/run_pipeline.py --docker

# Manual OpenFOAM execution
cd openfoam/f1_baseline && blockMesh && snappyHexMesh -overwrite && simpleFoam

# Install tools (macOS)
chmod +x scripts/install.sh && ./scripts/install.sh
```

## Parametric Variables

| Variable | Default | Range | Unit |
|---|---|---|---|
| ride_height | 0.030 | 0.020 - 0.050 | meters |
| front_wing_angle | 14.0 | 10 - 20 | degrees |
| rear_wing_angle | 16.0 | 10 - 22 | degrees |
| diffuser_angle | 12.0 | 6 - 18 | degrees |
| sidepod_undercut | 0.15 | 0.08 - 0.20 | meters |

## Code Conventions

- Python dataclasses for structured data (`SimulationParams`, `SimulationResult`)
- Blender script uses `argparse` with `--` separator for Blender CLI arg passthrough
- All dimensions in SI units (meters, degrees, m/s, kg/m^3)
- FIA reference constants at module level (`CAR_LENGTH = 5.640`, `REFERENCE_AREA = 1.5`)
- OpenFOAM dict files in v2406 format with inline rationale comments
- Results stored as JSON arrays of `SimulationResult` objects

## What's NOT Implemented Yet

- **No CFD runs completed** - all 32 results in sweep_all.json are empirical estimates
- **OpenClaw AI orchestration** - referenced in README but does not exist in code
- **export_stl.py** - listed in README directory structure but not created
- **postprocessing/ directory** - listed in README but does not exist
- **docs/ directory** - listed in README but does not exist
- No ML/surrogate models (only empirical correlations in `estimate_coefficients()`)
- No tests, no CI/CD, no web UI
- No multi-variable interaction sweeps (only single-variable sweeps)
- Empirical model missing: Cd sensitivity to ride_height/diffuser/sidepod, wing stall behavior

## Related Docs

- [DESIGN.md](DESIGN.md) - Architecture decisions and aerodynamic rationale
- [EXPERIMENT.md](EXPERIMENT.md) - Parameter sweep results and experiment tracking
- [RESEARCH.md](RESEARCH.md) - Literature references and research directions
- [NEXT-TO-DO.md](NEXT-TO-DO.md) - Prioritized backlog
