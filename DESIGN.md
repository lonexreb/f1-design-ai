# Design Decisions & Architecture

## Pipeline Flow

```
                                 +---> Omniverse Flow (GPU CFD) ---+
                                 |                                  |
Parameter --> Blender --> STL -->-+---> OpenFOAM (CPU CFD) --------+---> Force Extraction --> Results JSON
Selection     Geometry    Export  |                                  |
                  |              +---> Modulus PINN (~1ms) --------+
                  |              |                                  |
                  |              +---> ML Surrogate (GP/MLP) ------+
                  |              |                                  |
                  +--- (skip) ---+---> Empirical Estimate ---------+
```

### Multi-Backend Fallback Chain
The pipeline supports 5 simulation backends with automatic fallback:
1. **Omniverse Flow** - GPU-accelerated CFD (fastest physics-based)
2. **OpenFOAM** - CPU CFD with simpleFoam (validated, default)
3. **Modulus PINN** - Physics-informed neural network (~1ms, auto-trains on first use)
4. **ML Surrogate** - Best available trained model from ml/models/ (GP > MLP > PINN > Linear)
5. **Empirical** - Correlation-based estimates (always available, instant)

Each backend returns a compatible `SimulationResult`. If the chosen backend fails, the pipeline falls back to empirical estimates.

### Graceful Degradation
- Blender unavailable: skip geometry, use estimation backends
- OpenFOAM unavailable: use Docker wrapper, or fall to Modulus/surrogate/empirical
- Omniverse unavailable: fall to OpenFOAM or below
- No trained models: auto-train from existing results, or use empirical
- Nothing installed: empirical estimates still work

## Key Technical Decisions

### simpleFoam (steady RANS) over transient solvers
- **Chosen**: simpleFoam with SIMPLEC algorithm
- **Alternatives**: pimpleFoam (URANS), LES, DES
- **Rationale**: 10-100x cheaper than transient methods. Sufficient accuracy for design-space exploration (5-10% error on forces). Validated for F1 by Ravelli & Savini 2018.
- **Trade-off**: Cannot capture unsteady wake behavior or transient aero effects.

### k-omega SST turbulence model
- **Chosen**: kOmegaSST
- **Alternatives**: Spalart-Allmaras, k-epsilon, RSM
- **Rationale**: Best of both worlds (k-omega near walls, k-epsilon in freestream). Standard for external automotive/aerospace aero. Handles adverse pressure gradients better than k-epsilon.
- **Trade-off**: More expensive than SA (2 extra equations). Less accurate than RSM for highly 3D separated flows.

### Wall functions (y+ ~30-50) over resolved boundary layers
- **Chosen**: kqRWallFunction, omegaWallFunction, nutkWallFunction
- **Alternatives**: Low-Re wall treatment (y+ < 1)
- **Rationale**: Reduces cell count ~10x near walls. Acceptable for force prediction (5-10% error). Faster turnaround for parameter sweeps.
- **Trade-off**: Cannot resolve boundary layer transition or skin friction details.

### NACA-based airfoil profiles for wings
- **Chosen**: Modified NACA 4-digit thickness distribution
- **Alternatives**: Custom F1 profiles, Clark Y, NACA 6-series
- **Rationale**: Well-characterized, easy to parameterize. 12% thickness (front wing), 10% (rear main), 8% (DRS flap).
- **Trade-off**: Real F1 wings use custom multi-element profiles optimized for specific conditions.

### Single combined STL for OpenFOAM
- **Chosen**: Export all geometry as `f1_car_combined.stl`, single "car" patch
- **Alternatives**: Per-component STLs with separate patch names
- **Rationale**: Simpler snappyHexMesh configuration. Avoids inter-component gap/overlap issues. Sufficient for total force measurement.
- **Trade-off**: Cannot extract per-component forces (front wing Cl vs rear wing Cl separately).
- **Future**: Switch to multi-region STL when per-component analysis is needed.

### Empirical correlation fallback
- **Chosen**: Linear sensitivity model with independent per-variable factors
- **Rationale**: Immediate parameter exploration before CFD runs. Based on published data and physics intuition. Instantaneous computation.
- **Known limitations**: No cross-variable interactions. Missing Cd sensitivity for ride_height/diffuser/sidepod. No wing stall modeling. Linear where reality is nonlinear. See [EXPERIMENT.md](EXPERIMENT.md) for detailed analysis.

## ML Surrogate Architecture

### Why 4 model types?
Each serves a different purpose in the pipeline:

| Model | Why it exists | When to use |
|---|---|---|
| **Linear** | Fast baseline, interpretable. Polynomial features capture some nonlinearity. | First sanity check. Compare others against it. |
| **MLP** | Captures complex nonlinear relationships. PyTorch + CUDA for speed. | General-purpose surrogate when >50 data points available. |
| **GP** | Provides uncertainty estimates. Enables active learning (propose where to simulate next). | Active learning loop. Small datasets (32-200 points). |
| **PINN** | Physics constraints in loss function prevent unphysical predictions. | When model must respect Cd>0, Cl<0, range bounds. |

### GP for active learning
- **Chosen**: GPyTorch (CUDA) with sklearn fallback
- **Acquisition functions**: Max variance (explore uncertain regions) or UCB (balance exploit/explore)
- **Rationale**: With only 32 data points, each new CFD simulation is expensive. GP uncertainty tells us where the model is most wrong, so we simulate there next.
- **Trade-off**: GP scales O(n^3) with dataset size. Fine for <500 points, need sparse GP beyond that.

### Modulus PINN physics constraints
Six constraints enforced via physics loss in `scripts/modulus_surrogate.py`:
1. Cd must be positive
2. Cl must be negative (downforce)
3. Cd in realistic range [0.5, 1.5]
4. Cl in range [-6.0, -1.0]
5. L/D consistency: ld_predicted ≈ |cl|/cd
6. L/D in range [1.0, 7.0]

Physics loss weight: 0.1 (data loss weight: 1.0). Tuned to guide without overriding data.

### Autoresearch pattern (Karpathy's approach adapted)
- **Chosen**: Seed paper + experiment runner + manual loop (with aresearch CLI support)
- **Rationale**: Mirrors Karpathy's "one GPU, one file, one metric" philosophy. Human writes research directions, agent iterates on model architecture/hyperparameters.
- **Current limitation**: Manual loop runs only 7 fixed experiments (4 baseline models + 3 HP variations). True autoresearch needs LLM-driven hypothesis generation per iteration.
- **Future**: Connect to OpenClaw/NemoClaw for LLM-in-the-loop experiment design.

### config.yaml disconnection (known debt)
- `config.yaml` defines all parameters, hyperparameters, and paths centrally
- `ml/` modules use hard-coded values in `data_prep.py` (PARAM_BOUNDS) and `surrogate.py` (model architectures)
- These MUST be wired together. Currently dual sources of truth = drift risk.

## NVIDIA Omniverse Integration

### STL -> USD conversion pipeline
- `scripts/convert_to_usd.py` converts Blender-generated STL to USD format
- Supports both binary and ASCII STL with auto-detection
- Generates Omniverse-compatible scene with wind tunnel domain, boundary conditions, and material definitions
- Falls back to built-in USDA writer if pxr (OpenUSD) library not available

### Omniverse Flow backend
- `scripts/omniverse_sim.py` detects Omniverse installation across OS paths
- Three resolution presets: low (fast iteration), medium (default), high (validation)
- Attempts Flow API first, falls back to Kit CLI subprocess
- Returns same `SimulationResult` format as OpenFOAM backend

### Warp LBM (not implemented)
- Placeholder function in `omniverse_sim.py`. Returns None.
- Intended for Lattice Boltzmann Method on GPU via NVIDIA Warp
- Would complement Flow for different flow regimes

## Geometry Design

### Why Red Bull RB19/RB20 as baseline?
- Dominant car of 2023-2024 era
- Design philosophy well-documented: aggressive ground effect, tight sidepod packaging
- Represents state-of-the-art within FIA 2024 regulations

Specific features modeled:
- V-shaped monocoque cross-section (opens underbody volume)
- Full-width front wing (conditions flow for entire car)
- Shark-style sidepod inlet with aggressive undercut
- High-throat Venturi tunnel floor
- DRS-capable rear wing with separate flap element

### Simplifications vs Real F1 Car

| Feature | This Model | Real F1 | Impact |
|---|---|---|---|
| Front wing | Single element + endplates | 4-5 element cascade | Underestimates max Cl_front |
| Rear wing | Main plane + DRS flap | Main + flap + beam wing | Missing ~10% rear Cl |
| Wheels | Simple cylinders | Rotating with rim/tire profile | Missing wheel wake effects |
| Floor | Smooth + simple tunnels | Strakes, fences, edge vortex generators | Floor Cl underestimated |
| Suspension | Not modeled | Push/pull rod, wishbones | Missing ~5% total Cd |
| Bargeboards | Not modeled | Complex vortex management | Sidepod flow conditioning missing |
| Engine cooling | Not modeled | Internal flow through sidepods | Missing cooling drag |
| Surface detail | Smooth | VGs, Gurney flaps, serrations | Missing small-scale aero devices |

## OpenFOAM Configuration

### Domain Sizing
- Upstream: 30m (~5 car lengths) - standard for external aero
- Downstream: 60m (~10 car lengths) - captures wake recovery
- Lateral: 15m each side (~7.5 car widths) - blockage < 1%
- Height: 20m (~21 car heights) - prevents artificial acceleration

### Mesh Refinement Strategy
Five refinement regions by aerodynamic importance:
1. **Car surface** (level 5): Captures geometry detail
2. **Feature edges** (level 5): Sharp edges, trailing edges
3. **Near-car box** (level 3): -4m to 6m X, -2m to 2m Y - overall flow field
4. **Wake region** (level 2): Extended downstream
5. **Base mesh**: 60x20x15 cells (coarse structured)

Max global cells: 5 million.

### Boundary Conditions
- **Inlet**: Fixed velocity 83.33 m/s (300 km/h), turbulence intensity ~0.5%
- **Outlet**: Zero gradient for all fields
- **Car surface**: No-slip walls
- **Ground**: Moving wall at 83.33 m/s (simulates car motion relative to road)
- **Top/sides**: Symmetry planes

### Solver Settings
- SIMPLEC algorithm (consistent pressure correction, faster convergence)
- Relaxation: p=0.3, U=0.7, k=0.5, omega=0.5 (conservative for stability)
- 2nd-order velocity (linearUpwind), 1st-order turbulence (upwind for stability)
- Convergence: all residuals < 1e-5, max 2000 iterations
- Force monitoring: Cd, Cl, pressure coefficient, y+ every iteration
