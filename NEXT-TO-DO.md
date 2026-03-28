# Next Steps - Prioritized Backlog

## P1: Validate the Pipeline End-to-End
_Nothing else matters until the Blender -> OpenFOAM -> results chain actually runs._

### 1.1 Run Blender geometry generation
- Execute: `blender --background --python blender/f1_car_generator.py`
- Verify STL files created in `openfoam/f1_baseline/constant/triSurface/`
- Check mesh quality: watertight, no degenerate faces, correct scale (meters)
- Use `surfaceCheck` (OpenFOAM utility) on generated STLs

### 1.2 Run first CFD simulation
- Execute full OpenFOAM pipeline: blockMesh -> snappyHexMesh -> simpleFoam
- Monitor residual convergence (target: all < 1e-5)
- Verify y+ values on car surface (target: 30-50 for wall functions)
- Compare Cd/Cl against empirical estimates to calibrate confidence
- Document wall time and mesh cell count

### 1.3 Fix README accuracy
- Remove or create references to missing files: `export_stl.py`, `postprocessing/`, `docs/`
- Clarify OpenClaw is aspirational, not implemented
- Update directory structure to match reality

---

## P2: Improve Empirical Model

### 2.1 Fix Cd sensitivity gaps
- Add Cd sensitivity to ride_height (underbody flow restriction increases drag at low rh)
- Add Cd sensitivity to diffuser angle (flow separation increases drag above ~15 deg)
- Add Cd sensitivity to sidepod undercut

### 2.2 Add nonlinear effects
- Wing stall: front wing effectiveness should plateau/decrease above ~18-20 deg
- Diffuser separation: smoother transition around 15 deg threshold
- Fix ride height dead zone (rh=30mm and rh=35mm currently produce identical results)

### 2.3 Add cross-variable interactions
- ride_height x diffuser_angle (most critical - ride height directly affects diffuser inlet conditions)
- front_wing_angle x rear_wing_angle (aero balance coupling)

### 2.4 Calibrate against data
- Compare baseline against Ravelli & Savini reported values
- Cross-reference with DrivAerNet++ dataset distributions
- Add confidence intervals or uncertainty bounds

---

## P3: Testing and Robustness

### 3.1 Unit tests for empirical correlations
- Test edge cases: extreme parameter values, boundary conditions
- Test monotonicity: more wing angle should always increase |Cl| up to stall
- Test physical reasonableness of all outputs

### 3.2 Integration tests
- Test Blender script runs without errors in `--background` mode
- Test OpenFOAM case setup validity (`foamDictionary` checks)
- Test JSON output format from sweep results

### 3.3 Input validation
- Parameter bounds checking in SimulationParams
- Warn on physically unreasonable combinations
- Validate STL geometry before passing to OpenFOAM

---

## P4: Enhanced Parametric Design

### 4.1 Multi-variable sweeps
- Grid search over 2-3 variables simultaneously
- Identify interaction effects
- Generate response surface plots

### 4.2 Additional parametric variables
- Nose cone geometry (height, taper ratio)
- Floor edge detail (strake count, spacing)
- Rear wing DRS gap angle
- Beam wing configuration
- Brake duct openings

### 4.3 Geometry refinements
- Add suspension arms (significant drag source)
- Add mirrors and antenna
- Improve floor strake detail (currently simplified flat plates)
- Improve wheel geometry (currently simple cylinders)

---

## P5: ML Surrogate Model

### 5.1 Train surrogate on CFD data
- Need minimum ~50-100 CFD data points first
- Start with Gaussian Process Regression (small data regime)
- Compare against PINNs approach (R^2=0.968 from literature)
- Active learning to choose next CFD points efficiently

### 5.2 Optimization loop
- Bayesian optimization using surrogate model
- Multi-objective: maximize |Cl|/Cd subject to aero balance constraints
- Circuit-specific configs (high-downforce vs low-drag)

---

## P6: Infrastructure

### 6.1 Docker compose for reproducible CFD
- Package OpenFOAM + Blender in containers
- Parameterize via environment variables
- Enable cloud execution (AWS/GCP spot instances)

### 6.2 CI/CD
- GitHub Actions: lint, test, validate OpenFOAM dicts
- Automated STL generation check on PR

### 6.3 Visualization dashboard
- ParaView scripts for automated post-processing
- Matplotlib/Plotly for parameter sweep visualization
- Side-by-side configuration comparison

---

## Parking Lot (Long-term)
- OpenClaw AI orchestration layer
- FEM structural analysis (Elmer) for composite layup
- Transient simulation (DES/LES for unsteady aero)
- Wheel rotation modeling with rotating boundary conditions
- Thermal management (brake cooling, PU cooling ducts)
- Multi-car interaction (dirty air, DRS following distance)
- Lap simulation integration (aero maps -> lap time prediction)
