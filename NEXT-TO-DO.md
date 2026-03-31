# Next Steps - Prioritized Backlog

## P1: Validate Pipeline End-to-End
_Nothing else matters until the Blender -> simulation -> results chain actually runs._

### 1.1 Run Blender geometry generation
- Execute: `blender --background --python blender/f1_car_generator.py`
- Verify STL files created in `openfoam/f1_baseline/constant/triSurface/`
- Check mesh quality: watertight, no degenerate faces, correct scale (meters)

### 1.2 Run first real simulation
- Try OpenFOAM: `python3 scripts/run_pipeline.py --backend openfoam`
- Or Omniverse: `python3 scripts/run_pipeline.py --backend omniverse`
- Compare Cd/Cl against empirical estimates (Cd=0.95, Cl=-3.5)
- Document wall time and mesh cell count

### ~~1.3 Fix README accuracy~~ DONE (2026-03-31)
- ~~Remove references to non-existent files~~ README fully rewritten
- ~~Clarify OpenClaw is aspirational~~ Removed from README architecture
- ~~Update directory structure~~ Now matches actual project layout

---

## ~~P2: Wire config.yaml to ML Modules~~ DONE (2026-03-31)
_~~Central config exists but ml/ modules ignore it. Dual sources of truth = drift risk.~~_

**Implementation**: Added `ml/config.py` — a Config loader with caching and typed access to all config.yaml sections.

### ~~2.1 Make ml/data_prep.py read from config.yaml~~ DONE
- `PARAM_NAMES` and `PARAM_BOUNDS` now loaded from config.yaml via `ml.config.load_config()`
- `train_test_split` and `random_seed` defaults from config

### ~~2.2 Make ml/surrogate.py read hyperparameters from config.yaml~~ DONE
- MLP: hidden_layers, dropout, learning_rate, epochs from config
- GP: kernel, n_restarts from config
- PINN: epochs, physics_loss_weight from config
- All accept explicit kwargs that override config defaults

### ~~2.3 Make ml/autoresearch_config.py use config.yaml~~ DONE
- Reads `autoresearch.iterations`, `autoresearch.models_to_try`
- Reads `autoresearch.seed_question` and injects into seed paper

---

## P3: Enhance Autoresearch Loop
_Current manual loop runs 7 fixed experiments. Karpathy's pattern needs LLM-driven hypothesis generation._

### 3.1 Add LLM-driven hypothesis generation
- After each experiment, send results + seed paper to LLM (Claude API or local Ollama)
- LLM proposes next experiment: model type, hyperparameters, rationale
- Parse LLM response into `run_experiment()` call
- Track hypothesis -> result -> next hypothesis chain

### 3.2 Add `research_directions.md`
- Human-written guidance file (equivalent of Karpathy's `program.md`)
- Specifies: current objective, constraints, what to explore, what to avoid
- Agent reads every iteration (can be updated mid-run)

### 3.3 Scale experiment count (partially done)
- ~~Replace 7 fixed experiments with configurable `--iterations N`~~ 11 experiments ran via manual loop
- Add time budget per experiment (like autoresearch's 5-minute fixed budget)
- Add early stopping when improvements plateau
- **Result**: PINN is best model (test MSE=0.0038). See EXPERIMENT.md for full results.

---

## P4: Unify Active Learning + Autoresearch
_Currently parallel workflows. Should be one loop: "improve the model AND choose what to simulate next."_

### 4.1 Integrated loop design
- Autoresearch improves the surrogate model (architecture, hyperparameters)
- Active learning uses the improved GP to propose next simulation points
- New simulation data retrains the model
- Single loop: train model -> propose simulation -> run simulation -> retrain

### 4.2 Experiment genealogy
- Track parent-child relationships between experiments
- "Experiment 47 used the GP model from experiment 32, with 3 new CFD data points"

---

## P5: OpenClaw/NemoClaw Integration
_The agent execution environment for fully autonomous design iteration._

### 5.1 Create tool manifest
- `openclaw/tools.json` exposing: `run_simulation`, `get_current_best`, `propose_experiment`, `get_experiment_history`
- Each tool maps to existing Python functions

### 5.2 Create sandbox policy
- `openclaw/sandbox-policy.yaml` for NemoClaw
- Allow: filesystem access to project dir, shell execution for Blender/OpenFOAM, network for LLM inference
- Block: everything else

### 5.3 Create research_directions.md as agent system prompt
- Agent reads this to understand optimization goals
- Circuit-specific variants: `research_directions_monaco.md`, `research_directions_monza.md`

### 5.4 End-to-end test
- Install NemoClaw, configure sandbox
- Agent autonomously: reads directions -> proposes params -> runs simulation -> evaluates -> iterates

---

## P6: Improve Empirical Model
_The empirical fallback has known flaws (see EXPERIMENT.md)._

### 6.1 Fix Cd sensitivity gaps
- Add Cd response to ride_height, diffuser_angle, sidepod_undercut
- Fix ride height dead zone (rh=30mm and rh=35mm produce identical results)

### 6.2 Add nonlinear effects
- Wing stall above ~18-20 deg
- Smoother diffuser separation transition around 15 deg

### 6.3 Add cross-variable interactions
- ride_height x diffuser_angle (most critical)
- front_wing_angle x rear_wing_angle (aero balance)

---

## P7: Testing and Infrastructure

### 7.1 Unit tests for ML pipeline
- Test each surrogate model trains and predicts without error
- Test data_prep normalization/denormalization round-trips
- Test physics violation detection

### 7.2 Integration tests
- Test Blender script in --background mode
- Test OpenFOAM case validity
- Test each backend produces valid SimulationResult

### 7.3 CI/CD
- GitHub Actions: lint, test, validate OpenFOAM dicts
- Automated model training check on PR

### 7.4 Visualization
- ParaView scripts for automated post-processing
- Parameter sweep response surface plots
- Model comparison charts

---

## Parking Lot (Long-term)
- FEM structural analysis (Elmer) for composite layup
- Transient simulation (DES/LES for unsteady aero)
- Wheel rotation modeling with rotating boundary conditions
- Thermal management (brake cooling, PU cooling ducts)
- Multi-car interaction (dirty air, DRS following)
- Lap simulation integration (aero maps -> lap time prediction)
- Sparse GP or neural process for >500 data points
- DrivAerNet++ pre-training for transfer learning
