# Research References & Directions

> **Last reviewed**: 2026-03-31 | **Data status**: 32 empirical points, 0 CFD runs | **Best surrogate**: PINN (test MSE=0.0038, R² 0.85-0.92 across all targets)

## Core References

### Ravelli & Savini (2018) - "Aerodynamic Simulation of a 2017 F1 Car with Open-Source CFD"
- **Relevance**: Primary methodological reference for OpenFOAM setup
- **Key findings**: Validated simpleFoam + k-omega SST against wind tunnel data. 5.7% error on Cd (sim=0.961 vs wt=0.909). Wall functions (y+ ~30-50) sufficient for engineering accuracy.
- **What we adopted**: simpleFoam solver, k-omega SST, wall functions, snappyHexMesh workflow
- **Limitations**: 2017 regulations (pre-ground-effect era), simplified geometry

### DrivAerNet++ (NeurIPS 2024)
- **Relevance**: Largest automotive CFD dataset for ML surrogate pre-training
- **Key findings**: 8,150 3D car designs with full CFD results. ML predicts Cd from geometry accurately. Publicly available.
- **Potential use**: Pre-train surrogate, fine-tune on F1-specific data
- **Limitations**: Generic automotive (not F1), different Re regime, no ground effect

### Physics-Informed Neural Networks for F1 Front Wings
- **Relevance**: Demonstrates PINNs can replace CFD for component-level analysis
- **Key findings**: R^2=0.968 for front wing pressure prediction. Orders of magnitude faster than CFD.
- **What we adopted**: PINN architecture in `scripts/modulus_surrogate.py` with physics loss constraints
- **Limitations**: Single-component (wing in isolation), no full-car coupling

### AI Design Agents (ArXiv 2503.23315, 2025)
- **Relevance**: Multi-agent framework for automotive aerodynamic design
- **Key findings**: Agents automate sketching, styling, 3D modeling, CFD meshing, and simulation. Reduces design cycles from weeks to minutes. Uses VLMs + LLMs + geometric deep learning.
- **Status**: Architectural inspiration for the OpenClaw orchestration concept

---

## Agentic Research Frameworks

### Karpathy's Autoresearch (2026-03-07)
- **What it is**: 630-line Python script enabling AI agents to conduct ML research autonomously. "One GPU, one file, one metric."
- **Core pattern**: Agent reads `program.md` (human research directions) -> modifies `train.py` -> trains for 5 min fixed budget -> evaluates val_bpb -> keeps/reverts -> loops
- **Results**: 700 experiments in 2 days, 20 additive improvements, 11% efficiency gain on "Time to GPT-2". ~12 experiments/hour per GPU.
- **Our adaptation** (`ml/autoresearch_config.py`):
  - `SEED_PAPER` = problem description (5 params -> 3 aero targets, 4 model types)
  - `RUN_SCRIPT` = experiment runner accepting --model, --lr, --epochs, etc.
  - `_manual_research_loop()` = baseline comparison + 7 HP experiments (limited)
  - Falls back to manual loop if `aresearch` CLI not installed
- **Gap**: Current implementation runs 7 fixed experiments. True autoresearch needs LLM-driven hypothesis generation per loop iteration.
- **Source**: github.com/karpathy/autoresearch

### NVIDIA NemoClaw / OpenClaw (GTC 2026-03-16)
- **What it is**: NemoClaw is NVIDIA's security-hardened wrapper around OpenClaw, the open-source autonomous AI agent platform.
- **Architecture**: Plugin (CLI) -> Blueprint (orchestration) -> Sandbox (OpenShell isolation) -> Inference (provider-routed LLM calls)
- **Key capability**: Agent can read/write files, execute shell commands, control applications - all within sandboxed environment with declarative YAML policies.
- **Relevance**: Provides the execution environment for autonomous F1 design iteration. The agent can orchestrate Blender -> OpenFOAM/Omniverse -> result analysis -> parameter adjustment loops.
- **Status in this project**: Referenced in README as "OpenClaw (AI Assistant)" but not implemented. Tool manifest and sandbox policy needed.
- **Source**: github.com/NVIDIA/NemoClaw, open-claw.org

### The Autoresearch + OpenClaw Convergence
The key insight connecting these frameworks to this project:

| Autoresearch Concept | F1 Aero Mapping |
|---|---|
| `program.md` (human directions) | `research_directions.md` (aero goals, constraints) |
| `train.py` (what agent modifies) | `SimulationParams` (design variables) |
| `val_bpb` (single metric) | L/D ratio, \|Cl\|, or circuit-specific metric |
| 5 min fixed training budget | Empirical: instant / CFD: configurable |
| Git diff tracking | `results/experiments.jsonl` |
| One GPU | One simulation (or simulation batch) |

OpenClaw provides the sandboxed execution environment; autoresearch provides the experiment loop pattern.

---

## NVIDIA Simulation Stack

### NVIDIA Omniverse Flow
- **What**: GPU-accelerated CFD simulation within Omniverse platform
- **Integration**: `scripts/omniverse_sim.py` detects Omniverse installation, converts STL->USD, runs Flow simulation
- **Advantage**: Orders of magnitude faster than CPU OpenFOAM for iterative design
- **Status**: Backend implemented, requires Omniverse installation with Flow extension

### NVIDIA Modulus
- **What**: Framework for building physics-informed neural network surrogates
- **Integration**: `scripts/modulus_surrogate.py` implements F1AeroNet (4-layer MLP with physics loss)
- **Physics constraints enforced**: Cd>0, Cl<0, Cd in [0.5, 1.5], Cl in [-6.0, -1.0], L/D consistency
- **Performance**: ~1ms inference, auto-trains from sweep_all.json if no model exists
- **Status**: Fully implemented, falls back to plain PyTorch if nvidia-modulus not installed

### NVIDIA Warp
- **What**: Lattice Boltzmann Method (LBM) solver on GPU
- **Status**: Placeholder in `omniverse_sim.py` (returns None). Not implemented.

---

## F1 Aerodynamic Fundamentals

### Ground Effect (Primary Downforce Mechanism)
- Venturi tunnels under floor create low pressure -> downforce without wing drag penalty
- Floor generates 40-60% of total downforce on modern F1 cars
- Highly sensitive to ride height: stalls below ~15-20mm, optimal around 25-35mm
- Interacts strongly with diffuser angle and front wing wake

### Key Aerodynamic Ranges (2024 F1)
- Cd: 0.7-1.2 (Monza low-drag to Monaco high-downforce)
- Cl: -3.0 to -5.5 (negative = downforce)
- L/D ratio: 3.0-5.0
- Aero balance: 44-48% front
- Reynolds number: ~30M at 300 km/h over 5.64m

### FIA 2024 Technical Regulation Key Dimensions
- Car length: 5.640m, width: 2.000m, wheelbase: ~3.600m
- 18-inch wheels (720mm diameter)
- Front wing: full car width. Rear wing: 950mm max span.

---

## Research Gaps & Directions

### Near-term: Wire config.yaml + fix autoresearch
- `config.yaml` defines all parameters/hyperparameters but ml/ modules ignore it (hard-coded values)
- Autoresearch manual loop limited to 7 experiments; needs LLM-driven hypothesis generation
- Active learning and autoresearch are disconnected workflows

### Medium-term: Real CFD data + improved surrogates
- Run 50-100 CFD simulations (OpenFOAM or Omniverse) to build real training data
- Gaussian Process with active learning to choose highest-value next simulation
- Cross-variable interaction terms (ride_height x diffuser_angle is dominant)
- Aero balance modeling (front vs rear downforce split)

### Medium-term: Rewrite README.md
- README.md is significantly outdated (see DESIGN.md "README Accuracy" section)
- Should reflect the actual multi-backend pipeline, ML surrogate stack, and documentation structure
- Use CLAUDE.md as source of truth for directory structure and commands

### Long-term: Autonomous design agent (OpenClaw)
- OpenClaw/NemoClaw sandbox with F1 tool manifest
- Agent reads `research_directions.md`, proposes experiments, runs simulations, evaluates, loops
- Circuit-specific optimization profiles (Monaco vs Monza)
- Integration with structural analysis for weight/stiffness constraints

---

## Codebase Health (2026-03-31)

| Metric | Value |
|---|---|
| Core pipeline code | ~1,850 lines Python |
| ML pipeline code | ~1,700 lines Python |
| Configuration | 149 lines YAML |
| Documentation | ~5,500 lines across 5 Markdown files |
| Total tracked codebase | ~9,200 lines |
| Empirical data points | 32 (sweep_all.json) |
| CFD-validated data points | 0 |
| Trained ML models | 4 (PINN 179KB, MLP 48KB, GP 13KB, Linear 2KB — all trained on empirical data) |
| Autoresearch experiments | 11 completed (4 baseline + 3 MLP HP + 2 Linear HP + 2 PINN HP) |
| Best test MSE | 0.0038 (PINN, physics_weight=0.05) |
| Physics violations | 0 across all 11 experiments |
| Test coverage | 0% (no tests exist) |
| Git commits | ~10 (initial + Omniverse/ML integration + review fixes) |
