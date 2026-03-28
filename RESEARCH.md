# Research References & Directions

## Core References

### Ravelli & Savini (2018) - "Aerodynamic Simulation of a 2017 F1 Car with Open-Source CFD"
- **Relevance**: Primary methodological reference for this project
- **Key findings**: Validated OpenFOAM (simpleFoam + k-omega SST) against wind tunnel data. 5.7% error on drag coefficient (Cd_sim=0.961 vs Cd_wt=0.909). Wall-function approach (y+ ~30-50) sufficient for engineering accuracy. snappyHexMesh workflow validated for F1 geometry.
- **What we adopted**: simpleFoam solver, k-omega SST, wall functions, snappyHexMesh workflow, mesh refinement strategy
- **Limitations**: 2017 regulations (pre-ground-effect era), simplified geometry without wheels or suspension

### DrivAerNet++ (NeurIPS 2024)
- **Relevance**: Largest automotive CFD dataset, potential for ML surrogate training
- **Key findings**: 8,150 3D car designs with full CFD results. Demonstrated ML can predict Cd from geometry with good accuracy. Dataset publicly available.
- **Potential use**: Pre-train surrogate model on automotive data, then fine-tune on F1-specific simulations
- **Limitations**: Generic automotive (sedans, SUVs), not F1-specific. Different Reynolds number regime. No ground effect modeling.

### Physics-Informed Neural Networks for F1 Front Wings
- **Relevance**: Demonstrates PINNs can replace CFD for component-level analysis
- **Key findings**: R^2 = 0.968 for front wing pressure prediction. Orders of magnitude faster than full CFD. Requires physics constraints (Navier-Stokes residuals) in loss function.
- **Potential use**: Component-level surrogate for rapid front/rear wing angle optimization
- **Limitations**: Single-component analysis (wing in isolation). Does not capture wing-body interaction or ground effect coupling.

### AI Design Agents (2025) - Multi-Agent Blender+OpenFOAM Framework
- **Relevance**: Architectural reference for the "OpenClaw" orchestration concept
- **Key findings**: Multi-agent system with researcher + designer + simulator + analyzer roles. Automated Blender + OpenFOAM workflow via LLM agents. Demonstrated autonomous design iteration loop.
- **Potential use**: Blueprint for implementing the OpenClaw AI orchestration layer
- **Status in this project**: Aspirational. Referenced in README but not implemented.

---

## F1 Aerodynamic Fundamentals

### Ground Effect (Primary Downforce Mechanism)
- Venturi tunnels under floor create low pressure region -> downforce without drag penalty of wings
- Modern F1 floor generates 40-60% of total downforce
- Highly sensitive to ride height: stalls below ~15-20mm gap, optimal around 25-35mm
- Interacts strongly with diffuser angle and front wing wake quality
- Floor edge sealing (via vortices from edge fences) is critical for performance

### Key Aerodynamic Ranges for 2024 F1 Cars
- Cd: 0.7-1.2 (low-drag Monza trim vs high-downforce Monaco)
- Cl: -3.0 to -5.5 (negative = downforce)
- L/D ratio: 3.0-5.0 (aerodynamic efficiency)
- Aero balance: typically 45-47% front (percentage of total downforce on front axle)
- Reynolds number: ~30M at 300 km/h over 5.64m car length

### FIA 2024 Technical Regulation Key Dimensions
- Car length: 5.640m max
- Car width: 2.000m max
- Wheelbase: ~3.600m (typical)
- 18-inch wheels (720mm diameter)
- Floor reference plane width: 1.600m
- Front wing: full car width allowed
- Rear wing: 950mm max span

---

## Research Gaps & Directions

### Near-term: Empirical Model Improvement
The current empirical model (`estimate_coefficients()` in run_pipeline.py) uses independent linear sensitivities per variable. Critical gaps:
- Cross-variable interaction terms, especially ride_height x diffuser_angle
- Nonlinear effects: wing stall, diffuser separation
- Cd sensitivity to all parameters (currently missing for ride_height, diffuser, sidepod)
- Aero balance modeling (front vs rear downforce split)

**Approach**: Run CFD at 50-100 design points, fit response surface model with interaction terms.

### Medium-term: ML Surrogate Models
- **Gaussian Process Regression**: Works well with small datasets (50-200 CFD points). Provides uncertainty estimates for active learning.
- **Graph Neural Networks on mesh**: Per DrivAerNet++ methodology. Requires larger dataset.
- **PINNs for components**: Front wing and diffuser optimization. R^2=0.968 demonstrated in literature.
- **Active learning**: Use GP uncertainty to choose which CFD points maximize model improvement per compute dollar.

### Long-term: Autonomous Design Agent (OpenClaw)
- Implement multi-agent orchestration layer
- Agent decides what to simulate based on optimization objectives and model uncertainty
- Circuit-specific optimization (Monaco high-downforce vs Monza low-drag)
- Real-time aerodynamic map generation for lap simulation input
- Integration with structural analysis (Elmer FEM) for weight/stiffness constraints
