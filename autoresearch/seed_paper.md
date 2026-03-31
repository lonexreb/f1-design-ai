# Predicting F1 Aerodynamic Coefficients with ML Surrogates

## Research Question
Can we train a machine learning model to accurately predict F1 car aerodynamic
coefficients (drag coefficient Cd, lift coefficient Cl, and lift-to-drag ratio L/D)
from 5 design parameters, using limited training data (32-100 points)?

## Background
Full CFD simulation of an F1 car takes 2-4 hours per configuration using OpenFOAM
(simpleFoam, k-omega SST, ~5M cells). A fast surrogate model would enable:
- Real-time parameter exploration and optimization
- Bayesian optimization with uncertainty quantification
- Circuit-specific aero configuration tuning

## Input Features (5 parameters)
| Parameter | Range | Unit | Aerodynamic Effect |
|-----------|-------|------|-------------------|
| ride_height | 0.020 - 0.050 | meters | Ground effect (Venturi tunnels, 40-60% of downforce) |
| front_wing_angle | 10 - 20 | degrees | Front downforce, wake quality |
| rear_wing_angle | 10 - 22 | degrees | Rear downforce, primary drag source |
| diffuser_angle | 6 - 18 | degrees | Diffuser expansion, flow separation risk |
| sidepod_undercut | 0.08 - 0.20 | meters | Underbody flow, cooling drag |

## Target Outputs (3 values)
- **Cd** (drag coefficient): 0.7 - 1.2 range, must be > 0
- **Cl** (lift coefficient): -3.0 to -5.5 range, must be < 0 (negative = downforce)
- **L/D** (lift-to-drag ratio): 3.0 - 5.0 range, efficiency metric

## Available Data
- 32 empirical estimate data points (parameter sweeps, no CFD validation)
- Data format: JSON with params dict and Cd/Cl/L/D values
- Located at: results/sweep_all.json

## Baseline Models to Try
1. **Linear** - Polynomial regression (degree 2) with Ridge regularization
2. **MLP** - Multi-layer perceptron [64, 64, 32] with ReLU, dropout 0.1
3. **GP** - Gaussian Process with Matérn 2.5 kernel (provides uncertainty)
4. **PINN** - Physics-informed neural network with aerodynamic constraints

## Evaluation Metrics
- R² for each target (Cd, Cl, L/D) on held-out test set (20%)
- Total MSE across all targets
- Physics violations: count of unphysical predictions (Cd < 0, Cl > 0)

## Key Challenges
- **Very small dataset** (32 points) — regularization and data efficiency critical
- **Nonlinear interactions** — ride_height × diffuser coupling is dominant
- **Physical constraints** — predictions must be aerodynamically plausible
- **Multi-output** — Cd, Cl, L/D are correlated (L/D = |Cl|/Cd)

## Experiment Entry Point
```bash
python3 ml/experiment.py --model <model_name> --data results/sweep_all.json
```

## Research Directions to Explore
1. Feature engineering: polynomial interactions, physics-based features
2. Architecture search: layer sizes, activation functions, normalization
3. Training strategy: learning rate schedules, early stopping, ensemble methods
4. Physics-informed losses: Navier-Stokes residuals, consistency constraints
5. Data augmentation: synthetic points from empirical model uncertainty
6. Transfer learning: pre-train on DrivAerNet++ automotive data, fine-tune on F1
