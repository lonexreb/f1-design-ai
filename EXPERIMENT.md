# Experiment Tracking

## Baseline Configuration

| Parameter | Value | Rationale |
|---|---|---|
| ride_height | 30 mm | Mid-range ground effect; avoids floor stall below 20mm |
| front_wing_angle | 14 deg | Moderate downforce; conditions flow for underbody |
| rear_wing_angle | 16 deg | Balanced; higher for high-downforce circuits |
| diffuser_angle | 12 deg | Below separation threshold (~15 deg per literature) |
| sidepod_undercut | 150 mm | RB20-style aggressive undercut |

### Baseline Predicted Performance (Empirical)
- Cd = 0.95 (typical F1 range: 0.7-1.2)
- Cl = -3.50 (typical F1 range: -3.0 to -5.5)
- L/D = 3.68 (typical F1: 3.0-5.0)
- **Status**: Empirical estimate only. No CFD validation.

### Reference Values from Literature
- Ravelli & Savini 2018: Cd=0.961, Cl=-2.890 (simplified F1 geometry, 5.7% drag error vs wind tunnel)
- Typical 2024 F1 car: Cd ~0.9-1.1, Cl ~-3.5 to -5.0 (configuration dependent)

---

## Experiment 1: Single-Variable Sweeps (2026-03-28)

**Method**: All sweeps use `--estimate-only` (empirical correlations from `estimate_coefficients()`). Each sweep varies one parameter, holds others at baseline. Source: `results/sweep_all.json` (32 data points).

### 1A: Ride Height Sweep

| ride_height (mm) | Cd | Cl | L/D |
|---|---|---|---|
| 20 | 0.950 | -4.200 | 4.42 |
| 25 | 0.950 | -3.850 | 4.05 |
| 30 | 0.950 | -3.500 | 3.68 |
| 35 | 0.950 | -3.500 | 3.68 |
| 40 | 0.950 | -3.238 | 3.41 |
| 50 | 0.950 | -2.713 | 2.86 |

**Observations**:
- Cd does not change with ride height. Physically unrealistic - lower ride heights should increase drag from underbody flow restriction.
- Model shows rh=20mm as maximum |Cl| despite claiming floor stall (rh_factor=0.85). The stall penalty is applied multiplicatively but the base factor at 20mm still produces higher |Cl| than baseline. The stall model needs refinement.
- 30mm and 35mm produce identical results (both fall in the `rh < 35` branch where 30mm gives factor=1.0 and 35mm also gives factor=1.0).

### 1B: Front Wing Angle Sweep

| angle (deg) | Cd | Cl | L/D |
|---|---|---|---|
| 10 | 0.874 | -3.080 | 3.52 |
| 12 | 0.912 | -3.290 | 3.61 |
| 14 | 0.950 | -3.500 | 3.68 |
| 16 | 0.988 | -3.710 | 3.76 |
| 18 | 1.026 | -3.920 | 3.82 |
| 20 | 1.064 | -4.130 | 3.88 |

**Observations**:
- L/D monotonically increases with angle. No stall behavior modeled - real wings lose effectiveness above ~18-20 deg.
- Sensitivities are perfectly linear (Cd: +0.019/deg, Cl: -0.105/deg). Oversimplified but directionally correct.
- Front wing affects both Cd and Cl, which is correct behavior.

### 1C: Rear Wing Angle Sweep

| angle (deg) | Cd | Cl | L/D |
|---|---|---|---|
| 10 | 0.751 | -2.660 | 3.54 |
| 12 | 0.817 | -2.940 | 3.60 |
| 14 | 0.884 | -3.220 | 3.64 |
| 16 | 0.950 | -3.500 | 3.68 |
| 18 | 1.017 | -3.780 | 3.72 |
| 20 | 1.083 | -4.060 | 3.75 |
| 22 | 1.150 | -4.340 | 3.78 |

**Observations**:
- Rear wing has stronger Cd sensitivity (3.5%/deg) than front wing (2%/deg). Directionally correct - rear wing operates in higher-energy flow.
- L/D increases monotonically but at a decreasing rate. Better behavior than front wing model.
- No stall modeling here either.

### 1D: Diffuser Angle Sweep

| angle (deg) | Cd | Cl | L/D |
|---|---|---|---|
| 6 | 0.950 | -3.150 | 3.32 |
| 8 | 0.950 | -3.220 | 3.39 |
| 10 | 0.950 | -3.360 | 3.54 |
| 12 | 0.950 | -3.500 | 3.68 |
| 14 | 0.950 | -3.640 | 3.83 |
| 16 | 0.950 | -3.325 | 3.50 |
| 18 | 0.950 | -2.975 | 3.13 |

**Observations**:
- **Cd is completely insensitive to diffuser angle**. This is the most significant model flaw. Real diffusers increase drag substantially at high angles due to flow separation.
- The model correctly shows a performance peak around 14 deg and drop-off above 15 deg (separation).
- The transition at 15 deg is too abrupt (-0.05/deg penalty vs +0.02/deg benefit below threshold).
- Optimal diffuser angle per this model: 14 deg.

### 1E: Sidepod Undercut Sweep

| undercut (mm) | Cd | Cl | L/D |
|---|---|---|---|
| 80 | 0.950 | -3.378 | 3.56 |
| 100 | 0.950 | -3.413 | 3.59 |
| 120 | 0.950 | -3.448 | 3.63 |
| 150 | 0.950 | -3.500 | 3.68 |
| 180 | 0.950 | -3.553 | 3.74 |
| 200 | 0.950 | -3.588 | 3.78 |

**Observations**:
- Weakest sensitivity of all parameters. Cl ranges only -3.38 to -3.59 across the full sweep.
- No Cd effect. This may be approximately correct for small geometry changes but likely underestimates impact at extremes.
- More undercut = more downforce, which is directionally correct (better diffuser feed).

---

## Key Findings

1. **Ride height and rear wing angle** are the two most impactful variables for total downforce
2. **Front wing angle** has the best L/D improvement per degree (most efficient downforce source)
3. **Diffuser angle** has a clear optimum around 14 deg but the model has a critical Cd flaw
4. **Multi-variable interactions are not captured** (ride_height x diffuser is critically important in reality)
5. **All results need CFD validation** - empirical estimates provide directional guidance only

### Empirical Model Flaws Summary

| Issue | Severity | Impact |
|---|---|---|
| No Cd sensitivity to ride_height | High | Cannot optimize drag vs downforce trade-off for ground clearance |
| No Cd sensitivity to diffuser angle | High | Diffuser L/D optimization is unreliable |
| No wing stall modeling | Medium | Overpredicts performance at extreme wing angles |
| No cross-variable interactions | High | ride_height x diffuser interaction is dominant in real F1 |
| Linear sensitivities everywhere | Medium | Misses nonlinear behavior near design boundaries |
| rh=30mm and rh=35mm identical | Low | Dead zone in ride height model |

---

## Planned Experiments

### Next: CFD Validation of Baseline
- Run full pipeline for baseline configuration (30mm, 14/16 deg, 12 deg, 150mm)
- Compare Cd, Cl, L/D against empirical prediction (Cd=0.95, Cl=-3.5)
- Calibrate empirical model based on CFD delta
- Expected runtime: 2-4 hours on modern workstation

### Next: CFD at Extreme Points
- Min drag config: ride_height=30mm, fw=10 deg, rw=10 deg, diff=12 deg
- Max downforce config: ride_height=25mm, fw=20 deg, rw=22 deg, diff=14 deg
- Purpose: calibrate model at extremes, not just near baseline

### Future: Multi-Variable Grids
- 2D sweep: ride_height x diffuser_angle (known strong interaction)
- 2D sweep: front_wing_angle x rear_wing_angle (aero balance optimization)
- Requires CFD or improved empirical model with interaction terms
