#!/usr/bin/env python3
"""
Karpathy's Autoresearch - F1 Surrogate Model Research
======================================================
Sets up and runs Autoresearch to iteratively discover the best ML architecture
for predicting F1 aerodynamic coefficients from design parameters.

Autoresearch uses LLMs to:
1. Design experiments (model architecture, hyperparameters)
2. Run experiments (train + evaluate)
3. Analyze results and propose improvements
4. Iterate until convergence or budget exhaustion

Setup:
    python3 ml/autoresearch_config.py --setup

Run:
    python3 ml/autoresearch_config.py --run --iterations 10

Status:
    python3 ml/autoresearch_config.py --status
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
AUTORESEARCH_DIR = PROJECT_DIR / "autoresearch"
RESULTS_DIR = PROJECT_DIR / "results"
ML_DIR = PROJECT_DIR / "ml"

# Autoresearch seed document - describes the research problem
SEED_PAPER = """# Predicting F1 Aerodynamic Coefficients with ML Surrogates

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
"""

# Autoresearch run script
RUN_SCRIPT = """#!/usr/bin/env python3
\"\"\"Autoresearch experiment entry point.\"\"\"
import json
import sys
from pathlib import Path

project_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_dir))

from ml.experiment import run_experiment

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gp")
    parser.add_argument("--data", default=str(project_dir / "results" / "sweep_all.json"))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", default="64,64,32")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--output", default="autoresearch/latest_result.json")
    args = parser.parse_args()

    kwargs = {}
    if args.model == "mlp":
        kwargs = {"lr": args.lr, "epochs": args.epochs,
                  "hidden_layers": [int(x) for x in args.hidden.split(",")]}
    elif args.model == "linear":
        kwargs = {"degree": args.degree}
    elif args.model == "pinn":
        kwargs = {"epochs": args.epochs, "physics_weight": args.physics_weight}

    result = run_experiment(args.model, args.data, model_kwargs=kwargs)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"Result saved: {args.output}")

    # Print score for Autoresearch to parse
    te = result.get("test_eval", {})
    score = (te.get("r2_cd", 0) + te.get("r2_cl", 0) + te.get("r2_ld_ratio", 0)) / 3
    print(f"SCORE: {score:.4f}")

if __name__ == "__main__":
    main()
"""


def setup_autoresearch():
    """Create Autoresearch directory structure and seed files."""
    print("=" * 60)
    print("  Setting Up Karpathy's Autoresearch")
    print("=" * 60)

    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    # Write seed paper
    seed_path = AUTORESEARCH_DIR / "seed_paper.md"
    seed_path.write_text(SEED_PAPER)
    print(f"  Created: {seed_path}")

    # Write run script
    run_path = AUTORESEARCH_DIR / "run.py"
    run_path.write_text(RUN_SCRIPT)
    print(f"  Created: {run_path}")

    # Create experiment log
    log_path = AUTORESEARCH_DIR / "experiment_log.json"
    if not log_path.exists():
        log_path.write_text("[]")
    print(f"  Created: {log_path}")

    # Check if aresearch is installed
    try:
        result = subprocess.run(
            ["aresearch", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"  Autoresearch: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  Autoresearch: NOT INSTALLED")
        print("  Install: pip install aresearch")
        print("  Or: pip install -r requirements.txt")

    print(f"\n  Setup complete. Directory: {AUTORESEARCH_DIR}")
    print("\n  To run Autoresearch:")
    print("    aresearch run --seed autoresearch/seed_paper.md \\")
    print("      --script 'python3 autoresearch/run.py' \\")
    print("      --iterations 10")

    return True


def run_autoresearch(iterations: int = 10):
    """Execute Autoresearch research loop."""
    print("=" * 60)
    print(f"  Running Autoresearch ({iterations} iterations)")
    print("=" * 60)

    # Try direct aresearch CLI
    cmd = [
        "aresearch", "run",
        "--seed", str(AUTORESEARCH_DIR / "seed_paper.md"),
        "--script", f"python3 {AUTORESEARCH_DIR / 'run.py'}",
        "--iterations", str(iterations),
        "--output-dir", str(AUTORESEARCH_DIR / "runs"),
    ]

    try:
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR), timeout=3600 * 2)
        return result.returncode == 0
    except FileNotFoundError:
        print("  Autoresearch CLI not found. Running manual experiment loop...")
        return _manual_research_loop(iterations)
    except subprocess.TimeoutExpired:
        print("  Autoresearch timed out after 2 hours")
        return False


def _manual_research_loop(iterations: int) -> bool:
    """
    Fallback: Run a manual experiment loop when aresearch CLI isn't available.
    Tests all model types and hyperparameter variations.
    """
    sys.path.insert(0, str(PROJECT_DIR))
    from ml.experiment import run_experiment, compare_all_models

    data_path = str(RESULTS_DIR / "sweep_all.json")

    # First: compare all baseline models
    print("\n  Phase 1: Baseline model comparison")
    results = compare_all_models(data_path)

    # Log results
    log_path = AUTORESEARCH_DIR / "experiment_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.extend(results if isinstance(results, list) else [results])

    # Phase 2: Hyperparameter search for best model
    print("\n  Phase 2: Hyperparameter search")
    hp_experiments = [
        ("mlp", {"hidden_layers": [32, 32], "lr": 0.001, "epochs": 300}),
        ("mlp", {"hidden_layers": [128, 128, 64], "lr": 0.0005, "epochs": 800}),
        ("mlp", {"hidden_layers": [64, 64, 64, 32], "lr": 0.001, "epochs": 500}),
        ("linear", {"degree": 2, "alpha": 0.1}),
        ("linear", {"degree": 3, "alpha": 1.0}),
        ("pinn", {"epochs": 1000, "physics_weight": 0.05}),
        ("pinn", {"epochs": 2000, "physics_weight": 0.2}),
    ]

    for model_name, kwargs in hp_experiments[:iterations]:
        try:
            result = run_experiment(model_name, data_path, model_kwargs=kwargs)
            log.append(result)
        except Exception as e:
            print(f"  SKIP {model_name} {kwargs}: {e}")
            log.append({"model": model_name, "kwargs": kwargs, "error": str(e), "error_type": type(e).__name__})

    # Save experiment log
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"\n  Experiment log: {log_path} ({len(log)} experiments)")

    # Find best model
    best = None
    best_score = -float("inf")
    for r in log:
        if "error" in r or "test_eval" not in r:
            continue
        te = r["test_eval"]
        score = (te.get("r2_cd", 0) + te.get("r2_cl", 0) + te.get("r2_ld_ratio", 0)) / 3
        if score > best_score:
            best_score = score
            best = r

    if best:
        print(f"\n  Best model: {best['model']}")
        print(f"  Score: {best_score:.4f} (avg R²)")
        print(f"  Saved: {best.get('model_path', 'N/A')}")

    return True


def show_status():
    """Show Autoresearch experiment status."""
    print("=" * 60)
    print("  Autoresearch Status")
    print("=" * 60)

    log_path = AUTORESEARCH_DIR / "experiment_log.json"
    if not log_path.exists():
        print("  No experiments run yet. Run --setup first.")
        return

    log = json.loads(log_path.read_text())
    print(f"  Total experiments: {len(log)}")

    # Find best
    best = None
    best_score = -float("inf")
    for r in log:
        if "error" in r or "test_eval" not in r:
            continue
        te = r["test_eval"]
        score = (te.get("r2_cd", 0) + te.get("r2_cl", 0) + te.get("r2_ld_ratio", 0)) / 3
        if score > best_score:
            best_score = score
            best = r

    if best:
        te = best["test_eval"]
        print(f"\n  Best model: {best['model']}")
        print(f"  Avg R²: {best_score:.4f}")
        print(f"    R² Cd: {te['r2_cd']:.4f}")
        print(f"    R² Cl: {te['r2_cl']:.4f}")
        print(f"    R² L/D: {te['r2_ld_ratio']:.4f}")
        print(f"    Physics violations: {te['physics_violations']}")

    # Models per type
    from collections import Counter
    counts = Counter(r.get("model", "unknown") for r in log)
    print(f"\n  Experiments by model: {dict(counts)}")


def main():
    parser = argparse.ArgumentParser(description="Karpathy's Autoresearch - F1 ML")
    parser.add_argument("--setup", action="store_true", help="Initialize Autoresearch")
    parser.add_argument("--run", action="store_true", help="Run research loop")
    parser.add_argument("--status", action="store_true", help="Show experiment status")
    parser.add_argument("--iterations", type=int, default=10, help="Research iterations")
    args = parser.parse_args()

    if args.setup:
        setup_autoresearch()
    elif args.run:
        run_autoresearch(args.iterations)
    elif args.status:
        show_status()
    else:
        # Default: show status, suggest next steps
        if not AUTORESEARCH_DIR.exists():
            print("  Autoresearch not initialized. Run: python3 ml/autoresearch_config.py --setup")
        else:
            show_status()


if __name__ == "__main__":
    main()
