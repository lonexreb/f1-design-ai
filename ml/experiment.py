#!/usr/bin/env python3
"""
Experiment Runner for ML Surrogate Models
==========================================
Trains, evaluates, and compares surrogate models. Designed to be called by
Karpathy's Autoresearch for automated experiment iteration.

Usage:
    # Run single experiment
    python3 ml/experiment.py --model mlp --data results/sweep_all.json

    # Compare all models
    python3 ml/experiment.py --compare --data results/sweep_all.json

    # Called by Autoresearch
    python3 ml/experiment.py --model gp --lr 0.01 --epochs 300 --output results/experiment.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from ml.data_prep import prepare_dataset, destandardize_targets, TARGET_NAMES
from ml.surrogate import create_model, MODELS


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                    target_stats: dict) -> dict:
    """
    Compute evaluation metrics on destandardized predictions.

    Returns dict with per-target MSE, MAE, R², and physics violation counts.
    """
    # Destandardize for meaningful metrics
    Y_true_raw = destandardize_targets(Y_true, target_stats)
    Y_pred_raw = destandardize_targets(Y_pred, target_stats)

    metrics = {}

    for i, name in enumerate(TARGET_NAMES):
        true = Y_true_raw[:, i]
        pred = Y_pred_raw[:, i]

        mse = float(np.mean((true - pred) ** 2))
        mae = float(np.mean(np.abs(true - pred)))

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        metrics[f"mse_{name}"] = mse
        metrics[f"mae_{name}"] = mae
        metrics[f"r2_{name}"] = r2

    # Physics violation checks (on raw predictions)
    cd_pred = Y_pred_raw[:, 0]
    cl_pred = Y_pred_raw[:, 1]

    violations = 0
    violations += int(np.sum(cd_pred < 0))        # Cd must be positive
    violations += int(np.sum(cl_pred > 0))         # Cl must be negative (downforce)
    violations += int(np.sum(cd_pred > 2.0))       # Cd unrealistically high
    violations += int(np.sum(cl_pred < -8.0))      # Cl unrealistically low

    metrics["physics_violations"] = violations
    metrics["total_mse"] = float(np.mean((Y_true_raw - Y_pred_raw) ** 2))

    return metrics


def run_experiment(model_name: str, data_path: str,
                   model_kwargs: Optional[dict] = None,
                   save_model: bool = True) -> dict:
    """
    Run a complete training + evaluation experiment.

    Args:
        model_name: Name from MODELS registry
        data_path: Path to results JSON
        model_kwargs: Optional hyperparameters for model
        save_model: Whether to save the trained model

    Returns:
        Complete experiment results dict
    """

    print(f"\n{'='*60}")
    print(f"  Experiment: {model_name}")
    print(f"{'='*60}")

    # Prepare data
    ds = prepare_dataset(data_path)
    print(f"  Data: {ds['n_samples']} samples ({ds['n_train']} train, {ds['n_test']} test)")

    # Create model
    kwargs = model_kwargs or {}
    model = create_model(model_name, **kwargs)
    print(f"  Model: {model.name} ({type(model).__name__})")

    # Train
    print(f"  Training...")
    train_metrics = model.fit(ds["X_train"], ds["Y_train"])
    print(f"  Train time: {train_metrics.get('train_time_s', 0):.2f}s")

    # Evaluate on train set
    Y_train_pred = model.predict(ds["X_train"])
    train_eval = compute_metrics(ds["Y_train"], Y_train_pred, ds["target_stats"])

    # Evaluate on test set
    Y_test_pred = model.predict(ds["X_test"])
    test_eval = compute_metrics(ds["Y_test"], Y_test_pred, ds["target_stats"])

    # Print results
    print(f"\n  Train metrics:")
    print(f"    R² Cd: {train_eval['r2_cd']:.4f}, Cl: {train_eval['r2_cl']:.4f}, L/D: {train_eval['r2_ld_ratio']:.4f}")
    print(f"  Test metrics:")
    print(f"    R² Cd: {test_eval['r2_cd']:.4f}, Cl: {test_eval['r2_cl']:.4f}, L/D: {test_eval['r2_ld_ratio']:.4f}")
    print(f"    Physics violations: {test_eval['physics_violations']}")

    # Save model
    model_path = None
    if save_model:
        ext = ".pkl" if model_name == "linear" else ".pt"
        model_path = PROJECT_DIR / "ml" / "models" / f"{model_name}_latest{ext}"
        model.save(model_path)
        print(f"  Model saved: {model_path}")

    # Compile results
    result = {
        "model": model_name,
        "model_kwargs": kwargs,
        "data_path": str(data_path),
        "n_samples": ds["n_samples"],
        "n_train": ds["n_train"],
        "n_test": ds["n_test"],
        "train_metrics": train_metrics,
        "train_eval": train_eval,
        "test_eval": test_eval,
        "model_path": str(model_path) if model_path else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return result


def compare_all_models(data_path: str) -> list:
    """Run experiments for all available models and compare."""
    print("\n" + "#" * 60)
    print("  Model Comparison")
    print("#" * 60)

    results = []

    for name in MODELS:
        try:
            result = run_experiment(name, data_path)
            results.append(result)
        except Exception as e:
            print(f"\n  SKIP {name}: {e}")
            results.append({"model": name, "error": str(e)})

    # Summary table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<10} | {'R² Cd':>8} | {'R² Cl':>8} | {'R² L/D':>8} | {'Violations':>10} | {'Time':>6}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for r in results:
        if "error" in r:
            print(f"  {r['model']:<10} | {'FAILED':>8} | {r['error'][:30]}")
            continue
        te = r["test_eval"]
        tt = r["train_metrics"].get("train_time_s", 0)
        print(f"  {r['model']:<10} | {te['r2_cd']:>8.4f} | {te['r2_cl']:>8.4f} | "
              f"{te['r2_ld_ratio']:>8.4f} | {te['physics_violations']:>10} | {tt:>5.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="ML Surrogate Experiment Runner")
    parser.add_argument("--model", type=str, default="gp",
                        choices=list(MODELS.keys()),
                        help="Model architecture to train")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_DIR / "results" / "sweep_all.json"),
                        help="Training data path")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all models")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    # MLP hyperparameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=str, default="64,64,32",
                        help="Hidden layer sizes, comma-separated")
    args = parser.parse_args()

    if args.compare:
        results = compare_all_models(args.data)
    else:
        kwargs = {}
        if args.model == "mlp":
            kwargs = {
                "lr": args.lr,
                "epochs": args.epochs,
                "hidden_layers": [int(x) for x in args.hidden.split(",")],
            }
        elif args.model == "pinn":
            kwargs = {"epochs": args.epochs}

        results = run_experiment(args.model, args.data, model_kwargs=kwargs)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
