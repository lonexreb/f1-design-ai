#!/usr/bin/env python3
"""
Active Learning Loop for F1 Aerodynamic Design
===============================================
Uses Gaussian Process uncertainty to intelligently propose the next simulation
parameters that will most improve the surrogate model. Closes the loop between
ML training and CFD simulation.

Loop:
    1. Train GP surrogate on current data
    2. Find parameter region with highest uncertainty
    3. Run simulation (Omniverse / OpenFOAM / empirical) at proposed params
    4. Add result to training data
    5. Repeat

Usage:
    # Run 5 active learning iterations using empirical estimates
    python3 ml/active_learning.py --iterations 5 --backend estimate

    # Run with Omniverse CFD backend
    python3 ml/active_learning.py --iterations 3 --backend omniverse

    # Analyze current uncertainty landscape
    python3 ml/active_learning.py --analyze
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from ml.data_prep import (
    load_results, results_to_arrays, normalize, denormalize,
    standardize_targets, destandardize_targets, PARAM_NAMES, PARAM_BOUNDS,
)
from ml.surrogate import GPSurrogate


def propose_next_params(gp_model: GPSurrogate,
                        X_train_norm: np.ndarray,
                        n_candidates: int = 1000,
                        acquisition: str = "uncertainty",
                        seed: int = None) -> dict:
    """
    Propose next simulation parameters using GP uncertainty.

    Args:
        gp_model: Trained GP surrogate
        X_train_norm: Current normalized training inputs
        n_candidates: Random candidates to evaluate
        acquisition: "uncertainty" (max variance) or "ucb" (upper confidence bound)
        seed: Random seed

    Returns:
        dict with parameter names as keys and proposed values
    """
    rng = np.random.RandomState(seed)

    # Generate random candidates in [0, 1]^5
    candidates = rng.uniform(0, 1, size=(n_candidates, len(PARAM_NAMES)))

    # Remove candidates too close to existing training points
    min_dist = np.min(
        np.linalg.norm(candidates[:, None, :] - X_train_norm[None, :, :], axis=2),
        axis=1,
    )
    # Keep candidates with at least some distance from training data
    far_enough = min_dist > 0.05
    if far_enough.sum() > 10:
        candidates = candidates[far_enough]

    # Get uncertainty estimates
    uncertainty = gp_model.uncertainty(candidates)

    if uncertainty is None:
        # Fallback: random selection
        idx = rng.randint(len(candidates))
    elif acquisition == "uncertainty":
        # Max total uncertainty (sum across outputs)
        total_var = uncertainty.sum(axis=1)
        idx = np.argmax(total_var)
    elif acquisition == "ucb":
        # Upper confidence bound: mean + 2*std
        predictions = gp_model.predict(candidates)
        ucb = np.abs(predictions).sum(axis=1) + 2.0 * np.sqrt(uncertainty.sum(axis=1))
        idx = np.argmax(ucb)
    else:
        idx = np.argmax(uncertainty.sum(axis=1))

    # Denormalize to original scale
    best_candidate = candidates[idx:idx + 1]
    best_params_raw = denormalize(best_candidate)[0]

    # Build params dict
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        # Clip to bounds
        val = float(np.clip(best_params_raw[i], lo, hi))
        params[name] = round(val, 6)

    if uncertainty is not None:
        max_unc = float(uncertainty[idx].sum())
        params["_uncertainty"] = max_unc

    return params


def run_simulation(params: dict, backend: str = "estimate") -> dict:
    """
    Run a simulation with the given parameters using the specified backend.

    Returns result dict compatible with results JSON format.
    """
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

    # Import SimulationParams from pipeline
    from run_pipeline import SimulationParams, estimate_coefficients

    sim_params = SimulationParams(
        ride_height=params["ride_height"],
        front_wing_angle=params["front_wing_angle"],
        rear_wing_angle=params["rear_wing_angle"],
        diffuser_angle=params["diffuser_angle"],
        sidepod_undercut=params["sidepod_undercut"],
    )

    if backend == "omniverse":
        from omniverse_sim import run_omniverse_cfd
        result = run_omniverse_cfd(sim_params)
        if result:
            return {
                "params": params,
                "cd": result["cd"],
                "cl": result["cl"],
                "ld_ratio": result["ld_ratio"],
                "converged": result.get("converged", False),
                "iterations": result.get("iterations", 0),
                "wall_time_s": result.get("wall_time_s", 0),
                "notes": "Active learning - Omniverse CFD",
            }
        print("  Omniverse not available, falling back to empirical estimate")

    if backend == "openfoam":
        from run_pipeline import run_blender, find_blender, run_openfoam, find_openfoam, extract_results
        blender_cmd = find_blender()
        if blender_cmd:
            run_blender(sim_params, blender_cmd)
        try:
            of_cmds = find_openfoam()
            run_openfoam(of_cmds)
            of_result = extract_results(sim_params)
            return {
                "params": params,
                "cd": of_result.cd,
                "cl": of_result.cl,
                "ld_ratio": of_result.ld_ratio,
                "converged": of_result.converged,
                "iterations": of_result.iterations,
                "wall_time_s": of_result.wall_time_s,
                "notes": "Active learning - OpenFOAM CFD",
            }
        except (SystemExit, Exception) as e:
            print(f"  OpenFOAM not available ({e}), falling back to empirical estimate")

    if backend == "modulus":
        from modulus_surrogate import predict_modulus
        result = predict_modulus(params)
        if result:
            return {
                "params": params,
                "cd": result["cd"],
                "cl": result["cl"],
                "ld_ratio": result["ld_ratio"],
                "converged": True,
                "iterations": 0,
                "wall_time_s": 0.001,
                "notes": "Active learning - Modulus PINN",
            }

    # Empirical estimate fallback
    result = estimate_coefficients(sim_params)
    return {
        "params": {name: getattr(sim_params, name) for name in PARAM_NAMES},
        "cd": result.cd,
        "cl": result.cl,
        "ld_ratio": result.ld_ratio,
        "converged": False,
        "iterations": 0,
        "wall_time_s": 0.0,
        "notes": "Active learning - Empirical estimate",
    }


def active_learning_loop(data_path: str, n_iterations: int = 5,
                         backend: str = "estimate",
                         acquisition: str = "uncertainty",
                         output_path: str = None) -> list:
    """
    Run the full active learning loop.

    Args:
        data_path: Path to initial results JSON
        n_iterations: Number of AL iterations
        backend: Simulation backend (omniverse, openfoam, modulus, estimate)
        acquisition: Acquisition function (uncertainty, ucb)
        output_path: Path to save updated results

    Returns:
        List of all results (original + new)
    """
    print("=" * 60)
    print(f"  Active Learning Loop ({n_iterations} iterations)")
    print(f"  Backend: {backend} | Acquisition: {acquisition}")
    print("=" * 60)

    # Load existing data
    data = load_results(Path(data_path))
    print(f"  Initial data: {len(data)} samples")

    new_results = []

    for iteration in range(n_iterations):
        print(f"\n  --- Iteration {iteration + 1}/{n_iterations} ---")

        # Convert to arrays
        X, Y = results_to_arrays(data)
        X_norm, _ = normalize(X)
        Y_std, _ = standardize_targets(Y)

        # Train GP
        print("  Training GP surrogate...")
        gp = GPSurrogate()
        train_metrics = gp.fit(X_norm, Y_std)
        print(f"  GP trained ({train_metrics.get('backend', 'unknown')} backend)")

        # Propose next parameters
        proposed = propose_next_params(
            gp, X_norm,
            n_candidates=2000,
            acquisition=acquisition,
            seed=42 + iteration,
        )

        uncertainty = proposed.pop("_uncertainty", None)
        print("  Proposed params:")
        for name, val in proposed.items():
            lo, hi = PARAM_BOUNDS[name]
            pct = (val - lo) / (hi - lo) * 100
            print(f"    {name}: {val:.4f} ({pct:.0f}% of range)")
        if uncertainty is not None:
            print(f"  Uncertainty score: {uncertainty:.4f}")

        # Run simulation
        print(f"  Running simulation ({backend})...")
        result = run_simulation(proposed, backend)
        print(f"  Result: Cd={result['cd']:.4f}, Cl={result['cl']:.4f}, L/D={result['ld_ratio']:.2f}")

        # Add to dataset
        data.append(result)
        new_results.append(result)

    # Save updated dataset
    if output_path is None:
        output_path = str(Path(data_path).parent / "sweep_active_learning.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"\n  Updated dataset saved: {output_path} ({len(data)} total samples)")
    print(f"  New samples added: {len(new_results)}")

    return data


def analyze_uncertainty(data_path: str):
    """Analyze current model uncertainty landscape."""
    print("=" * 60)
    print("  Uncertainty Analysis")
    print("=" * 60)

    data = load_results(Path(data_path))
    X, Y = results_to_arrays(data)
    X_norm, _ = normalize(X)
    Y_std, y_stats = standardize_targets(Y)

    print(f"  Data: {len(data)} samples")

    gp = GPSurrogate()
    gp.fit(X_norm, Y_std)

    # Sample grid for uncertainty visualization
    n_grid = 100
    rng = np.random.RandomState(42)
    grid = rng.uniform(0, 1, size=(n_grid, len(PARAM_NAMES)))

    uncertainty = gp.uncertainty(grid)
    if uncertainty is None:
        print("  Could not compute uncertainty")
        return

    total_unc = uncertainty.sum(axis=1)
    print("\n  Uncertainty statistics:")
    print(f"    Mean: {total_unc.mean():.4f}")
    print(f"    Max:  {total_unc.max():.4f}")
    print(f"    Min:  {total_unc.min():.4f}")

    # Find top-5 most uncertain regions
    top_idx = np.argsort(total_unc)[-5:][::-1]
    grid_raw = denormalize(grid)

    print("\n  Top 5 uncertain parameter regions:")
    for rank, idx in enumerate(top_idx):
        params = grid_raw[idx]
        print(f"    #{rank+1} (unc={total_unc[idx]:.4f}):")
        for i, name in enumerate(PARAM_NAMES):
            print(f"      {name}: {params[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Active Learning for F1 Aero Design")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of active learning iterations")
    parser.add_argument("--backend", type=str, default="estimate",
                        choices=["omniverse", "openfoam", "modulus", "estimate"],
                        help="Simulation backend for new data points")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_DIR / "results" / "sweep_all.json"),
                        help="Initial data path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for updated dataset")
    parser.add_argument("--acquisition", type=str, default="uncertainty",
                        choices=["uncertainty", "ucb"],
                        help="Acquisition function")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze uncertainty landscape (no new simulations)")
    args = parser.parse_args()

    if args.analyze:
        analyze_uncertainty(args.data)
    else:
        active_learning_loop(
            args.data,
            n_iterations=args.iterations,
            backend=args.backend,
            acquisition=args.acquisition,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
