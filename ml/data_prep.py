#!/usr/bin/env python3
"""
Data Preparation for ML Surrogate Models
=========================================
Loads simulation results (empirical or CFD) from JSON files and prepares
training-ready arrays for surrogate model training.

Usage:
    from ml.data_prep import load_results, prepare_dataset
    X, Y, meta = prepare_dataset("results/sweep_all.json")
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()

PARAM_NAMES = ["ride_height", "front_wing_angle", "rear_wing_angle",
               "diffuser_angle", "sidepod_undercut"]
TARGET_NAMES = ["cd", "cl", "ld_ratio"]

PARAM_BOUNDS = {
    "ride_height": (0.020, 0.050),
    "front_wing_angle": (10.0, 20.0),
    "rear_wing_angle": (10.0, 22.0),
    "diffuser_angle": (6.0, 18.0),
    "sidepod_undercut": (0.08, 0.20),
}


def load_results(path: Path) -> list:
    """Load simulation results from JSON file."""
    data = json.loads(Path(path).read_text())
    return data


def results_to_arrays(data: list) -> tuple:
    """
    Convert list of result dicts to numpy arrays.

    Returns:
        X: (N, 5) array of parameters
        Y: (N, 3) array of targets [cd, cl, ld_ratio]
    """
    X = np.array(
        [[d["params"][name] for name in PARAM_NAMES] for d in data],
        dtype=np.float32,
    )
    Y = np.array(
        [[d["cd"], d["cl"], d["ld_ratio"]] for d in data],
        dtype=np.float32,
    )
    return X, Y


def normalize(X: np.ndarray, bounds: Optional[dict] = None) -> tuple:
    """
    Normalize parameters to [0, 1] using known bounds.

    Returns:
        X_norm: normalized array
        bounds: dict of (lo, hi) per parameter (for inverse transform)
    """
    if bounds is None:
        bounds = PARAM_BOUNDS

    X_norm = np.zeros_like(X)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = bounds[name]
        X_norm[:, i] = (X[:, i] - lo) / (hi - lo)

    return X_norm, bounds


def denormalize(X_norm: np.ndarray, bounds: Optional[dict] = None) -> np.ndarray:
    """Reverse normalization from [0, 1] to original scale."""
    if bounds is None:
        bounds = PARAM_BOUNDS

    X = np.zeros_like(X_norm)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = bounds[name]
        X[:, i] = X_norm[:, i] * (hi - lo) + lo
    return X


def standardize_targets(Y: np.ndarray) -> tuple:
    """
    Standardize targets to zero mean, unit variance.

    Returns:
        Y_std: standardized array
        stats: dict with mean and std for inverse transform
    """
    mean = Y.mean(axis=0)
    std = Y.std(axis=0) + 1e-8
    Y_std = (Y - mean) / std
    return Y_std, {"mean": mean, "std": std}


def destandardize_targets(Y_std: np.ndarray, stats: dict) -> np.ndarray:
    """Reverse standardization."""
    return Y_std * stats["std"] + stats["mean"]


def train_test_split(X: np.ndarray, Y: np.ndarray,
                     train_ratio: float = 0.8,
                     seed: int = 42) -> dict:
    """
    Split data into train and test sets.

    Returns:
        dict with keys: X_train, X_test, Y_train, Y_test, train_idx, test_idx
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * train_ratio)

    train_idx = indices[:split]
    test_idx = indices[split:]

    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "Y_train": Y[train_idx],
        "Y_test": Y[test_idx],
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def prepare_dataset(data_path: str, train_ratio: float = 0.8,
                    seed: int = 42) -> dict:
    """
    Full data preparation pipeline: load → arrays → normalize → split.

    Args:
        data_path: Path to results JSON
        train_ratio: Fraction for training
        seed: Random seed

    Returns:
        dict with all prepared data and metadata
    """
    data = load_results(Path(data_path))
    X, Y = results_to_arrays(data)
    X_norm, bounds = normalize(X)
    Y_std, y_stats = standardize_targets(Y)

    split = train_test_split(X_norm, Y_std, train_ratio, seed)

    return {
        # Raw data
        "X_raw": X,
        "Y_raw": Y,
        # Processed data
        "X_norm": X_norm,
        "Y_std": Y_std,
        # Train/test split (normalized/standardized)
        **split,
        # Metadata for inverse transforms
        "param_bounds": bounds,
        "target_stats": y_stats,
        "param_names": PARAM_NAMES,
        "target_names": TARGET_NAMES,
        "n_samples": len(data),
        "n_train": len(split["X_train"]),
        "n_test": len(split["X_test"]),
    }


if __name__ == "__main__":
    # Quick test
    data_path = PROJECT_DIR / "results" / "sweep_all.json"
    if data_path.exists():
        ds = prepare_dataset(str(data_path))
        print(f"Loaded {ds['n_samples']} samples ({ds['n_train']} train, {ds['n_test']} test)")
        print(f"X shape: {ds['X_raw'].shape}, Y shape: {ds['Y_raw'].shape}")
        print(f"Cd range: {ds['Y_raw'][:,0].min():.3f} - {ds['Y_raw'][:,0].max():.3f}")
        print(f"Cl range: {ds['Y_raw'][:,1].min():.3f} - {ds['Y_raw'][:,1].max():.3f}")
        print(f"L/D range: {ds['Y_raw'][:,2].min():.2f} - {ds['Y_raw'][:,2].max():.2f}")
    else:
        print(f"No data found at {data_path}")
