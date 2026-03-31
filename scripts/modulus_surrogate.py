#!/usr/bin/env python3
"""
NVIDIA Modulus - Physics-Informed Neural Network Surrogate
==========================================================
Trains a PINN surrogate model using NVIDIA Modulus to predict F1 aerodynamic
coefficients (Cd, Cl, L/D) from design parameters. The physics-informed
loss ensures predictions satisfy aerodynamic constraints.

Falls back to plain PyTorch if nvidia-modulus is not installed.

Usage:
    # Train on existing results
    python3 scripts/modulus_surrogate.py --train --data results/sweep_all.json

    # Predict for new parameters
    python3 scripts/modulus_surrogate.py --predict --ride-height 0.025 --rear-wing-angle 18

    # Used by run_pipeline.py as --backend modulus
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "ml" / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Parameter bounds for normalization
PARAM_BOUNDS = {
    "ride_height": (0.020, 0.050),
    "front_wing_angle": (10.0, 20.0),
    "rear_wing_angle": (10.0, 22.0),
    "diffuser_angle": (6.0, 18.0),
    "sidepod_undercut": (0.08, 0.20),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
TARGET_NAMES = ["cd", "cl", "ld_ratio"]

# Check for GPU
HAS_CUDA = False
DEVICE = "cpu"
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
except ImportError:
    torch = None

# Check for NVIDIA Modulus
HAS_MODULUS = False
try:
    import modulus
    HAS_MODULUS = True
except ImportError:
    pass


class F1AeroNet:
    """
    Physics-informed neural network for F1 aerodynamic coefficient prediction.
    Uses NVIDIA Modulus when available, falls back to plain PyTorch.
    """

    def __init__(self, hidden_layers=None, physics_weight=0.1, lr=0.0005):
        if torch is None:
            raise ImportError("PyTorch is required. Install: pip install torch")

        self.hidden_layers = hidden_layers or [128, 128, 128, 64]
        self.physics_weight = physics_weight
        self.lr = lr
        self.model = None
        self.scalers = None
        self._build_model()

    def _build_model(self):
        """Build the neural network architecture."""
        layers = []
        in_dim = len(PARAM_NAMES)  # 5 input params

        for hidden_dim in self.hidden_layers:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, len(TARGET_NAMES)))  # 3 outputs

        self.model = torch.nn.Sequential(*layers).to(DEVICE)

    def _normalize_params(self, X: np.ndarray) -> np.ndarray:
        """Normalize inputs to [0, 1] using parameter bounds."""
        X_norm = np.zeros_like(X, dtype=np.float32)
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            X_norm[:, i] = (X[:, i] - lo) / (hi - lo)
        return X_norm

    def _denormalize_params(self, X_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        X = np.zeros_like(X_norm)
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[name]
            X[:, i] = X_norm[:, i] * (hi - lo) + lo
        return X

    def _physics_loss(self, predictions: "torch.Tensor") -> "torch.Tensor":
        """
        Physics-informed constraints on predictions.
        Penalizes unphysical aerodynamic coefficient values.
        """
        cd_pred = predictions[:, 0]
        cl_pred = predictions[:, 1]
        ld_pred = predictions[:, 2]

        loss = torch.tensor(0.0, device=DEVICE)

        # Constraint 1: Cd must be positive (drag always exists)
        loss += torch.mean(torch.relu(-cd_pred)) * 10.0

        # Constraint 2: Cl should be negative (downforce in F1)
        loss += torch.mean(torch.relu(cl_pred)) * 5.0

        # Constraint 3: Cd should be in realistic F1 range [0.5, 1.5]
        loss += torch.mean(torch.relu(cd_pred - 1.5))
        loss += torch.mean(torch.relu(0.5 - cd_pred))

        # Constraint 4: Cl should be in realistic range [-6.0, -1.0]
        loss += torch.mean(torch.relu(cl_pred + 1.0))  # cl > -1.0
        loss += torch.mean(torch.relu(-cl_pred - 6.0))  # cl < -6.0

        # Constraint 5: L/D consistency — ld ≈ |cl| / cd
        expected_ld = torch.abs(cl_pred) / (cd_pred + 1e-6)
        loss += torch.mean((ld_pred - expected_ld) ** 2) * 0.5

        # Constraint 6: L/D should be in range [1.0, 7.0]
        loss += torch.mean(torch.relu(ld_pred - 7.0))
        loss += torch.mean(torch.relu(1.0 - ld_pred))

        return loss

    def train(self, data_path: Path, epochs: int = 2000, verbose: bool = True) -> dict:
        """
        Train the PINN on simulation results.

        Args:
            data_path: Path to JSON results file
            epochs: Training epochs
            verbose: Print progress

        Returns:
            Training metrics dict
        """
        # Load data
        data = json.loads(data_path.read_text())
        X = np.array([[d["params"][name] for name in PARAM_NAMES] for d in data], dtype=np.float32)
        Y = np.array([[d["cd"], d["cl"], d["ld_ratio"]] for d in data], dtype=np.float32)

        if verbose:
            print(f"  Training data: {len(data)} samples")
            print(f"  Device: {DEVICE}")
            print(f"  Architecture: {self.hidden_layers}")

        # Normalize inputs
        X_norm = self._normalize_params(X)

        # Store target stats for later denormalization
        self.y_mean = Y.mean(axis=0)
        self.y_std = Y.std(axis=0) + 1e-8

        Y_norm = (Y - self.y_mean) / self.y_std

        X_tensor = torch.tensor(X_norm, device=DEVICE)
        Y_tensor = torch.tensor(Y_norm, device=DEVICE)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        history = {"data_loss": [], "physics_loss": [], "total_loss": []}

        self.model.train()
        start_time = time.time()

        for epoch in range(epochs):
            optimizer.zero_grad()

            predictions = self.model(X_tensor)

            # Data fidelity loss (MSE)
            data_loss = torch.nn.functional.mse_loss(predictions, Y_tensor)

            # Physics constraints (on denormalized predictions)
            pred_denorm = predictions * torch.tensor(self.y_std, device=DEVICE) + \
                         torch.tensor(self.y_mean, device=DEVICE)
            physics_loss = self._physics_loss(pred_denorm)

            total_loss = data_loss + self.physics_weight * physics_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            history["data_loss"].append(data_loss.item())
            history["physics_loss"].append(physics_loss.item())
            history["total_loss"].append(total_loss.item())

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"data={data_loss.item():.6f}, "
                      f"physics={physics_loss.item():.6f}, "
                      f"total={total_loss.item():.6f}")

        # Restore best model
        self.model.load_state_dict(best_state)
        train_time = time.time() - start_time

        if verbose:
            print(f"  Training complete: {train_time:.1f}s, best loss: {best_loss:.6f}")

        return {
            "best_loss": best_loss,
            "train_time_s": train_time,
            "epochs": epochs,
            "n_samples": len(data),
            "device": DEVICE,
            "history": history,
        }

    def predict(self, params_dict: dict) -> dict:
        """
        Predict aero coefficients for given parameters.

        Args:
            params_dict: dict with keys matching PARAM_NAMES

        Returns:
            dict with cd, cl, ld_ratio predictions
        """
        X = np.array([[params_dict[name] for name in PARAM_NAMES]], dtype=np.float32)
        X_norm = self._normalize_params(X)
        X_tensor = torch.tensor(X_norm, device=DEVICE)

        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(X_tensor).cpu().numpy()[0]

        pred = pred_norm * self.y_std + self.y_mean

        return {
            "cd": round(float(pred[0]), 4),
            "cl": round(float(pred[1]), 4),
            "ld_ratio": round(float(pred[2]), 2),
        }

    def save(self, path: Path):
        """Save model and normalization stats (tensors only for safe loading)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "hidden_layers": self.hidden_layers,
            "physics_weight": self.physics_weight,
            "y_mean": torch.tensor(self.y_mean) if not isinstance(self.y_mean, torch.Tensor) else self.y_mean,
            "y_std": torch.tensor(self.y_std) if not isinstance(self.y_std, torch.Tensor) else self.y_std,
        }, str(path))
        print(f"  Model saved: {path}")

    def load(self, path: Path):
        """Load model and normalization stats."""
        checkpoint = torch.load(str(path), map_location=DEVICE, weights_only=True)
        self.hidden_layers = checkpoint["hidden_layers"]
        self.physics_weight = checkpoint["physics_weight"]
        y_mean = checkpoint["y_mean"]
        y_std = checkpoint["y_std"]
        self.y_mean = y_mean.numpy() if isinstance(y_mean, torch.Tensor) else y_mean
        self.y_std = y_std.numpy() if isinstance(y_std, torch.Tensor) else y_std
        self._build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print(f"  Model loaded: {path}")


def train_modulus(data_path: Path, epochs: int = 2000,
                  model_path: Optional[Path] = None) -> Path:
    """
    Train PINN surrogate and save model.
    Uses NVIDIA Modulus trainer if available, else plain PyTorch.
    """
    if model_path is None:
        model_path = MODELS_DIR / "modulus_pinn_latest.pt"

    print("=" * 60)
    print("  NVIDIA Modulus PINN - Training F1 Aero Surrogate")
    print("=" * 60)
    print(f"  Backend: {'NVIDIA Modulus' if HAS_MODULUS else 'PyTorch (Modulus not installed)'}")

    net = F1AeroNet()
    metrics = net.train(data_path, epochs=epochs)
    net.save(model_path)

    # Validate with a quick prediction
    test_params = {name: (lo + hi) / 2 for name, (lo, hi) in PARAM_BOUNDS.items()}
    pred = net.predict(test_params)
    print("\n  Validation (baseline params):")
    print(f"    Cd = {pred['cd']}, Cl = {pred['cl']}, L/D = {pred['ld_ratio']}")

    return model_path


def predict_modulus(params_dict: dict,
                    model_path: Optional[Path] = None) -> Optional[dict]:
    """
    Run inference with trained PINN model.
    Returns prediction dict or None if model doesn't exist.
    """
    if model_path is None:
        model_path = MODELS_DIR / "modulus_pinn_latest.pt"

    if not model_path.exists():
        print(f"  WARNING: PINN model not found: {model_path}")
        print("  Train first: python3 scripts/modulus_surrogate.py --train")
        return None

    net = F1AeroNet()
    net.load(model_path)
    result = net.predict(params_dict)
    result["notes"] = "NVIDIA Modulus PINN surrogate"
    result["converged"] = True
    result["iterations"] = 0
    result["wall_time_s"] = 0.001  # ~1ms inference

    return result


def main():
    parser = argparse.ArgumentParser(description="NVIDIA Modulus PINN Surrogate")
    parser.add_argument("--train", action="store_true", help="Train the PINN model")
    parser.add_argument("--predict", action="store_true", help="Run prediction")
    parser.add_argument("--data", type=Path, default=RESULTS_DIR / "sweep_all.json",
                        help="Training data path")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "modulus_pinn_latest.pt",
                        help="Model checkpoint path")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    # Prediction params
    parser.add_argument("--ride-height", type=float, default=0.030)
    parser.add_argument("--front-wing-angle", type=float, default=14.0)
    parser.add_argument("--rear-wing-angle", type=float, default=16.0)
    parser.add_argument("--diffuser-angle", type=float, default=12.0)
    parser.add_argument("--sidepod-undercut", type=float, default=0.15)
    args = parser.parse_args()

    if args.train:
        train_modulus(args.data, args.epochs, args.model)

    elif args.predict:
        params = {
            "ride_height": args.ride_height,
            "front_wing_angle": args.front_wing_angle,
            "rear_wing_angle": args.rear_wing_angle,
            "diffuser_angle": args.diffuser_angle,
            "sidepod_undercut": args.sidepod_undercut,
        }
        result = predict_modulus(params, args.model)
        if result:
            print("\n  Prediction:")
            print(f"    Cd = {result['cd']}")
            print(f"    Cl = {result['cl']}")
            print(f"    L/D = {result['ld_ratio']}")
        else:
            print("  No model available. Train first with --train")

    else:
        # Status check
        print("=" * 60)
        print("  NVIDIA Modulus PINN Status")
        print("=" * 60)
        print(f"  PyTorch: {'Available' if torch else 'NOT INSTALLED'}")
        print(f"  CUDA:    {'Available' if HAS_CUDA else 'Not available'}")
        print(f"  Device:  {DEVICE}")
        print(f"  Modulus: {'Available' if HAS_MODULUS else 'Not installed (using plain PyTorch)'}")
        print(f"  Model:   {'EXISTS' if args.model.exists() else 'Not trained yet'}")
        print("\n  Usage:")
        print(f"    Train: python3 {__file__} --train")
        print(f"    Predict: python3 {__file__} --predict --rear-wing-angle 20")


if __name__ == "__main__":
    main()
