#!/usr/bin/env python3
"""
Surrogate Model Architectures
==============================
Multiple ML model architectures for predicting F1 aerodynamic coefficients
from design parameters. All models share a common interface for use with
Autoresearch experiment runner.

Models:
    LinearSurrogate  - Polynomial regression (sklearn)
    MLPSurrogate     - Multi-layer perceptron (PyTorch)
    GPSurrogate      - Gaussian Process with uncertainty (GPyTorch)
    ModulusSurrogate - PINN wrapper (NVIDIA Modulus / PyTorch)

Usage:
    from ml.surrogate import MODELS, create_model
    model = create_model("gp")
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    uncertainty = model.uncertainty(X_test)  # GP only
"""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()


class SurrogateModel(ABC):
    """Base class for all surrogate models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """
        Train model on data.
        Args:
            X: (N, 5) normalized input params
            Y: (N, 3) standardized targets [cd, cl, ld_ratio]
        Returns:
            Training metrics dict
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets for inputs.
        Args:
            X: (N, 5) normalized input params
        Returns:
            (N, 3) predicted targets
        """

    def uncertainty(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Return prediction uncertainty (if supported).
        Returns None by default, overridden by GP model.
        """
        return None

    def save(self, path: Path):
        """Save model to disk."""
        raise NotImplementedError

    def load(self, path: Path):
        """Load model from disk."""
        raise NotImplementedError


class LinearSurrogate(SurrogateModel):
    """
    Polynomial regression with Ridge regularization.
    Good baseline, works with very small datasets.
    """

    name = "linear"

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha
        self.pipeline = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures

        self.pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
            ("ridge", Ridge(alpha=self.alpha)),
        ])

        start = time.time()
        self.pipeline.fit(X, Y)
        train_time = time.time() - start

        Y_pred = self.pipeline.predict(X)
        mse = np.mean((Y - Y_pred) ** 2)
        n_features = self.pipeline.named_steps["poly"].n_output_features_

        return {
            "train_time_s": train_time,
            "train_mse": float(mse),
            "n_features": n_features,
            "degree": self.degree,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: Path):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "degree": self.degree}, path)

    def load(self, path: Path):
        import joblib
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.degree = data["degree"]


class MLPSurrogate(SurrogateModel):
    """
    Multi-layer perceptron neural network.
    Good for medium-sized datasets (50-500 points).
    """

    name = "mlp"

    def __init__(self, hidden_layers=None, lr: float = 0.001,
                 epochs: int = 500, dropout: float = 0.1):
        self.hidden_layers = hidden_layers or [64, 64, 32]
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.model = None

    def _build(self, in_dim: int, out_dim: int):
        import torch
        import torch.nn as nn

        layers = []
        dim = in_dim
        for h in self.hidden_layers:
            layers.extend([nn.Linear(dim, h), nn.ReLU(), nn.Dropout(self.dropout)])
            dim = h
        layers.append(nn.Linear(dim, out_dim))

        self.model = nn.Sequential(*layers)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        import torch

        self._build(X.shape[1], Y.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self._device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        self.model.train()
        start = time.time()
        best_loss = float("inf")

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = torch.nn.functional.mse_loss(pred, Y_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        train_time = time.time() - start

        return {
            "train_time_s": train_time,
            "train_mse": best_loss,
            "epochs": self.epochs,
            "device": self._device,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            return self.model(X_t).cpu().numpy()

    def save(self, path: Path):
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "hidden_layers": self.hidden_layers,
            "in_dim": self.model[0].in_features,
            "out_dim": self.model[-1].out_features,
        }, str(path))

    def load(self, path: Path):
        import torch
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        self.hidden_layers = checkpoint["hidden_layers"]
        self._build(checkpoint["in_dim"], checkpoint["out_dim"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()


class GPSurrogate(SurrogateModel):
    """
    Gaussian Process regression with Matérn kernel.
    Best for small datasets (< 200 points). Provides uncertainty estimates
    for active learning.
    """

    name = "gp"

    def __init__(self, kernel: str = "matern25", n_restarts: int = 10):
        self.kernel_name = kernel
        self.n_restarts = n_restarts
        self.models = []  # One GP per output dimension

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        try:
            return self._fit_gpytorch(X, Y)
        except ImportError:
            return self._fit_sklearn(X, Y)

    def _fit_gpytorch(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Fit using GPyTorch (CUDA-accelerated)."""
        import torch
        import gpytorch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._use_gpytorch = True

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        self.models = []
        self.likelihoods = []

        start = time.time()

        for dim in range(Y.shape[1]):
            Y_t = torch.tensor(Y[:, dim], dtype=torch.float32, device=device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

            class ExactGP(gpytorch.models.ExactGP):
                def __init__(self, train_x, train_y, lh):
                    super().__init__(train_x, train_y, lh)
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=2.5)
                    )

                def forward(self, x):
                    return gpytorch.distributions.MultivariateNormal(
                        self.mean_module(x), self.covar_module(x)
                    )

            model = ExactGP(X_t, Y_t, likelihood).to(device)
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(200):
                optimizer.zero_grad()
                output = model(X_t)
                loss = -mll(output, Y_t)
                loss.backward()
                optimizer.step()

            model.eval()
            likelihood.eval()
            self.models.append(model)
            self.likelihoods.append(likelihood)

        # Store metadata for save/load reconstruction
        self._n_train = X.shape[0]
        self._n_features = X.shape[1]

        train_time = time.time() - start

        return {
            "train_time_s": train_time,
            "backend": "gpytorch",
            "device": device,
            "n_outputs": Y.shape[1],
        }

    def _fit_sklearn(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Fallback: sklearn GaussianProcessRegressor."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        self._use_gpytorch = False
        self._n_train = X.shape[0]
        self._n_features = X.shape[1]
        self.models = []

        kernel = ConstantKernel() * Matern(nu=2.5)

        start = time.time()
        for dim in range(Y.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                random_state=42,
            )
            gp.fit(X, Y[:, dim])
            self.models.append(gp)

        train_time = time.time() - start

        return {
            "train_time_s": train_time,
            "backend": "sklearn",
            "n_outputs": Y.shape[1],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        if self._use_gpytorch:
            import torch
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            for model, lh in zip(self.models, self.likelihoods):
                with torch.no_grad():
                    pred = lh(model(X_t))
                    preds.append(pred.mean.cpu().numpy())
        else:
            for gp in self.models:
                preds.append(gp.predict(X))

        return np.stack(preds, axis=-1)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Return prediction variance for each output dimension."""
        variances = []
        if self._use_gpytorch:
            import torch
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            for model, lh in zip(self.models, self.likelihoods):
                with torch.no_grad():
                    pred = lh(model(X_t))
                    variances.append(pred.variance.cpu().numpy())
        else:
            for gp in self.models:
                _, std = gp.predict(X, return_std=True)
                variances.append(std ** 2)

        return np.stack(variances, axis=-1)

    def save(self, path: Path):
        """Save GP model. Uses .pkl extension for sklearn, .pt for GPyTorch."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._use_gpytorch:
            import joblib
            # Ensure sklearn GP saves with .pkl extension
            if str(path).endswith(".pt"):
                path = path.with_suffix(".pkl")
            joblib.dump({"models": self.models, "use_gpytorch": False}, path)
        else:
            import torch
            torch.save({
                "models": [m.state_dict() for m in self.models],
                "likelihoods": [lh.state_dict() for lh in self.likelihoods],
                "use_gpytorch": True,
                "n_train": self._n_train,
                "n_features": self._n_features,
            }, str(path))

    def load(self, path: Path):
        path_str = str(path)
        if path_str.endswith(".pkl"):
            import joblib
            data = joblib.load(path)
            self._use_gpytorch = False
            self.models = data["models"]
        else:
            # GPyTorch checkpoint saved via torch.save
            import torch
            import gpytorch
            data = torch.load(path_str, map_location="cpu", weights_only=True)
            self._use_gpytorch = True
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            n_train = data.get("n_train", 10)
            n_features = data.get("n_features", 5)

            # Reconstruct GP models from saved state dicts
            dummy_x = torch.zeros(n_train, n_features)
            dummy_y = torch.zeros(n_train)

            self.models = []
            self.likelihoods = []
            for m_state, lh_state in zip(data["models"], data["likelihoods"]):
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device)

                class ExactGP(gpytorch.models.ExactGP):
                    def __init__(self, train_x, train_y, lh):
                        super().__init__(train_x, train_y, lh)
                        self.mean_module = gpytorch.means.ConstantMean()
                        self.covar_module = gpytorch.kernels.ScaleKernel(
                            gpytorch.kernels.MaternKernel(nu=2.5)
                        )

                    def forward(self, x):
                        return gpytorch.distributions.MultivariateNormal(
                            self.mean_module(x), self.covar_module(x)
                        )

                model = ExactGP(dummy_x, dummy_y, likelihood).to(self._device)
                model.load_state_dict(m_state)
                likelihood.load_state_dict(lh_state)
                model.eval()
                likelihood.eval()
                self.models.append(model)
                self.likelihoods.append(likelihood)


class ModulusSurrogate(SurrogateModel):
    """
    NVIDIA Modulus PINN wrapper.
    Delegates to scripts/modulus_surrogate.py.
    """

    name = "pinn"

    def __init__(self, epochs: int = 2000, physics_weight: float = 0.1):
        self.epochs = epochs
        self.physics_weight = physics_weight
        self._net = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        import sys
        sys.path.insert(0, str(PROJECT_DIR / "scripts"))
        from modulus_surrogate import F1AeroNet

        self._net = F1AeroNet(physics_weight=self.physics_weight)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

        self._net.model.train()
        optimizer = torch.optim.AdamW(self._net.model.parameters(), lr=0.0005)

        start = time.time()
        best_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self._net.model(X_t)
            loss = torch.nn.functional.mse_loss(pred, Y_t)
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self._net.model.state_dict().items()}

        # Restore best model state
        if best_state is not None:
            self._net.model.load_state_dict(best_state)
        self._net.model.eval()

        # Set y_mean/y_std on the net so save() and predict() work correctly
        self._net.y_mean = Y.mean(axis=0)
        self._net.y_std = Y.std(axis=0) + 1e-8
        train_time = time.time() - start

        return {
            "train_time_s": train_time,
            "train_mse": best_loss,
            "epochs": self.epochs,
            "device": device,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = getattr(self, "_device", "cuda" if torch.cuda.is_available() else "cpu")
        self._net.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            return self._net.model(X_t).cpu().numpy()

    def save(self, path: Path):
        if self._net is not None:
            self._net.save(path)

    def load(self, path: Path):
        import sys
        sys.path.insert(0, str(PROJECT_DIR / "scripts"))
        from modulus_surrogate import F1AeroNet
        self._net = F1AeroNet(physics_weight=self.physics_weight)
        self._net.load(path)


# Model registry
MODELS = {
    "linear": LinearSurrogate,
    "mlp": MLPSurrogate,
    "gp": GPSurrogate,
    "pinn": ModulusSurrogate,
}


def create_model(name: str, **kwargs) -> SurrogateModel:
    """Create a model by name from the registry."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)


if __name__ == "__main__":
    print("Available surrogate models:")
    for name, cls in MODELS.items():
        print(f"  {name}: {cls.__doc__.strip().split(chr(10))[0]}")
