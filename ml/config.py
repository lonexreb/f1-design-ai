#!/usr/bin/env python3
"""
Configuration Loader
====================
Loads config.yaml and provides typed access to all ML, parameter, and
pipeline settings. Single source of truth — eliminates hard-coded duplicates
across ml/ modules.

Usage:
    from ml.config import load_config
    cfg = load_config()
    bounds = cfg.param_bounds()          # {"ride_height": (0.02, 0.05), ...}
    mlp_hp = cfg.ml_hyperparams("mlp")   # {"hidden_layers": [64,64,32], ...}
"""

from pathlib import Path
from typing import Optional

import yaml

PROJECT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.yaml"

_cached_config = None
_cached_path = None


def load_config(path: Optional[Path] = None) -> "Config":
    """Load and cache config.yaml. Returns a Config wrapper."""
    global _cached_config, _cached_path
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if _cached_config is not None and _cached_path == path:
        return _cached_config

    raw = yaml.safe_load(path.read_text())
    _cached_config = Config(raw)
    _cached_path = path
    return _cached_config


def reload_config(path: Optional[Path] = None) -> "Config":
    """Force reload config (clears cache)."""
    global _cached_config, _cached_path
    _cached_config = None
    _cached_path = None
    return load_config(path)


class Config:
    """Typed wrapper around config.yaml dict."""

    def __init__(self, raw: dict):
        self._raw = raw

    def param_bounds(self) -> dict:
        """Return {name: (min, max)} for all parametric variables."""
        params = self._raw.get("parameters", {})
        return {
            name: (float(spec["min"]), float(spec["max"]))
            for name, spec in params.items()
        }

    def param_defaults(self) -> dict:
        """Return {name: default_value} for all parametric variables."""
        params = self._raw.get("parameters", {})
        return {
            name: float(spec["default"])
            for name, spec in params.items()
        }

    def param_names(self) -> list:
        """Return ordered list of parameter names."""
        return list(self._raw.get("parameters", {}).keys())

    def ml_hyperparams(self, model_name: str) -> dict:
        """Return hyperparameters for a specific model type."""
        ml = self._raw.get("ml", {})
        return dict(ml.get(model_name, {}))

    @property
    def ml(self) -> dict:
        return self._raw.get("ml", {})

    @property
    def train_test_split(self) -> float:
        return self._raw.get("ml", {}).get("train_test_split", 0.8)

    @property
    def random_seed(self) -> int:
        return self._raw.get("ml", {}).get("random_seed", 42)

    @property
    def device(self) -> str:
        return self._raw.get("ml", {}).get("device", "auto")

    @property
    def autoresearch(self) -> dict:
        return self._raw.get("autoresearch", {})

    @property
    def active_learning(self) -> dict:
        return self._raw.get("active_learning", {})

    @property
    def simulation(self) -> dict:
        return self._raw.get("simulation", {})

    @property
    def paths(self) -> dict:
        return self._raw.get("paths", {})
