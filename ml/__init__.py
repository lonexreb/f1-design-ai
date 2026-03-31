"""
F1 Aerodynamic Design - ML Surrogate Models
============================================
Machine learning surrogate models for predicting F1 aerodynamic coefficients
from design parameters. Supports multiple model architectures and integrates
with Karpathy's Autoresearch for automated experiment iteration.

Modules:
    data_prep           - Data loading, normalization, train/test splitting
    surrogate           - Model architectures (Linear, MLP, GP, PINN)
    experiment          - Experiment runner for Autoresearch
    autoresearch_config - Autoresearch setup and configuration
    active_learning     - GP uncertainty-driven active learning loop
"""
