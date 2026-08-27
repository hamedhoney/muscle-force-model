"""
muscle_force_model
===================

A kinematics-driven deep learning pipeline for estimating trunk muscle forces
without electromyography (EMG), based on a CNN + AutoEncoder ensemble
architecture trained per-muscle and per cross-validation fold.

Public API:
    - constants: canonical muscle / disc level names
    - schemas: lightweight dataclasses describing model inputs/outputs
    - models: PyTorch model definitions (CNN, AutoEncoder, Encoder, Decoder)
    - preprocessing: feature extraction from raw motion-capture style data
    - dataset: PyTorch Dataset/DataLoader utilities for training
    - train: training entry points (two-stage autoencoder + CNN distillation)
    - inference: ensemble inference entry point
    - evaluate: validation metrics and reporting
"""

from importlib.metadata import PackageNotFoundError, version

from muscle_force_model.constants import CSA_LEVELS, DISC_NAMES, MUSCLE_NAMES

try:
    __version__ = version("muscle-force-model")
except PackageNotFoundError:  # pragma: no cover - local/dev install
    __version__ = "0.0.0+local"

__all__ = [
    "CSA_LEVELS",
    "DISC_NAMES",
    "MUSCLE_NAMES",
    "__version__",
]
