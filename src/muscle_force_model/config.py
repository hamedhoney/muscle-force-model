"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    output_dir: str = "checkpoints"

    n_folds: int = 5
    seed: int = 0

    encode_dim: int = 2
    encode_feature: int = 150
    cnn_features: int = 64

    ae_epochs: int = 200
    ae_lr: float = 1e-3
    ae_batch_size: int = 16
    ae_weight_decay: float = 1e-5

    cnn_epochs: int = 200
    cnn_lr: float = 1e-3
    cnn_batch_size: int = 16
    cnn_weight_decay: float = 1e-5

    device: str = "cpu"
    muscles: list[str] = field(default_factory=list)  # empty = train all muscles

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
