"""PyTorch Dataset for trial-level training data and subject-wise k-fold splitting.

Each trial is stored as a single ``.npz`` file with the arrays produced by
``preprocessing.extract_features`` plus a ``muscle_forces`` label array of
shape ``(T, n_muscle)`` and a ``subject_id`` string, e.g.::

    np.savez(
        "trial_0001.npz",
        segment_kin=..., norm_muscle_kin=..., external_moments=...,
        muscle_moment_arms=..., muscle_csa=..., anthro=...,
        muscle_forces=..., subject_id="sub01",
    )

See ``scripts/make_synthetic_data.py`` for a script that generates a full
synthetic dataset in this format, useful for smoke-testing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Trial:
    segment_kin: np.ndarray
    norm_muscle_kin: np.ndarray
    external_moments: np.ndarray
    muscle_moment_arms: np.ndarray
    muscle_csa: np.ndarray
    anthro: np.ndarray
    muscle_forces: np.ndarray
    subject_id: str
    inf_time: np.ndarray | None = None


def load_trial(path: str | Path) -> Trial:
    with np.load(path, allow_pickle=True) as f:
        return Trial(
            segment_kin=f["segment_kin"].astype(np.float32),
            norm_muscle_kin=f["norm_muscle_kin"].astype(np.float32),
            external_moments=f["external_moments"].astype(np.float32),
            muscle_moment_arms=f["muscle_moment_arms"].astype(np.float32),
            muscle_csa=f["muscle_csa"].astype(np.float32),
            anthro=f["anthro"].astype(np.float32),
            muscle_forces=f["muscle_forces"].astype(np.float32),
            subject_id=str(f["subject_id"]),
            inf_time=f["inf_time"].astype(np.float32) if "inf_time" in f else None,
        )


def list_trial_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("trial_*.npz"))
    if not files:
        raise FileNotFoundError(f"No trial_*.npz files found in {data_dir}")
    return files


class MuscleForceDataset(Dataset):
    """Wraps a list of trial files. Every item drops the leading batch axis
    that ``preprocessing.extract_features`` produces (shape ``(1, ...)``),
    since batching is handled by the DataLoader.
    """

    def __init__(self, trial_files: list[Path]):
        self.trial_files = list(trial_files)

    def __len__(self) -> int:
        return len(self.trial_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        trial = load_trial(self.trial_files[idx])
        return {
            "segment_kin": torch.from_numpy(trial.segment_kin[0]),
            "norm_muscle_kin": torch.from_numpy(trial.norm_muscle_kin[0]),
            "external_moments": torch.from_numpy(trial.external_moments[0]),
            "muscle_moment_arms": torch.from_numpy(trial.muscle_moment_arms[0]),
            "muscle_csa": torch.from_numpy(trial.muscle_csa[0]),
            "anthro": torch.from_numpy(trial.anthro[0]),
            "muscle_forces": torch.from_numpy(trial.muscle_forces),
        }


def subject_kfold_split(
    trial_files: list[Path], n_folds: int, seed: int = 0
) -> list[tuple[list[Path], list[Path]]]:
    """Split trials into ``n_folds`` (train, val) pairs, grouped by subject so
    that no subject appears in both the train and val set of the same fold.
    """
    subjects: dict[str, list[Path]] = {}
    for path in trial_files:
        with np.load(path, allow_pickle=True) as f:
            subject_id = str(f["subject_id"])
        subjects.setdefault(subject_id, []).append(path)

    subject_ids = sorted(subjects.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    n_folds = min(n_folds, len(subject_ids)) or 1
    fold_buckets: list[list[str]] = [[] for _ in range(n_folds)]
    for i, subject_id in enumerate(subject_ids):
        fold_buckets[i % n_folds].append(subject_id)

    splits = []
    for fold_idx in range(n_folds):
        val_subjects = set(fold_buckets[fold_idx])
        train_files = [p for sid in subject_ids if sid not in val_subjects for p in subjects[sid]]
        val_files = [p for sid in fold_buckets[fold_idx] for p in subjects[sid]]
        splits.append((train_files, val_files))
    return splits
