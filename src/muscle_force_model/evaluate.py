"""Validation metrics for comparing predicted vs. ground-truth muscle forces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muscle_force_model.constants import MUSCLE_NAMES


@dataclass
class MuscleMetrics:
    muscle: str
    rmse: float
    nrmse: float
    r: float


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> list[MuscleMetrics]:
    """Per-muscle RMSE, RMSE normalized by the true signal's range, and
    Pearson correlation, averaged across trials.

    Parameters
    ----------
    pred, true:
        Arrays of shape ``(n_trials, T, n_muscle)`` or ``(T, n_muscle)``.
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        true = true[np.newaxis, ...]

    results = []
    for m, muscle_name in enumerate(MUSCLE_NAMES):
        rmses, nrmses, rs = [], [], []
        for trial in range(pred.shape[0]):
            p = pred[trial, :, m]
            t = true[trial, :, m]
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            rng = float(np.ptp(t)) + 1e-8
            nrmse = rmse / rng
            r = float(np.corrcoef(p, t)[0, 1]) if np.std(t) > 1e-8 else float("nan")
            rmses.append(rmse)
            nrmses.append(nrmse)
            rs.append(r)
        results.append(
            MuscleMetrics(
                muscle=muscle_name,
                rmse=float(np.mean(rmses)),
                nrmse=float(np.mean(nrmses)),
                r=float(np.nanmean(rs)),
            )
        )
    return results


def format_metrics_table(metrics: list[MuscleMetrics]) -> str:
    header = f"{'Muscle':<28}{'RMSE':>10}{'NRMSE':>10}{'r':>8}"
    lines = [header, "-" * len(header)]
    for m in metrics:
        lines.append(f"{m.muscle:<28}{m.rmse:>10.2f}{m.nrmse:>10.3f}{m.r:>8.3f}")
    return "\n".join(lines)
