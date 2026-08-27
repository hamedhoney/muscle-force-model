"""Ensemble inference: predict muscle forces from kinematics alone, no EMG.

For each muscle, loads every fold's trained (CNN, AutoEncoder) pair, runs the
CNN on the trial's biomechanical features, decodes the CNN's predicted code
through the frozen decoder, and averages the resulting force estimate across
folds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import torch

from muscle_force_model.constants import ENCODE_DIM, MUSCLE_NAMES, N_MUSCLE
from muscle_force_model.models import CNN, AutoEncoder
from muscle_force_model.preprocessing import FeatureBundle, extract_features
from muscle_force_model.schemas import Area, Moment, MuscleForce, MuscleKinematics

logger = logging.getLogger(__name__)


def _list_fold_dirs(model_dir: str | Path) -> list[Path]:
    model_dir = Path(model_dir)
    folds = sorted(p for p in model_dir.iterdir() if p.is_dir())
    if not folds:
        raise FileNotFoundError(f"No fold subdirectories found under {model_dir}")
    return folds


def load_model_config(model_dir: str | Path) -> dict[str, int]:
    """Load the architecture hyperparameters ``train.py`` saved alongside
    checkpoints, falling back to defaults for checkpoints saved without one.
    """
    import yaml

    config_path = Path(model_dir) / "model_config.yaml"
    defaults = {"encode_dim": ENCODE_DIM, "encode_feature": 150, "cnn_features": 64}
    if not config_path.exists():
        return defaults
    with open(config_path) as f:
        return {**defaults, **yaml.safe_load(f)}


def predict_from_features(
    features: FeatureBundle,
    model_dir: str | Path,
    encode_dim: int = ENCODE_DIM,
    encode_feature: int = 150,
    cnn_features: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Run the fold ensemble on already-extracted features.

    Returns an array of shape ``(n_timepoints, n_muscle)`` on the same time
    base as ``features.inf_time``.
    """
    folds = _list_fold_dirs(model_dir)

    autoencoder = AutoEncoder(encode_size=encode_dim, encode_feature=encode_feature, n_muscle=N_MUSCLE).to(device)
    cnn = CNN(encode_dim, cnn_features).to(device)

    chest_pelvis = torch.tensor(features.segment_kin, dtype=torch.float32, device=device)
    nmk = torch.tensor(features.norm_muscle_kin, dtype=torch.float32, device=device)
    mom = torch.tensor(features.external_moments, dtype=torch.float32, device=device)
    mom_arms = torch.tensor(features.muscle_moment_arms, dtype=torch.float32, device=device)
    dem = torch.tensor(features.anthro, dtype=torch.float32, device=device)
    mca = torch.tensor(features.muscle_csa, dtype=torch.float32, device=device)

    n_timepoints = features.segment_kin.shape[1]
    output = np.zeros((n_timepoints, N_MUSCLE, len(folds)))

    for fold_idx, fold_dir in enumerate(folds):
        for muscle_idx, muscle_name in enumerate(MUSCLE_NAMES):
            ae_path = fold_dir / f"{muscle_name}_auto_encoder.pt"
            cnn_path = fold_dir / f"{muscle_name}_cnn.pt"
            if not ae_path.exists():
                warn(f"No autoencoder checkpoint for {muscle_name} in {fold_dir}")
                continue
            if not cnn_path.exists():
                warn(f"No CNN checkpoint for {muscle_name} in {fold_dir}")
                continue

            autoencoder.load_state_dict(torch.load(ae_path, weights_only=True, map_location=device))
            cnn.load_state_dict(torch.load(cnn_path, weights_only=True, map_location=device))
            autoencoder.eval()
            cnn.eval()

            with torch.no_grad():
                encode_pred = cnn(chest_pelvis, nmk, mom, mom_arms, dem, mca)
                decoded, _ = autoencoder(encode_pred.permute(0, 2, 1), muscle_idx, training=False)

            output[:, muscle_idx, fold_idx] = decoded[0, :, muscle_idx].cpu().numpy()

    return np.mean(output, axis=-1).clip(min=0.1)


def predict_muscle_forces(
    subject_anthro: dict[str, float],
    motion_data: dict[str, Any],
    external_moments: Moment,
    muscle_areas: Area,
    muscle_kinematics: MuscleKinematics,
    muscle_moment_arms: Moment,
    model_dir: str | Path,
    inputs_chart_path: str | Path | None = None,
    device: str = "cpu",
) -> MuscleForce:
    """Full pipeline: raw trial data -> features -> ensemble prediction ->
    resampled back onto the simulation time base -> ``MuscleForce`` output.
    """
    features = extract_features(
        motion_data=motion_data,
        external_moments=external_moments,
        muscle_areas=muscle_areas,
        muscle_kinematics=muscle_kinematics,
        muscle_moment_arms=muscle_moment_arms,
        subject_anthro=subject_anthro,
    )

    sim_time = np.asarray(muscle_kinematics.time, dtype=float)
    sim_rate = 1 / (sim_time[1] - sim_time[0])

    forces_on_inf_time = predict_from_features(features, model_dir, device=device)
    forces_on_sim_time = np.stack(
        [np.interp(sim_time, features.inf_time, forces_on_inf_time[:, m]) for m in range(N_MUSCLE)],
        axis=1,
    )

    if inputs_chart_path:
        _plot_inputs(features, sim_time, forces_on_sim_time, inputs_chart_path)

    return MuscleForce.from_array(
        forces=forces_on_sim_time, time=sim_time, rate=sim_rate, muscle_names=MUSCLE_NAMES
    )


def _plot_inputs(features: FeatureBundle, sim_time, forces, chart_path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, squeeze=False)
        axs[0, 0].set_title("Chest and Pelvis Position")
        axs[0, 0].plot(features.inf_time, np.c_[features.pelvis_angles, features.chest_angles])
        axs[0, 1].set_title("Normalized Muscle Lengths")
        axs[0, 1].plot(features.inf_time, features.norm_muscle_kin[0, :, :N_MUSCLE], label=MUSCLE_NAMES)
        axs[1, 0].set_title("External Moment (L4L5)")
        axs[1, 0].plot(features.inf_time, features.external_moments[0, :, 1, :])
        axs[1, 1].set_title("Predicted Muscle Forces")
        axs[1, 1].plot(sim_time, forces, label=MUSCLE_NAMES)
        for ax in axs.flatten():
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)
        fig.set_size_inches(12, 8)
        fig.suptitle("Muscle Force Inference Inputs")
        fig.tight_layout()
        plt.savefig(chart_path)
        plt.close(fig)
    except Exception:
        logger.warning("Plotting inference inputs failed", exc_info=True)
