"""Two-stage training pipeline.

Stage 1 trains an ``AutoEncoder`` per muscle on ground-truth muscle force
labels: the encoder compresses the target muscle's force signal into a
low-dimensional code, the decoder reconstructs all muscle forces from it.

Stage 2 freezes that autoencoder and trains a ``CNN`` to predict the same
code directly from biomechanical inputs, with no force/EMG signal involved.
This is what makes EMG-free inference possible: at inference time the CNN's
predicted code is fed straight into the frozen decoder.

Checkpoints are written as ``{output_dir}/fold_{k}/{muscle}_auto_encoder.pt``
and ``{output_dir}/fold_{k}/{muscle}_cnn.pt``, matching the layout
``inference.py`` expects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from muscle_force_model.config import TrainConfig
from muscle_force_model.constants import MUSCLE_NAMES, N_MUSCLE
from muscle_force_model.dataset import (
    MuscleForceDataset,
    list_trial_files,
    subject_kfold_split,
)
from muscle_force_model.models import CNN, AutoEncoder

logger = logging.getLogger(__name__)


def _std(x: np.ndarray, axis) -> np.ndarray:
    return np.std(x, axis=axis, keepdims=True) + 1e-8


def compute_norm_stats(train_files: list[Path]) -> dict[str, np.ndarray]:
    """Compute per-channel mean/std over a training split, used to set the
    frozen normalization parameters on both models before training.
    """
    from muscle_force_model.dataset import load_trial

    chest_pelvis, nmk, mom, mom_arms, dem, mca, forces = [], [], [], [], [], [], []
    for path in train_files:
        trial = load_trial(path)
        chest_pelvis.append(trial.segment_kin[0])
        nmk.append(trial.norm_muscle_kin[0])
        mom.append(trial.external_moments[0])
        mom_arms.append(trial.muscle_moment_arms[0])
        dem.append(trial.anthro[0])
        mca.append(trial.muscle_csa[0])
        forces.append(trial.muscle_forces)

    chest_pelvis = np.stack(chest_pelvis)
    nmk = np.stack(nmk)
    mom = np.stack(mom)
    mom_arms = np.stack(mom_arms)
    dem = np.stack(dem)
    mca = np.stack(mca)
    forces = np.stack(forces)

    return {
        # (1, 1, 1 + 6*3), matches CNN.CHEST_PELVIS_std
        "chest_pelvis_std": _std(chest_pelvis, axis=(0, 1)),
        # (1, 1, 2*n_muscle), matches CNN.NMK_std
        "nmk_std": _std(nmk, axis=(0, 1)),
        # (1, 1, n_disc, 3), matches CNN.MOM_std
        "mom_std": _std(mom, axis=(0, 1)),
        # (1, 1, n_muscle, n_disc, 3), matches CNN.MOM_ARMS_std
        "mom_arms_std": _std(mom_arms, axis=(0, 1)),
        # (1, 3), matches CNN.D_mean / D_std
        "dem_mean": np.mean(dem, axis=0, keepdims=True),
        "dem_std": _std(dem, axis=0),
        # (1, 1, 1), matches CNN.MCA_mean / MCA_std
        "mca_mean": np.full((1, 1, 1), np.mean(mca)),
        "mca_std": np.full((1, 1, 1), np.std(mca) + 1e-8),
        # (1, 1, n_muscle), matches AutoEncoder.Y_mean / Y_std
        "forces_mean": np.mean(forces, axis=(0, 1))[np.newaxis, np.newaxis, :],
        "forces_std": _std(forces, axis=(0, 1)),
    }


def _set_autoencoder_stats(model: AutoEncoder, stats: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        model.Y_mean.copy_(torch.tensor(stats["forces_mean"], dtype=torch.float32))
        model.Y_std.copy_(torch.tensor(stats["forces_std"], dtype=torch.float32))


def _set_cnn_stats(model: CNN, stats: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        model.CHEST_PELVIS_std.copy_(torch.tensor(stats["chest_pelvis_std"], dtype=torch.float32))
        model.NMK_std.copy_(torch.tensor(stats["nmk_std"], dtype=torch.float32))
        model.MOM_std.copy_(torch.tensor(stats["mom_std"], dtype=torch.float32))
        model.MOM_ARMS_std.copy_(torch.tensor(stats["mom_arms_std"], dtype=torch.float32))
        model.D_mean.copy_(torch.tensor(stats["dem_mean"], dtype=torch.float32))
        model.D_std.copy_(torch.tensor(stats["dem_std"], dtype=torch.float32))
        model.MCA_mean.copy_(torch.tensor(stats["mca_mean"], dtype=torch.float32))
        model.MCA_std.copy_(torch.tensor(stats["mca_std"], dtype=torch.float32))


def train_autoencoder(
    muscle_idx: int,
    train_loader: DataLoader,
    stats: dict[str, np.ndarray],
    config: TrainConfig,
) -> AutoEncoder:
    model = AutoEncoder(
        encode_size=config.encode_dim, encode_feature=config.encode_feature, n_muscle=N_MUSCLE
    ).to(config.device)
    _set_autoencoder_stats(model, stats)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.ae_lr, weight_decay=config.ae_weight_decay)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(config.ae_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            forces = batch["muscle_forces"].to(config.device)
            optimizer.zero_grad()
            recon, _ = model(forces, muscle_idx, training=True)
            loss = loss_fn(recon, forces)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % max(1, config.ae_epochs // 10) == 0:
            logger.info(
                "  [autoencoder %s] epoch %d/%d loss=%.5f",
                MUSCLE_NAMES[muscle_idx],
                epoch + 1,
                config.ae_epochs,
                epoch_loss / len(train_loader),
            )
    return model


def train_cnn(
    muscle_idx: int,
    train_loader: DataLoader,
    autoencoder: AutoEncoder,
    stats: dict[str, np.ndarray],
    config: TrainConfig,
) -> CNN:
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    model = CNN(config.encode_dim, config.cnn_features).to(config.device)
    _set_cnn_stats(model, stats)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.cnn_lr, weight_decay=config.cnn_weight_decay)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(config.cnn_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            chest_pelvis = batch["segment_kin"].to(config.device)
            nmk = batch["norm_muscle_kin"].to(config.device)
            mom = batch["external_moments"].to(config.device)
            mom_arms = batch["muscle_moment_arms"].to(config.device)
            dem = batch["anthro"].to(config.device)
            mca = batch["muscle_csa"].to(config.device)
            forces = batch["muscle_forces"].to(config.device)

            with torch.no_grad():
                _, target_enc = autoencoder(forces, muscle_idx, training=True)

            optimizer.zero_grad()
            pred_enc = model(chest_pelvis, nmk, mom, mom_arms, dem, mca)
            loss = loss_fn(pred_enc.permute(0, 2, 1), target_enc)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % max(1, config.cnn_epochs // 10) == 0:
            logger.info(
                "  [cnn %s] epoch %d/%d loss=%.5f",
                MUSCLE_NAMES[muscle_idx],
                epoch + 1,
                config.cnn_epochs,
                epoch_loss / len(train_loader),
            )
    return model


def _save_model_config(config: TrainConfig) -> None:
    """Persist the architecture hyperparameters needed to reconstruct these
    checkpoints at inference/evaluation time (batch size, epochs, etc. don't
    matter then, only the shapes do).
    """
    import yaml

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "encode_dim": config.encode_dim,
                "encode_feature": config.encode_feature,
                "cnn_features": config.cnn_features,
            },
            f,
        )


def run_training(config: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    trial_files = list_trial_files(config.data_dir)
    splits = subject_kfold_split(trial_files, config.n_folds, config.seed)
    muscles = config.muscles or MUSCLE_NAMES
    _save_model_config(config)

    for fold_idx, (train_files, val_files) in enumerate(splits):
        logger.info("Fold %d/%d: %d train trials, %d val trials", fold_idx + 1, len(splits), len(train_files), len(val_files))
        fold_dir = Path(config.output_dir) / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        stats = compute_norm_stats(train_files)
        train_loader = DataLoader(
            MuscleForceDataset(train_files), batch_size=config.ae_batch_size, shuffle=True
        )

        for muscle_name in muscles:
            muscle_idx = MUSCLE_NAMES.index(muscle_name)
            logger.info("Training %s (fold %d)", muscle_name, fold_idx + 1)

            autoencoder = train_autoencoder(muscle_idx, train_loader, stats, config)
            cnn = train_cnn(muscle_idx, train_loader, autoencoder, stats, config)

            torch.save(autoencoder.state_dict(), fold_dir / f"{muscle_name}_auto_encoder.pt")
            torch.save(cnn.state_dict(), fold_dir / f"{muscle_name}_cnn.pt")

    logger.info("Training complete. Checkpoints saved to %s", config.output_dir)
