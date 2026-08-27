"""Command-line interface.

Usage:
    muscle-force-model train --config configs/train_config.yaml
    muscle-force-model predict --trial data/synthetic/trial_0000.npz --model-dir checkpoints
    muscle-force-model evaluate --data-dir data/synthetic --model-dir checkpoints
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from muscle_force_model.config import TrainConfig
from muscle_force_model.constants import MUSCLE_NAMES, N_MUSCLE
from muscle_force_model.dataset import list_trial_files, load_trial
from muscle_force_model.evaluate import compute_metrics, format_metrics_table
from muscle_force_model.inference import load_model_config, predict_from_features
from muscle_force_model.train import run_training

logger = logging.getLogger(__name__)


def _cmd_train(args: argparse.Namespace) -> None:
    config = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    run_training(config)


def _trial_to_features(trial) -> SimpleNamespace:
    inf_time = trial.inf_time
    if inf_time is None:
        inf_time = np.linspace(0, 1, trial.segment_kin.shape[1])
    return SimpleNamespace(
        inf_time=inf_time,
        segment_kin=trial.segment_kin,
        norm_muscle_kin=trial.norm_muscle_kin,
        external_moments=trial.external_moments,
        muscle_moment_arms=trial.muscle_moment_arms,
        muscle_csa=trial.muscle_csa,
        anthro=trial.anthro,
    )


def _cmd_predict(args: argparse.Namespace) -> None:
    trial = load_trial(args.trial)
    features = _trial_to_features(trial)
    model_config = load_model_config(args.model_dir)
    forces = predict_from_features(features, args.model_dir, device=args.device, **model_config)

    default_out_dir = Path("predictions")
    default_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output or default_out_dir / f"{Path(args.trial).stem}_predicted.npz")
    np.savez(out_path, forces=forces, muscle_names=np.array(MUSCLE_NAMES), time=features.inf_time)
    logger.info("Saved predicted forces to %s", out_path)

    print(f"{'Muscle':<28}{'Peak force (N)':>16}")
    for m, name in enumerate(MUSCLE_NAMES):
        print(f"{name:<28}{forces[:, m].max():>16.1f}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    trial_files = list_trial_files(args.data_dir)
    model_config = load_model_config(args.model_dir)
    all_pred, all_true = [], []
    for path in trial_files:
        trial = load_trial(path)
        features = _trial_to_features(trial)
        pred = predict_from_features(features, args.model_dir, device=args.device, **model_config)
        true = trial.muscle_forces
        pred_on_true_time = np.stack(
            [np.interp(np.linspace(0, 1, true.shape[0]), np.linspace(0, 1, pred.shape[0]), pred[:, m]) for m in range(N_MUSCLE)],
            axis=1,
        )
        all_pred.append(pred_on_true_time)
        all_true.append(true)

    metrics = compute_metrics(np.stack(all_pred), np.stack(all_true))
    print(format_metrics_table(metrics))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muscle-force-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Train per-muscle, per-fold models")
    p_train.add_argument("--config", type=str, default=None)
    p_train.add_argument("--data-dir", type=str, default=None)
    p_train.add_argument("--output-dir", type=str, default=None)
    p_train.set_defaults(func=_cmd_train)

    p_predict = subparsers.add_parser("predict", help="Run ensemble inference on a single trial")
    p_predict.add_argument("--trial", type=str, required=True)
    p_predict.add_argument("--model-dir", type=str, required=True)
    p_predict.add_argument("--output", type=str, default=None)
    p_predict.add_argument("--device", type=str, default="cpu")
    p_predict.set_defaults(func=_cmd_predict)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate model predictions against labels")
    p_eval.add_argument("--data-dir", type=str, required=True)
    p_eval.add_argument("--model-dir", type=str, required=True)
    p_eval.add_argument("--device", type=str, default="cpu")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
