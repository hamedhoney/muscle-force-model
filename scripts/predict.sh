#!/usr/bin/env bash
# Run inference on a single trial file.
# Usage: scripts/predict.sh path/to/trial.npz [model_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

TRIAL="${1:?Usage: scripts/predict.sh path/to/trial.npz [model_dir]}"
MODEL_DIR="${2:-checkpoints}"

uv run muscle-force-model predict --trial "$TRIAL" --model-dir "$MODEL_DIR"
