#!/usr/bin/env bash
# Train all muscle/fold models using configs/train_config.yaml.
# Generates synthetic data first if data/synthetic doesn't exist yet.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d "data/synthetic" ]; then
  echo "No data/synthetic directory found, generating synthetic dataset..."
  uv run python scripts/make_synthetic_data.py --output-dir data/synthetic
fi

uv run muscle-force-model train --config configs/train_config.yaml
