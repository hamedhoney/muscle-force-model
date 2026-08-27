# Muscle Force Model

A kinematics-driven deep learning pipeline for estimating trunk muscle
forces without electromyography (EMG). Given trunk/pelvis kinematics,
muscle-tendon lengths, external moments, muscle moment arms, and subject
anthropometry, the model estimates the force time series for 10 trunk
muscles per side (latissimus dorsi, erector spinae, rectus abdominis,
external oblique, internal oblique).

## How it works

Each muscle gets its own trained pair of models, repeated across k
cross-validation folds:

1. **AutoEncoder** (`Encoder` + `Decoder`): trained on ground-truth muscle
   force labels. The encoder compresses one target muscle's force signal
   into a low-dimensional latent code; the decoder reconstructs all 10
   muscles' forces from that code.
2. **CNN**: trained to predict the *same* latent code directly from
   biomechanical inputs alone, no force or EMG signal involved.

At inference time, the CNN's predicted code is fed straight into the frozen
decoder from step 1, skipping the encoder entirely, so no EMG or force
measurement is needed to produce an estimate. Predictions from all folds
are averaged into the final result.

## Repository layout

```
src/muscle_force_model/
    constants.py       muscle/disc names, fixed array orderings
    schemas.py          lightweight dataclasses for model I/O
    quat_utils.py        quaternion math (scipy-backed)
    filters.py            FIR filter design
    preprocessing.py    raw trial data -> fixed-size feature arrays
    models.py            CNN, AutoEncoder, Encoder, Decoder
    dataset.py            PyTorch Dataset + subject-wise k-fold splitting
    config.py              training configuration (YAML)
    train.py                two-stage training loop
    inference.py           ensemble inference
    evaluate.py            validation metrics
    cli.py                    train / predict / evaluate entry points
scripts/
    make_synthetic_data.py  generates a synthetic dataset for smoke-testing
tests/                          unit tests
configs/train_config.yaml       example training config
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
git clone https://github.com/<your-org>/muscle-force-model.git
cd muscle-force-model
uv venv
uv pip install -e ".[dev,viz]"
```

Or, using `uv sync` with the lockfile (once generated in your own clone):

```bash
uv sync --extra dev --extra viz
```

## Quickstart

The repo ships with a synthetic data generator so you can exercise the
whole pipeline (train, predict, evaluate) without needing real
motion-capture or musculoskeletal-model data.

```bash
# 1. Generate a small synthetic dataset (8 subjects x 3 trials)
uv run python scripts/make_synthetic_data.py --output-dir data/synthetic

# 2. Train (2-fold, few epochs, just to confirm the pipeline runs end to end)
uv run muscle-force-model train \
    --data-dir data/synthetic \
    --output-dir checkpoints

# 3. Predict on a single trial
uv run muscle-force-model predict \
    --trial data/synthetic/trial_0000.npz \
    --model-dir checkpoints

# 4. Evaluate predictions against the synthetic ground truth
uv run muscle-force-model evaluate \
    --data-dir data/synthetic \
    --model-dir checkpoints
```

Training hyperparameters (epochs, learning rate, number of folds, encoder
size, etc.) live in `configs/train_config.yaml`:

```bash
uv run muscle-force-model train --config configs/train_config.yaml
```

## Data format

Training and evaluation both read per-trial `.npz` files with these arrays
(see `scripts/make_synthetic_data.py` for a full worked example):

| key                   | shape                        | description                                  |
|-----------------------|-------------------------------|-----------------------------------------------|
| `segment_kin`         | `(1, T, 19)`                  | time, pelvis/chest Euler angles + derivatives |
| `norm_muscle_kin`     | `(1, T, 20)`                  | normalized muscle length + velocity            |
| `external_moments`    | `(1, T, 6, 3)`                 | external moment per spinal level, per plane   |
| `muscle_moment_arms`  | `(1, T, 10, 6, 3)`             | muscle moment arm per level, per plane        |
| `muscle_csa`          | `(1, 10, 1)`                    | peak muscle cross-sectional area              |
| `anthro`              | `(1, 3)`                        | age (yr), stature (cm), mass (kg)             |
| `muscle_forces`       | `(T, 10)`                        | ground-truth muscle force labels (training)   |
| `subject_id`          | scalar string                    | used for subject-wise k-fold splitting        |

`T` is 192 resampled timepoints by default (`constants.N_RESAMPLE`).

If your source data isn't already in this shape, use
`muscle_force_model.preprocessing.extract_features` to build it from raw
motion data plus `Moment` / `Area` / `MuscleKinematics` containers
(`muscle_force_model.schemas`) describing your trial.

## A note on the original dependencies

The original inference script this repository is built from depended on two
internal, non-public packages: `signals` (quaternion math and FIR filter
design) and `mechmodel.adams.dataclasses` (input/output container types).
Since those aren't publicly available, this repo reimplements their
functionality from scratch:

- `quat_utils.py` reimplements the quaternion operations
  (`fillmissing`, `multiply`, `rotateFrame`, `swingTwist`, `ldivide`,
  `windowFilter`, `interp`, `fromEulerAngles`, `toEulerAngles`) on top of
  `scipy.spatial.transform`.
- `filters.py` reimplements the flat-top FIR low-pass filter design.
- `schemas.py` reimplements the `Moment` / `Area` / `MuscleKinematics` /
  `MuscleForce` containers as plain dataclasses.

If you have access to the original packages, you can swap these modules out;
`preprocessing.py` and `inference.py` only depend on the attributes these
modules expose, not on any particular implementation.

## Testing

```bash
uv run pytest
```

## Disclaimer

This model estimates muscle forces from kinematic and anthropometric inputs
and is intended for biomechanics research. It has not been validated for
clinical decision-making, and predictions should be interpreted alongside
domain expertise and, where possible, ground-truth (e.g. EMG-informed)
comparisons.

## License

MIT. See [LICENSE](LICENSE).
