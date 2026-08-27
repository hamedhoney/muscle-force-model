# Data directory

Real motion-capture / musculoskeletal-model output does not belong in
version control. This directory is intentionally empty in the repository.

To try the pipeline end to end, generate a synthetic dataset:

```bash
uv run python scripts/make_synthetic_data.py --output-dir data/synthetic
```

To use your own data, produce one `.npz` file per trial matching the format
described in the top-level README's "Data format" section, either directly
or via `muscle_force_model.preprocessing.extract_features`.
