"""Canonical names/orderings shared across preprocessing, models, and inference.

Keeping these in one place guarantees that the array ordering produced during
feature extraction always matches the ordering the models were trained with.
"""

from __future__ import annotations

MUSCLE_NAMES: list[str] = [
    "RightLatissimusDorsi",
    "LeftLatissimusDorsi",
    "RightErectorSpinae",
    "LeftErectorSpinae",
    "RightRectusAbdominis",
    "LeftRectusAbdominis",
    "RightExternalOblique",
    "LeftExternalOblique",
    "RightInternalOblique",
    "LeftInternalOblique",
]

DISC_NAMES: list[str] = ["L5S1", "L4L5", "L3L4", "L2L3", "L1L2", "T12L1"]

CSA_LEVELS: list[str] = [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "S1",
    "T10",
    "T11",
    "T12",
    "T8",
    "T9",
]

N_MUSCLE = len(MUSCLE_NAMES)
N_DISC = len(DISC_NAMES)

# Number of resampled timepoints per trial used throughout preprocessing/training.
N_RESAMPLE = 192

# Dimensionality of the learned per-muscle force encoding.
ENCODE_DIM = 2

# Euler angle sequence used for pelvis/chest angles.
EULER_SEQ = "YZX"
