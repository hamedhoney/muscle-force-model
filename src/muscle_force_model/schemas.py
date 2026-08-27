"""Standalone data schemas describing model inputs and outputs.

The original inference script imported these shapes from a proprietary
``mechmodel.adams.dataclasses`` module. That package is not public, so this
module provides open, minimal, well-documented equivalents that expose the
same fields the original code relied on (``.contributors``, ``.muscles``,
``.time``, etc.). If your organization has access to the original
``mechmodel`` package, you can swap this module out and the rest of the
pipeline (preprocessing/inference) will work unmodified, since it only
depends on the attributes defined here.

All arrays are plain numpy arrays; nothing here depends on a simulation
engine, so these classes are equally useful for synthetic data, experimental
data, or data produced by any musculoskeletal modeling tool (OpenSim, AnyBody,
AMS/ADAMS, etc.) as long as it is mapped into this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MomentContributor:
    """A single contributor's moment time series at one anatomical level.

    For ``ExternalMoments`` the "contributor" is a spinal level (e.g. L4L5)
    and ``moment`` is the net external moment about that level over time.
    For ``MuscleMomentArms`` the contributor is a (level, muscle) pair and
    ``moment`` holds the muscle's moment-arm-derived moment contribution.
    """

    level: str
    moment: np.ndarray  # shape (T, 3) -> [X, Y, Z] planes
    name: str | None = None  # muscle name, only used for moment-arm contributors


@dataclass
class Moment:
    """Container matching the original ``dc.Moment`` interface.

    Used for both external moments (per spinal level) and muscle moment arms
    (per spinal level + muscle).
    """

    time: np.ndarray
    contributors: list[MomentContributor] = field(default_factory=list)


@dataclass
class MuscleAreaEntry:
    name: str
    max: float  # peak physiological cross-sectional area (cm^2)


@dataclass
class Area:
    """Container matching the original ``dc.Area`` interface."""

    muscleArea: list[MuscleAreaEntry] = field(default_factory=list)


@dataclass
class MuscleKinematicsEntry:
    name: str
    length: np.ndarray  # shape (T,)
    velocity: np.ndarray  # shape (T,)


@dataclass
class MuscleKinematics:
    """Container matching the original ``dc.MuscleKinematics`` interface."""

    time: np.ndarray
    muscles: list[MuscleKinematicsEntry] = field(default_factory=list)


@dataclass
class Force:
    """A single muscle's force time series and (optional) decomposition."""

    name: str
    total: np.ndarray  # shape (T,)
    activeUnmodulated: np.ndarray | None = None
    active: np.ndarray | None = None
    passive: np.ndarray | None = None


@dataclass
class MuscleForce:
    """Container matching the original ``dc.MuscleForce`` interface.

    This is both the training-label format and the inference output format.
    """

    formatVersion: str
    rate: float
    time: np.ndarray
    muscles: list[Force] = field(default_factory=list)

    def as_array(self, muscle_names: list[str]) -> np.ndarray:
        """Stack per-muscle ``total`` force into a (T, n_muscle) array."""
        by_name = {m.name: m for m in self.muscles}
        return np.stack([by_name[name].total for name in muscle_names], axis=1)

    @classmethod
    def from_array(
        cls,
        forces: np.ndarray,
        time: np.ndarray,
        rate: float,
        muscle_names: list[str],
        format_version: str = "1.0",
    ) -> MuscleForce:
        """Build a MuscleForce container from a (T, n_muscle) array."""
        muscles = [
            Force(name=name, total=forces[:, i])
            for i, name in enumerate(muscle_names)
        ]
        return cls(formatVersion=format_version, rate=rate, time=time, muscles=muscles)
