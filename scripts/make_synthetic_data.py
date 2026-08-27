#!/usr/bin/env python
"""Generate a synthetic dataset so the full train/predict/evaluate pipeline can
be smoke-tested without real motion-capture or musculoskeletal-model data.

Builds raw trial data in the same shape ``preprocessing.extract_features``
expects, runs it through the real preprocessing pipeline, and synthesizes a
plausible ground-truth muscle force target correlated with trunk flexion so
the two-stage training procedure has real signal to learn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from muscle_force_model.constants import DISC_NAMES, MUSCLE_NAMES, N_MUSCLE
from muscle_force_model.preprocessing import extract_features
from muscle_force_model.schemas import (
    Area,
    Moment,
    MomentContributor,
    MuscleAreaEntry,
    MuscleKinematics,
    MuscleKinematicsEntry,
)


def _random_quat_trajectory(rng: np.random.Generator, n: int, max_deg: float) -> np.ndarray:
    """Small smooth random rotations, returned as scalar-first quaternions."""
    from scipy.spatial.transform import Rotation

    t = np.linspace(0, 1, n)
    freqs = rng.uniform(0.5, 2.0, size=3)
    phases = rng.uniform(0, 2 * np.pi, size=3)
    angles_deg = max_deg * np.stack(
        [np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)], axis=1
    )
    quat_xyzw = Rotation.from_euler("XYZ", angles_deg, degrees=True).as_quat()
    return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)


def make_trial(rng: np.random.Generator, subject_id: str, mot_rate: float = 100.0, n_samples: int = 300):
    mot_time = np.arange(n_samples) / mot_rate

    pelvis_ori = _random_quat_trajectory(rng, n_samples, max_deg=5)
    chest_ori = _random_quat_trajectory(rng, n_samples, max_deg=30)
    drift_correction = np.array([1.0, 0.0, 0.0, 0.0])

    motion_data = {
        "joints": [{"rate": mot_rate, "time": mot_time}],
        "segments": [
            {"name": "Pelvis", "orientationFinal": pelvis_ori},
            {"name": "Chest", "orientationFinal": chest_ori, "driftCorrection": drift_correction},
        ],
    }

    sim_time = np.linspace(0, mot_time[-1], n_samples)
    trunk_flexion = np.abs(np.sin(2 * np.pi * 0.6 * sim_time)) * 30 + rng.normal(0, 1, n_samples)

    muscle_kin_entries = []
    for muscle_name in MUSCLE_NAMES:
        neutral_length = rng.uniform(0.2, 0.4)
        length = neutral_length * (1 + 0.05 * np.sin(2 * np.pi * 0.6 * sim_time + rng.uniform(0, 1)))
        velocity = np.gradient(length, sim_time)
        muscle_kin_entries.append(MuscleKinematicsEntry(name=muscle_name, length=length, velocity=velocity))
    muscle_kinematics = MuscleKinematics(time=sim_time, muscles=muscle_kin_entries)

    external_moment_contributors = []
    for level in DISC_NAMES:
        scale = rng.uniform(0.8, 1.2)
        moment = np.stack(
            [
                rng.normal(0, 2, n_samples),
                trunk_flexion * scale,
                rng.normal(0, 2, n_samples),
            ],
            axis=1,
        )
        external_moment_contributors.append(MomentContributor(level=level, moment=moment))
    external_moments = Moment(time=sim_time, contributors=external_moment_contributors)

    moment_arm_contributors = []
    for level in DISC_NAMES:
        for muscle_name in MUSCLE_NAMES:
            base_arm = rng.uniform(-0.05, 0.05, size=3)
            moment = np.tile(base_arm, (n_samples, 1)) + rng.normal(0, 0.001, (n_samples, 3))
            moment_arm_contributors.append(MomentContributor(level=level, moment=moment, name=muscle_name))
    muscle_moment_arms = Moment(time=sim_time, contributors=moment_arm_contributors)

    muscle_areas = Area(
        muscleArea=[
            MuscleAreaEntry(name=muscle_name, max=rng.uniform(3.0, 8.0))
            for muscle_name in MUSCLE_NAMES
        ]
    )

    subject_anthro = {
        "age_yr": float(rng.uniform(20, 60)),
        "stature_cm": float(rng.uniform(155, 190)),
        "mass_kg": float(rng.uniform(55, 100)),
    }

    features = extract_features(
        motion_data=motion_data,
        external_moments=external_moments,
        muscle_areas=muscle_areas,
        muscle_kinematics=muscle_kinematics,
        muscle_moment_arms=muscle_moment_arms,
        subject_anthro=subject_anthro,
    )

    # Synthetic ground-truth force: baseline muscle tone that ramps up with
    # trunk flexion angle, plus muscle-specific gain and noise. Gives the
    # two-stage training a real, learnable kinematics-to-force relationship.
    flexion_on_inf_time = np.interp(features.inf_time, sim_time, trunk_flexion)
    forces = np.zeros((features.inf_time.size, N_MUSCLE))
    for m in range(N_MUSCLE):
        gain = rng.uniform(3.0, 10.0)
        baseline = rng.uniform(5.0, 20.0)
        forces[:, m] = baseline + gain * flexion_on_inf_time + rng.normal(0, 3, features.inf_time.size)
    forces = forces.clip(min=0.1)

    return features, forces, subject_anthro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--trials-per-subject", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_idx = 0
    for subject_idx in range(args.n_subjects):
        subject_id = f"sub{subject_idx:02d}"
        for _ in range(args.trials_per_subject):
            features, forces, _ = make_trial(rng, subject_id)
            path = out_dir / f"trial_{trial_idx:04d}.npz"
            np.savez(
                path,
                segment_kin=features.segment_kin,
                norm_muscle_kin=features.norm_muscle_kin,
                external_moments=features.external_moments,
                muscle_moment_arms=features.muscle_moment_arms,
                muscle_csa=features.muscle_csa,
                anthro=features.anthro,
                muscle_forces=forces,
                subject_id=subject_id,
                inf_time=features.inf_time,
            )
            trial_idx += 1

    print(f"Wrote {trial_idx} synthetic trials to {out_dir}")


if __name__ == "__main__":
    main()
