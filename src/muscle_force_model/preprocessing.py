"""Feature extraction: raw motion-capture-style data -> fixed-size ML-ready arrays.

Same preprocessing the original ``inferMuscleForcesWithoutEmg`` script did
inline, pulled out into a function shared by training-data prep and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from muscle_force_model import filters, quat_utils
from muscle_force_model.constants import (
    DISC_NAMES,
    EULER_SEQ,
    MUSCLE_NAMES,
    N_DISC,
    N_MUSCLE,
    N_RESAMPLE,
)
from muscle_force_model.schemas import Area, Moment, MuscleKinematics

DRIFT_AXIS = np.array([0.0, 1.0, 0.0])

# Fixed rotation taking ISB (International Society of Biomechanics) segment
# coordinate conventions into this pipeline's internal X-left, Y-up, Z-forward
# convention.
_ISB_TO_INTERNAL = quat_utils.fromEulerAngles([90, 0, 0], seq="YXZ", degrees=True)


@dataclass
class FeatureBundle:
    """All model-ready arrays for a single trial, batch dimension included (B=1)."""

    inf_time: np.ndarray  # (nResample,)
    segment_kin: np.ndarray  # (1, nResample, 1 + 6*3)
    norm_muscle_kin: np.ndarray  # (1, nResample, 2*n_muscle)
    external_moments: np.ndarray  # (1, nResample, n_disc, 3)
    muscle_moment_arms: np.ndarray  # (1, nResample, n_muscle, n_disc, 3)
    muscle_csa: np.ndarray  # (1, n_muscle, 1)
    anthro: np.ndarray  # (1, 3)

    # Kept around for optional diagnostic plotting.
    pelvis_angles: np.ndarray
    chest_angles: np.ndarray
    pelvis_vel: np.ndarray
    chest_vel: np.ndarray
    pelvis_acc: np.ndarray
    chest_acc: np.ndarray


def _fix_upside_down_segments(pelvis_ori: np.ndarray, chest_ori: np.ndarray):
    """Correct a globally flipped (upside-down) rigid-body transform, which
    some upstream mocap pipelines occasionally emit. Checked against which
    way the pelvis's local up axis points in frame 0.
    """
    test_axis = quat_utils.rotateData(pelvis_ori[0:1, :], np.array([0.0, 1.0, 0.0]))[0]
    if test_axis[1] < 0:
        flip = quat_utils.fromEulerAngles([180, 0, 0], seq="XYZ", degrees=True)
        pelvis_ori = quat_utils.multiply(flip, pelvis_ori)
        chest_ori = quat_utils.multiply(flip, chest_ori)
    return pelvis_ori, chest_ori


def extract_features(
    motion_data: dict[str, Any],
    external_moments: Moment,
    muscle_areas: Area,
    muscle_kinematics: MuscleKinematics,
    muscle_moment_arms: Moment,
    subject_anthro: dict[str, float],
    n_resample: int = N_RESAMPLE,
) -> FeatureBundle:
    """Convert raw per-trial motion/moment/kinematics data into fixed-size,
    resampled arrays suitable as model input.

    Parameters
    ----------
    motion_data:
        Dict with ``"joints"`` (list with at least one entry containing
        ``"rate"`` and ``"time"``) and ``"segments"`` (list of dicts with
        ``"name"``, ``"orientationFinal"`` (T,4) quaternion array, and for the
        chest segment, ``"driftCorrection"`` (4,) quaternion).
    external_moments, muscle_moment_arms:
        ``Moment`` containers (see ``schemas.py``).
    muscle_areas:
        ``Area`` container.
    muscle_kinematics:
        ``MuscleKinematics`` container.
    subject_anthro:
        Dict with ``age_yr``, ``stature_cm``, ``mass_kg``.
    n_resample:
        Number of timepoints to resample every trial to (fixed model input length).
    """
    mot_rate = motion_data["joints"][0]["rate"]
    mot_time = np.asarray(motion_data["joints"][0]["time"], dtype=float)
    sim_time = np.asarray(muscle_kinematics.time, dtype=float)

    inf_time = np.interp(
        np.linspace(0, 1, n_resample), np.linspace(0, 1, sim_time.size), sim_time
    )

    lowpass = filters.flattop(6, fs=mot_rate)

    segments_by_name = {s["name"]: s for s in motion_data["segments"]}
    pelvis_ori = quat_utils.fillmissing(
        np.asarray(segments_by_name["Pelvis"]["orientationFinal"], dtype=float)
    )
    chest_ori = quat_utils.fillmissing(
        np.asarray(segments_by_name["Chest"]["orientationFinal"], dtype=float)
    )
    chest_ori = quat_utils.multiply(
        np.asarray(segments_by_name["Chest"]["driftCorrection"], dtype=float), chest_ori
    )

    pelvis_ori, chest_ori = _fix_upside_down_segments(pelvis_ori, chest_ori)

    pelvis_ori = quat_utils.rotateFrame(_ISB_TO_INTERNAL, pelvis_ori)
    chest_ori = quat_utils.rotateFrame(_ISB_TO_INTERNAL, chest_ori)

    # Remove pelvis yaw drift, low-pass filter, resample, convert to Euler angles.
    yaw, _swing = quat_utils.swingTwist(pelvis_ori, DRIFT_AXIS)
    pelvis_ori = quat_utils.ldivide(yaw, pelvis_ori)
    pelvis_ori = quat_utils.windowFilter(pelvis_ori, lowpass)
    pelvis_ori = quat_utils.interp(inf_time, mot_time, pelvis_ori)
    pelvis_angles = quat_utils.toEulerAngles(pelvis_ori, EULER_SEQ, degrees=True)

    chest_ori = quat_utils.ldivide(yaw, chest_ori)
    chest_ori = quat_utils.windowFilter(chest_ori, lowpass)
    chest_ori = quat_utils.interp(inf_time, mot_time, chest_ori)
    chest_angles = quat_utils.toEulerAngles(chest_ori, EULER_SEQ, degrees=True)

    dt = inf_time[1] - inf_time[0]
    chest_vel = np.diff(chest_angles, axis=0, prepend=chest_angles[0:1]) / dt
    pelvis_vel = np.diff(pelvis_angles, axis=0, prepend=pelvis_angles[0:1]) / dt
    chest_acc = np.diff(chest_vel, axis=0, prepend=chest_vel[0:1]) / dt
    pelvis_acc = np.diff(pelvis_vel, axis=0, prepend=pelvis_vel[0:1]) / dt

    segment_kin = np.concatenate(
        (
            inf_time[:, np.newaxis],
            pelvis_angles,
            chest_angles,
            pelvis_vel,
            chest_vel,
            pelvis_acc,
            chest_acc,
        ),
        axis=1,
    )[np.newaxis, ...]

    # Normalized muscle-tendon kinematics (length & velocity, normalized by neutral length).
    norm_muscle_kin = np.zeros((1, n_resample, 2 * N_MUSCLE), dtype=float)
    muscle_kin_by_name = {m.name: m for m in muscle_kinematics.muscles}
    for m, muscle_name in enumerate(MUSCLE_NAMES):
        muscle = muscle_kin_by_name[muscle_name]
        neutral_length = muscle.length[0]
        norm_muscle_kin[:, :, m] = np.interp(inf_time, sim_time, muscle.length) / neutral_length
        norm_muscle_kin[:, :, N_MUSCLE + m] = (
            np.interp(inf_time, sim_time, muscle.velocity) / neutral_length
        )

    # External moments per spinal level.
    external_moments_arr = np.zeros((1, n_resample, N_DISC, 3), dtype=float)
    contributor_by_level = {c.level: c for c in external_moments.contributors}
    for level_idx, level in enumerate(DISC_NAMES):
        contributor = contributor_by_level[level]
        for plane in range(3):
            external_moments_arr[:, :, level_idx, plane] = np.interp(
                inf_time, external_moments.time, contributor.moment[:, plane]
            )

    # Muscle moment arms per (muscle, level).
    muscle_moment_arms_arr = np.zeros((1, n_resample, N_MUSCLE, N_DISC, 3), dtype=float)
    moment_arms_by_key = {f"{c.level}|{c.name}": c for c in muscle_moment_arms.contributors}
    for m, muscle_name in enumerate(MUSCLE_NAMES):
        for level_idx, level in enumerate(DISC_NAMES):
            contributor = moment_arms_by_key[f"{level}|{muscle_name}"]
            for plane in range(3):
                muscle_moment_arms_arr[:, :, m, level_idx, plane] = np.interp(
                    inf_time, muscle_moment_arms.time, contributor.moment[:, plane]
                )

    # Muscle cross-sectional area.
    muscle_csa = np.zeros((1, N_MUSCLE, 1))
    muscle_area_by_name = {m.name: m for m in muscle_areas.muscleArea}
    for m, muscle_name in enumerate(MUSCLE_NAMES):
        muscle_csa[:, m, 0] = muscle_area_by_name[muscle_name].max

    # Anthropometry.
    anthro = np.array(
        [
            subject_anthro["age_yr"],
            subject_anthro["stature_cm"],
            subject_anthro["mass_kg"],
        ]
    )[np.newaxis, ...]

    return FeatureBundle(
        inf_time=inf_time,
        segment_kin=segment_kin,
        norm_muscle_kin=norm_muscle_kin,
        external_moments=external_moments_arr,
        muscle_moment_arms=muscle_moment_arms_arr,
        muscle_csa=muscle_csa,
        anthro=anthro,
        pelvis_angles=pelvis_angles,
        chest_angles=chest_angles,
        pelvis_vel=pelvis_vel,
        chest_vel=chest_vel,
        pelvis_acc=pelvis_acc,
        chest_acc=chest_acc,
    )
