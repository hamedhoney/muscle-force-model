import numpy as np

from muscle_force_model.constants import DISC_NAMES, MUSCLE_NAMES, N_DISC, N_MUSCLE, N_RESAMPLE
from muscle_force_model.preprocessing import extract_features
from muscle_force_model.schemas import (
    Area,
    Moment,
    MomentContributor,
    MuscleAreaEntry,
    MuscleKinematics,
    MuscleKinematicsEntry,
)


def _make_trivial_trial(n_samples: int = 64):
    mot_time = np.linspace(0, 1, n_samples)
    identity_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_samples, 1))

    motion_data = {
        "joints": [{"rate": n_samples, "time": mot_time}],
        "segments": [
            {"name": "Pelvis", "orientationFinal": identity_quat},
            {
                "name": "Chest",
                "orientationFinal": identity_quat,
                "driftCorrection": np.array([1.0, 0.0, 0.0, 0.0]),
            },
        ],
    }

    sim_time = np.linspace(0, 1, n_samples)
    muscle_kinematics = MuscleKinematics(
        time=sim_time,
        muscles=[
            MuscleKinematicsEntry(name=name, length=np.full(n_samples, 0.3), velocity=np.zeros(n_samples))
            for name in MUSCLE_NAMES
        ],
    )
    external_moments = Moment(
        time=sim_time,
        contributors=[
            MomentContributor(level=level, moment=np.zeros((n_samples, 3))) for level in DISC_NAMES
        ],
    )
    muscle_moment_arms = Moment(
        time=sim_time,
        contributors=[
            MomentContributor(level=level, moment=np.zeros((n_samples, 3)), name=name)
            for level in DISC_NAMES
            for name in MUSCLE_NAMES
        ],
    )
    muscle_areas = Area(muscleArea=[MuscleAreaEntry(name=name, max=5.0) for name in MUSCLE_NAMES])
    subject_anthro = {"age_yr": 30.0, "stature_cm": 175.0, "mass_kg": 75.0}

    return motion_data, external_moments, muscle_areas, muscle_kinematics, muscle_moment_arms, subject_anthro


def test_extract_features_shapes():
    motion_data, ext_mom, areas, kin, mom_arms, anthro = _make_trivial_trial()
    features = extract_features(motion_data, ext_mom, areas, kin, mom_arms, anthro)

    assert features.segment_kin.shape == (1, N_RESAMPLE, 1 + 6 * 3)
    assert features.norm_muscle_kin.shape == (1, N_RESAMPLE, 2 * N_MUSCLE)
    assert features.external_moments.shape == (1, N_RESAMPLE, N_DISC, 3)
    assert features.muscle_moment_arms.shape == (1, N_RESAMPLE, N_MUSCLE, N_DISC, 3)
    assert features.muscle_csa.shape == (1, N_MUSCLE, 1)
    assert features.anthro.shape == (1, 3)


def test_extract_features_static_trial_has_near_zero_kinematics():
    motion_data, ext_mom, areas, kin, mom_arms, anthro = _make_trivial_trial()
    features = extract_features(motion_data, ext_mom, areas, kin, mom_arms, anthro)

    np.testing.assert_allclose(features.pelvis_angles, 0.0, atol=1e-3)
    np.testing.assert_allclose(features.chest_angles, 0.0, atol=1e-3)
    np.testing.assert_allclose(features.norm_muscle_kin[0, :, :N_MUSCLE], 1.0, atol=1e-6)
