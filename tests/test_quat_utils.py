import numpy as np

from muscle_force_model import quat_utils


def test_euler_roundtrip():
    angles = np.array([[10.0, -20.0, 5.0], [0.0, 45.0, 0.0]])
    q = quat_utils.fromEulerAngles(angles, seq="YZX", degrees=True)
    back = quat_utils.toEulerAngles(q, seq="YZX", degrees=True)
    np.testing.assert_allclose(back, angles, atol=1e-4)


def test_multiply_identity():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = quat_utils.fromEulerAngles([30.0, 0.0, 0.0], seq="XYZ", degrees=True)
    result = quat_utils.multiply(identity, q)
    np.testing.assert_allclose(result, q, atol=1e-6)


def test_ldivide_self_is_identity():
    q = quat_utils.fromEulerAngles(np.array([[15.0, 5.0, -10.0]]), seq="XYZ", degrees=True)
    result = quat_utils.ldivide(q, q)
    np.testing.assert_allclose(np.abs(result[0]), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_fillmissing_fills_nan():
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
    q[2, :] = np.nan
    filled = quat_utils.fillmissing(q)
    assert not np.isnan(filled).any()
    np.testing.assert_allclose(np.linalg.norm(filled, axis=-1), 1.0, atol=1e-6)


def test_interp_matches_endpoints():
    old_time = np.array([0.0, 1.0])
    q = np.stack(
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            quat_utils.fromEulerAngles([90.0, 0.0, 0.0], seq="XYZ", degrees=True),
        ]
    )
    new_time = np.array([0.0, 1.0])
    interpolated = quat_utils.interp(new_time, old_time, q)
    np.testing.assert_allclose(np.abs(interpolated), np.abs(q), atol=1e-6)


def test_swing_twist_reconstructs_quaternion():
    axis = np.array([0.0, 1.0, 0.0])
    q = quat_utils.fromEulerAngles(np.array([[20.0, 30.0, 10.0]]), seq="YZX", degrees=True)
    twist, swing = quat_utils.swingTwist(q, axis)
    reconstructed = quat_utils.multiply(swing, twist)
    np.testing.assert_allclose(np.abs(reconstructed), np.abs(q), atol=1e-5)


def test_window_filter_preserves_unit_norm():
    from muscle_force_model.filters import flattop

    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (50, 1))
    kernel = flattop(6, fs=100)
    filtered = quat_utils.windowFilter(q, kernel)
    np.testing.assert_allclose(np.linalg.norm(filtered, axis=-1), 1.0, atol=1e-6)
