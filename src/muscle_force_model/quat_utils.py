"""Quaternion utilities for orientation preprocessing.

Replaces the proprietary ``signals.quat`` module used in the original
script. Built on ``scipy.spatial.transform.Rotation`` / ``Slerp``.

Quaternions here are scalar-first: ``[w, x, y, z]``, shape ``(..., 4)``.
Scipy uses scalar-last ``[x, y, z, w]``, so we convert at the boundary.
Swap this module for your own orientation code if you have one.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_EPS = 1e-12


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, _EPS, None)


def fillmissing(q: np.ndarray) -> np.ndarray:
    """Linearly fill NaN quaternion frames (mocap dropout) and renormalize.

    Fills each of the 4 components independently, which is an
    approximation, but is fine for short dropout gaps before filtering.
    """
    q = q.copy().astype(float)
    n = q.shape[0]
    idx = np.arange(n)
    for c in range(4):
        col = q[:, c]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            raise ValueError("Quaternion component is entirely NaN; cannot fill.")
        if nan_mask.any():
            col[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], col[~nan_mask])
            q[:, c] = col
    return _normalize(q)


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, broadcastable over leading dimensions."""
    r1 = Rotation.from_quat(_wxyz_to_xyzw(np.broadcast_to(q1, np.broadcast_shapes(q1.shape, q2.shape))))
    r2 = Rotation.from_quat(_wxyz_to_xyzw(np.broadcast_to(q2, np.broadcast_shapes(q1.shape, q2.shape))))
    return _xyzw_to_wxyz((r1 * r2).as_quat())


def conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (== inverse for unit quaternions)."""
    out = q.copy()
    out[..., 1:4] *= -1
    return out


def ldivide(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Left division: relative rotation from q1 to q2, i.e. conj(q1) (x) q2."""
    return multiply(conjugate(q1), q2)


def rotateData(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector(s) ``v`` by quaternion(s) ``q`` (broadcastable)."""
    r = Rotation.from_quat(_wxyz_to_xyzw(q))
    return r.apply(v)


def rotateFrame(q_transform: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Re-express orientation quaternion(s) ``q`` in a new reference frame.

    ``q_transform`` is the fixed rotation between old and new frame.
    q' = q_transform (x) q (x) conj(q_transform).
    """
    q_t = np.broadcast_to(q_transform, q.shape)
    return multiply(multiply(q_t, q), conjugate(q_t))


def fromEulerAngles(angles, seq: str, degrees: bool = True) -> np.ndarray:
    """Build quaternion(s) from Euler angles (any scipy-supported sequence)."""
    angles = np.asarray(angles, dtype=float)
    r = Rotation.from_euler(seq, angles, degrees=degrees)
    q = r.as_quat()
    if angles.ndim == 1:
        q = q[np.newaxis, :] if q.ndim == 1 else q
        return _xyzw_to_wxyz(q)[0] if q.shape[0] == 1 else _xyzw_to_wxyz(q)
    return _xyzw_to_wxyz(q)


def toEulerAngles(q: np.ndarray, seq: str, degrees: bool = True) -> np.ndarray:
    """Convert quaternion array (T, 4) to Euler angles (T, 3) for ``seq``."""
    r = Rotation.from_quat(_wxyz_to_xyzw(q))
    return r.as_euler(seq, degrees=degrees)


def swingTwist(q: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Swing-twist decomposition of q about a fixed unit ``axis``.

    Returns ``(twist, swing)`` such that ``q ~= swing (x) twist``, where
    ``twist`` is the rotation about ``axis`` (e.g. yaw about a vertical
    axis) and ``swing`` is the remaining rotation.
    """
    axis = axis / (np.linalg.norm(axis) + _EPS)
    w = q[..., 0:1]
    v = q[..., 1:4]
    proj = np.sum(v * axis, axis=-1, keepdims=True) * axis
    twist = np.concatenate([w, proj], axis=-1)
    twist = _normalize(twist)
    swing = multiply(q, conjugate(twist))
    return twist, swing


def interp(new_time: np.ndarray, old_time: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Spherical (SLERP) interpolation of an orientation quaternion time series."""
    order = np.argsort(old_time)
    old_time_sorted = old_time[order]
    q_sorted = q[order]
    # Slerp requires strictly increasing keyframe times.
    keep = np.concatenate([[True], np.diff(old_time_sorted) > 0])
    r = Rotation.from_quat(_wxyz_to_xyzw(q_sorted[keep]))
    slerp = Slerp(old_time_sorted[keep], r)
    new_time_clipped = np.clip(new_time, old_time_sorted[keep][0], old_time_sorted[keep][-1])
    return _xyzw_to_wxyz(slerp(new_time_clipped).as_quat())


def windowFilter(q: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Smooth a quaternion time series with a symmetric FIR ``kernel``.

    Enforces sign continuity first (unit quaternions have a double cover,
    so naive filtering can blend antipodal representations), then convolves
    each component and renormalizes. Approximate, but fine for the smooth,
    slow-changing signals typical of motion capture.
    """
    q = q.copy()
    for t in range(1, q.shape[0]):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]

    kernel = kernel / np.sum(kernel)
    pad = len(kernel) // 2
    filtered = np.zeros_like(q)
    for c in range(4):
        padded = np.pad(q[:, c], (pad, pad), mode="edge")
        filtered[:, c] = np.convolve(padded, kernel, mode="valid")[: q.shape[0]]
    return _normalize(filtered)
