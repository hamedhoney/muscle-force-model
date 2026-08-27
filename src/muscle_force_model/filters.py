"""Standalone filter-design helpers replacing the proprietary ``signals.filt`` module.

Only the flat-top FIR low-pass window used to smooth segment kinematics is
needed by this pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin


def flattop(cutoff_hz: float, fs: float, numtaps: int = 31) -> np.ndarray:
    """Design a symmetric FIR low-pass filter with a flat-top window.

    Parameters
    ----------
    cutoff_hz:
        Low-pass cutoff frequency in Hz.
    fs:
        Sampling rate of the signal to be filtered, in Hz.
    numtaps:
        Number of filter taps (kept odd for a zero-phase symmetric kernel).

    Returns
    -------
    1D array of FIR filter coefficients, normalized to unit DC gain.
    """
    if numtaps % 2 == 0:
        numtaps += 1
    nyq = fs / 2.0
    normalized_cutoff = min(max(cutoff_hz / nyq, 1e-6), 0.999)
    taps = firwin(numtaps, normalized_cutoff, window="flattop")
    return taps / np.sum(taps)
