"""Filtering and noise estimation."""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(data: np.ndarray, fs: float, low: float = 300.0, high: float = 6000.0,
             order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass along time (axis 0).

    Zero phase matters here: a causal filter would shift and distort the
    spike shape, which then leaks into the features.
    """
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0).astype(np.float32)


def common_median_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the across-channel median at each sample.

    Removes noise shared by all channels. Only useful with enough channels
    that one unit's spike does not dominate the median (roughly 8 or more).
    """
    return data - np.median(data, axis=1, keepdims=True)


def mad_noise(data: np.ndarray) -> np.ndarray:
    """Robust per-channel noise estimate, sigma = median(|x|) / 0.6745.

    Quiroga et al. 2004. Unlike the standard deviation, this barely moves
    when a channel has a lot of large spikes.
    """
    return np.median(np.abs(data), axis=0) / 0.6745
