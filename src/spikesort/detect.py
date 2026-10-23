"""Threshold-based spike detection and waveform extraction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .preprocess import mad_noise


def detect_spikes(
    data: np.ndarray,
    fs: float,
    threshold: float = 5.0,
    dead_time_ms: float = 1.0,
    align_window_ms: float = 0.5,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Detect negative threshold crossings on any channel.

    Each channel is normalised by its MAD noise, so `threshold` is in units
    of sigma. A spike is an event where the minimum across channels drops
    below -threshold. Events are aligned to the trough within
    `align_window_ms`, and events closer than `dead_time_ms` are merged,
    keeping the deepest one.

    Returns sorted sample indices of spike troughs.
    """
    if noise is None:
        noise = mad_noise(data)
    z = data / noise[None, :]
    zmin = z.min(axis=1)

    below = zmin < -threshold
    # Rising edges of the boolean mask are the crossing onsets.
    onsets = np.flatnonzero(below[1:] & ~below[:-1]) + 1
    if below[0]:
        onsets = np.r_[0, onsets]

    win = max(1, int(round(align_window_ms * 1e-3 * fs)))
    n = zmin.size
    troughs = np.empty(onsets.size, dtype=np.int64)
    for i, o in enumerate(onsets):
        stop = min(n, o + win)
        troughs[i] = o + int(np.argmin(zmin[o:stop]))

    return _enforce_dead_time(troughs, zmin, int(round(dead_time_ms * 1e-3 * fs)))


def _enforce_dead_time(times: np.ndarray, zmin: np.ndarray, dead: int) -> np.ndarray:
    """Greedy: walk events by depth, drop anything within `dead` of a kept event."""
    if times.size == 0:
        return times
    times = np.unique(times)
    order = np.argsort(zmin[times])  # deepest first
    kept = np.zeros(times.size, dtype=bool)
    taken = np.zeros(times.size, dtype=bool)
    for idx in order:
        if taken[idx]:
            continue
        kept[idx] = True
        lo = np.searchsorted(times, times[idx] - dead, side="left")
        hi = np.searchsorted(times, times[idx] + dead, side="right")
        taken[lo:hi] = True
    return times[kept]


def extract_waveforms(
    data: np.ndarray,
    times: np.ndarray,
    fs: float,
    pre_ms: float = 0.6,
    post_ms: float = 1.2,
    align: str = "com",
    align_window_ms: float = 0.3,
    return_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a snippet around each spike, aligned to the trough.

    Alignment modes:

    * ``"com"`` (default): centre on the trough's centre of mass. On the
      spike's deepest channel, take the samples within `align_window_ms` of
      the detected trough, keep the part deeper than 30% of the trough depth,
      and use their depth-weighted mean position. The snippet is then read
      off a cubic spline at that fractional position.
    * ``"parabola"``: refine the minimum sample with a parabola through its
      neighbours (sub-sample, but only corrects within half a sample).
    * ``"none"``: integer sample of the detected minimum.

    Alignment turned out to be the single biggest lever on accuracy (see the
    README). With integer alignment, noise decides which of two samples
    holds the minimum, so each neuron becomes two populations one sample
    apart and PCA separates them. Parabolic refinement fixes that at high
    SNR, but for smaller spikes the noisy minimum wanders several samples
    along a broad trough. The centre of mass averages over the whole trough,
    so noise mostly cancels.

    Returns (waveforms, aligned_times) where waveforms has shape
    (n_spikes, n_samples, n_channels). Spikes too close to the edges are
    dropped; with `return_mask` the boolean mask of kept input spikes is
    returned as a third value.
    """
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    w = max(1, int(round(align_window_ms * 1e-3 * fs)))
    margin = w + 2
    ok = (times - pre - margin >= 0) & (times + post + margin <= data.shape[0])
    times = times[ok]
    offsets = np.arange(-pre, post)
    if align == "none" or times.size == 0:
        idx = times[:, None] + offsets[None, :]
        return (data[idx], times, ok) if return_mask else (data[idx], times)

    ch = np.argmin(data[times], axis=1)
    if align == "parabola":
        ym1, y0, yp1 = data[times - 1, ch], data[times, ch], data[times + 1, ch]
        denom = ym1 - 2 * y0 + yp1
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(denom > 0, 0.5 * (ym1 - yp1) / denom, 0.0)
        delta = np.clip(delta, -0.5, 0.5)
    elif align == "com":
        win = np.arange(-w, w + 1)
        seg = -data[times[:, None] + win[None, :], ch[:, None]]  # trough is positive here
        depth = seg.max(axis=1, keepdims=True)
        weight = np.clip(seg - 0.3 * depth, 0, None)
        total = weight.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(total > 0, (weight * win[None, :]).sum(axis=1) / total, 0.0)
    else:
        raise ValueError(f"unknown align mode {align!r}")

    centre = times + delta
    coords = centre[:, None] + offsets[None, :]
    out = np.empty((times.size, offsets.size, data.shape[1]), dtype=np.float32)
    for c in range(data.shape[1]):
        out[:, :, c] = ndimage.map_coordinates(
            data[:, c].astype(np.float64), [coords.ravel()], order=3, mode="nearest"
        ).reshape(coords.shape)
    aligned = np.round(centre).astype(np.int64)
    return (out, aligned, ok) if return_mask else (out, aligned)
