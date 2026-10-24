"""Spike detection and waveform extraction."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .preprocess import mad_noise


def smooth_for_detection(data: np.ndarray, fs: float, width_ms: float = 0.12) -> np.ndarray:
    """Correlate each channel with a unit-norm Gaussian pulse of the given width.

    A generic matched filter: the pulse roughly matches a spike trough, so
    the output's signal-to-noise ratio is about two to three times that of
    a single sample, while the noise stays zero-mean. Zero phase, so
    troughs do not move.
    """
    n = int(round(4 * width_ms * 1e-3 * fs)) | 1
    t = (np.arange(n) - n // 2) / fs * 1e3
    g = np.exp(-0.5 * (t / width_ms) ** 2)
    g /= np.linalg.norm(g)
    return signal.fftconvolve(data, g[:, None], mode="same", axes=0).astype(np.float32)


def detect_spikes(
    data: np.ndarray,
    fs: float,
    threshold: float = 5.0,
    dead_time_ms: float = 2.0,
    align_window_ms: float = 0.5,
    noise: np.ndarray | None = None,
    method: str = "energy",
    smooth_ms: float = 0.12,
    channel_min: float = 4.5,
) -> np.ndarray:
    """Detect spikes as negative deflections, on any channel.

    ``method="threshold"``: the classic detector. Each channel is divided
    by its MAD noise and an event is any sample where the minimum across
    channels drops below -threshold.

    ``method="energy"`` (default): each channel is first smoothed with a
    generic spike-shaped pulse (`smooth_for_detection`) and normalised by
    the MAD of the result, which gains two to three times in
    signal-to-noise. The negative deflections are then combined across
    channels as sqrt(sum of squares), and an event needs that combined
    score above `threshold` and at least one channel above `channel_min`.
    The per-channel floor matters: the band-pass filter leaves a shallow
    undershoot 1 to 2 ms after every large spike, coherent across channels,
    and the combined score alone fires on it.

    Events are placed at the trough (or score peak) within `align_window_ms`
    of the crossing, and events closer than `dead_time_ms` are merged,
    keeping the deepest. Two milliseconds is deliberate for stage one: it
    also suppresses what is left of the undershoot. Overlapping spikes are
    stage two's job.

    Returns sorted sample indices.
    """
    if method == "threshold":
        if noise is None:
            noise = mad_noise(data)
        score = -(data / noise[None, :]).min(axis=1)
        above = score > threshold
    elif method == "energy":
        y = smooth_for_detection(data, fs, smooth_ms)
        z = -y / mad_noise(y)[None, :]
        score = np.sqrt((np.clip(z, 0, None) ** 2).sum(axis=1))
        above = (score > threshold) & (z.max(axis=1) > channel_min)
    else:
        raise ValueError(f"unknown detection method {method!r}")

    # Rising edges of the boolean mask are the crossing onsets.
    onsets = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if above[0]:
        onsets = np.r_[0, onsets]

    win = max(1, int(round(align_window_ms * 1e-3 * fs)))
    n = score.size
    peaks = np.empty(onsets.size, dtype=np.int64)
    for i, o in enumerate(onsets):
        peaks[i] = o + int(np.argmax(score[o:min(n, o + win)]))

    return _enforce_dead_time(peaks, -score, int(round(dead_time_ms * 1e-3 * fs)))


def detect_spikes_combined(
    data: np.ndarray,
    fs: float,
    threshold: float = 5.0,
    noise: np.ndarray | None = None,
    threshold_dead_ms: float = 1.0,
    energy_dead_ms: float = 2.0,
    merge_ms: float = 1.0,
) -> np.ndarray:
    """Union of the threshold and energy detectors.

    The threshold detector with a 1 ms dead time keeps collision partners
    for the clusterer to see. The energy detector with a 2 ms dead time
    finds small spikes while suppressing the filter undershoot after big
    ones. Energy events within `merge_ms` of a threshold event are
    dropped as duplicates; the rest are added.
    """
    a = detect_spikes(data, fs, threshold, threshold_dead_ms, noise=noise, method="threshold")
    b = detect_spikes(data, fs, threshold, energy_dead_ms, noise=noise, method="energy")
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    gap = int(round(merge_ms * 1e-3 * fs))
    i = np.clip(np.searchsorted(a, b), 1, a.size - 1)
    near = np.minimum(np.abs(a[i] - b), np.abs(a[i - 1] - b)) <= gap
    return np.union1d(a, b[~near])


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
