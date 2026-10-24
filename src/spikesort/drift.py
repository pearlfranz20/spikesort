"""Drift: estimate how far spikes moved along the probe over time, and undo it.

Probe motion changes which channels see each neuron. On a linear array the
footprint slides along the shank, so a unit's spikes drift in depth, its
stage-one cluster smears or splits, and a fixed template stops matching.
Two steps:

1. Estimate. Every detected spike gets a depth (power-weighted centre of
   mass of its per-channel trough amplitudes) and an amplitude. The
   recording is cut into time bins, each bin becomes an amplitude-weighted
   depth histogram, every pair of bins is registered by cross-correlation,
   and the pairwise shifts are reconciled into one displacement per bin by
   least squares (decentralised registration, Varol et al. 2021).

2. Correct. For each sample the channels are re-interpolated along the
   shank by the estimated displacement with a Gaussian-kernel (kriging)
   interpolant, as in Kilosort 2.5. The sorter then runs on data in which
   every neuron stays put.

Nothing here knows about ground truth: it works from detected spikes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class DriftEstimate:
    bin_centres_s: np.ndarray  # time of each bin centre
    displacement_um: np.ndarray  # estimated spike displacement per bin, mean zero
    n_spikes: np.ndarray  # spikes per bin
    pair_weights: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    depths: np.ndarray = field(default_factory=lambda: np.empty(0))  # per detected spike
    amplitudes: np.ndarray = field(default_factory=lambda: np.empty(0))
    times_s: np.ndarray = field(default_factory=lambda: np.empty(0))

    def at(self, t_s: np.ndarray) -> np.ndarray:
        """Displacement interpolated to arbitrary times, constant beyond the ends."""
        return np.interp(t_s, self.bin_centres_s, self.displacement_um)


def spike_depths(waveforms: np.ndarray, channel_positions: np.ndarray,
                 axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position of each spike along `axis`, and its amplitude.

    Each channel's amplitude is read at the spike's trough (the deepest
    sample on any channel, averaged with its two neighbours), not as the
    channel's own minimum over the window: the minimum of a noise-only
    channel is always a few sigma negative, which biases every spike
    towards the middle of the array and adds noise. Depth is the centre
    of mass of the squared amplitudes, so the nearest channel dominates.
    Amplitude is the trough on the deepest channel.

    The centre of mass compresses motion a little (a unit that moves 20
    microns reads as about 18 on a 20 micron pitch array) and is unreliable
    for units beyond the ends of the array, where it can even move the
    wrong way. Both are properties of the estimator, not of the sorter.
    """
    along = channel_positions @ axis  # (n_ch,)
    n, n_t, _ = waveforms.shape
    trough_idx = np.argmin(waveforms.min(axis=2), axis=1)  # (n,)
    lo = np.clip(trough_idx - 1, 0, n_t - 1)
    hi = np.clip(trough_idx + 1, 0, n_t - 1)
    rows = np.arange(n)
    at_trough = (waveforms[rows, lo] + waveforms[rows, trough_idx] + waveforms[rows, hi]) / 3.0
    amp = np.clip(-at_trough, 0, None).astype(np.float64)  # (n, n_ch)
    power = amp ** 2
    total = power.sum(axis=1)
    total[total == 0] = 1.0
    depth = (power * along[None, :]).sum(axis=1) / total
    return depth, amp.max(axis=1)


def _register(h_a: np.ndarray, h_b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag (bins) that best aligns h_b onto h_a, and the normalised peak correlation.

    Positive lag means h_b is h_a shifted towards higher depth.
    """
    n = h_a.size
    full = np.correlate(h_b, h_a, mode="full")  # index n-1 is zero lag
    lo, hi = max(0, n - 1 - max_lag), min(full.size, n + max_lag)
    seg = full[lo:hi]
    i = int(np.argmax(seg))
    lag = float(i + lo - (n - 1))
    peak = float(seg[i])
    if 0 < i < seg.size - 1:
        cm, c0, cp = seg[i - 1], seg[i], seg[i + 1]
        den = cm - 2 * c0 + cp
        if den < 0:
            lag += float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
    norm = np.linalg.norm(h_a) * np.linalg.norm(h_b)
    return lag, peak / norm if norm > 0 else 0.0


def estimate_drift(
    times_s: np.ndarray,
    depths: np.ndarray,
    amplitudes: np.ndarray,
    duration_s: float,
    bin_s: float = 2.0,
    depth_step_um: float = 1.0,
    smooth_um: float = 4.0,
    max_shift_um: float = 60.0,
    min_spikes: int = 10,
    min_corr: float = 0.2,
) -> DriftEstimate:
    """Estimate spike displacement along the probe as a function of time.

    Each time bin's spikes become a depth histogram, weighted by amplitude
    and smoothed by `smooth_um`. Every pair of bins with enough spikes is
    registered by cross-correlation (sub-bin precision from a parabola on
    the peak), giving a relative shift and a confidence (the normalised
    peak). The per-bin displacements that best explain all pairwise shifts
    are found by weighted least squares, anchored to mean zero.

    Registering all pairs rather than each bin against its predecessor
    stops errors from accumulating along the recording, and pairs of bins
    that share no units simply get a low weight.
    """
    n_bins = max(1, int(np.ceil(duration_s / bin_s)))
    edges = np.arange(n_bins + 1) * bin_s
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip((times_s / bin_s).astype(int), 0, n_bins - 1)

    grid_lo = depths.min() - max_shift_um
    grid_hi = depths.max() + max_shift_um
    grid = np.arange(grid_lo, grid_hi + depth_step_um, depth_step_um)
    hist = np.zeros((n_bins, grid.size - 1))
    counts = np.bincount(which, minlength=n_bins)
    for b in range(n_bins):
        m = which == b
        if m.any():
            h, _ = np.histogram(depths[m], bins=grid, weights=np.sqrt(amplitudes[m]))
            hist[b] = ndimage.gaussian_filter1d(h, smooth_um / depth_step_um)

    valid = np.flatnonzero(counts >= min_spikes)
    max_lag = int(round(max_shift_um / depth_step_um))
    rows, rhs, weights = [], [], []
    pair_w = np.zeros((n_bins, n_bins))
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            lag, corr = _register(hist[a], hist[b], max_lag)
            if corr < min_corr:
                continue
            row = np.zeros(n_bins)
            row[b], row[a] = 1.0, -1.0
            rows.append(row)
            rhs.append(lag * depth_step_um)
            weights.append(corr)
            pair_w[a, b] = pair_w[b, a] = corr

    disp = np.zeros(n_bins)
    if rows:
        a_mat = np.array(rows) * np.sqrt(weights)[:, None]
        b_vec = np.array(rhs) * np.sqrt(weights)
        # Weak ridge towards zero fixes the gauge (only differences are observed).
        a_mat = np.vstack([a_mat, 1e-3 * np.eye(n_bins)])
        b_vec = np.concatenate([b_vec, np.zeros(n_bins)])
        disp = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
        # Bins without data take the value of the nearest estimated bin.
        if valid.size and valid.size < n_bins:
            disp = np.interp(np.arange(n_bins), valid, disp[valid])
        disp -= disp.mean()
    return DriftEstimate(centres, disp, counts, pair_w, depths, amplitudes, times_s)


def shift_channels(
    x: np.ndarray,
    channel_positions: np.ndarray,
    displacement_um: np.ndarray,
    axis: np.ndarray,
    kernel_um: float | None = None,
    step_um: float = 0.25,
) -> np.ndarray:
    """Re-sample every channel at its own position plus the displacement.

    Spikes that drifted by +d along `axis` are brought back by reading each
    channel's value from the point d further along, interpolated from the
    real channels with a Gaussian kernel (kriging): the interpolation
    weights are K(P + d, P) K(P, P)^-1. The displacement is quantised to
    `step_um` so only a few hundred weight matrices are needed, and samples
    sharing one are shifted in a single matrix product. Positions beyond
    the array get extrapolated with decaying weights, so edge channels fade
    rather than wrap.
    """
    pos = np.asarray(channel_positions, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if kernel_um is None:
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        kernel_um = float(np.median(d.min(axis=1)))

    def kern(p, q):
        d2 = ((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / kernel_um ** 2)

    k_inv = np.linalg.inv(kern(pos, pos) + 1e-6 * np.eye(pos.shape[0]))
    steps = np.round(displacement_um / step_um).astype(np.int64)
    out = np.empty_like(x)
    for q in np.unique(steps):
        m = steps == q
        w = kern(pos + q * step_um * axis[None, :], pos) @ k_inv  # (n_ch, n_ch)
        out[m] = x[m] @ w.T
    return out
