"""Stage two: template matching by greedy matching pursuit.

Stage one (detect, align, cluster) assigns each detected event to one unit.
When two neurons fire within a millisecond their waveforms add, the sum
matches neither template, and the event ends up in a junk cluster or the
wrong unit. Stage two goes back to the filtered trace with the templates
from stage one and explains it as a sum of shifted, scaled templates:

    repeat
        for every template k and start time t, how much would subtracting
        a_k T_k at t reduce the residual energy?
        take the best non-overlapping set of (k, t), refine each to a
        fraction of a sample, subtract, update the residual
    until nothing left reduces the energy by more than `threshold` of a
    template's own energy

Because the subtraction removes the first spike of a collision, the second
one is exposed on the next round. This is the same idea as the
deconvolution step in Kilosort, kept simple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, ndimage

from .detect import _enforce_dead_time


@dataclass
class MatchResult:
    """Output of `match_templates`."""

    times: np.ndarray  # trough sample of each fitted spike, sorted
    labels: np.ndarray  # index into the template list
    amplitudes: np.ndarray  # scale applied to the template
    starts: np.ndarray  # integer sample where the fitted template begins
    shifts: np.ndarray  # fractional shift of the template (samples)
    residual: np.ndarray  # trace after subtracting every fitted template
    rounds: int  # rounds that fitted at least one spike


def select_templates(
    waveforms: np.ndarray,
    labels: np.ndarray,
    noise: np.ndarray,
    min_spikes: int = 10,
    outlier_ratio: float = 1.5,
    max_outlier_frac: float = 0.5,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Decide which stage-one clusters are real units worth a template.

    A cluster made of collisions has members that each contain some other
    neuron's spike, so the residual after subtracting the cluster's own
    median template is far above the noise floor. For a clean unit the
    residual energy is about n_samples * sum(noise**2) per spike, with a
    spread of roughly 10%. A member is an outlier if its residual energy
    exceeds `outlier_ratio` times that floor, and a cluster is dropped if
    more than `max_outlier_frac` of its members are outliers, or if it has
    fewer than `min_spikes` spikes.

    Returns (kept, dropped): kept maps cluster id to its template, dropped
    maps cluster id to its spike count. Falls back to keeping everything if
    the test would drop every cluster.
    """
    floor = waveforms.shape[1] * float((noise.astype(np.float64) ** 2).sum())
    kept: dict[int, np.ndarray] = {}
    dropped: dict[int, int] = {}
    for k in np.unique(labels):
        w = waveforms[labels == k].astype(np.float64)
        if w.shape[0] < min_spikes:
            dropped[int(k)] = int(w.shape[0])
            continue
        tmpl = np.median(w, axis=0)
        resid = ((w - tmpl) ** 2).sum(axis=(1, 2))
        if (resid > outlier_ratio * floor).mean() > max_outlier_frac:
            dropped[int(k)] = int(w.shape[0])
            continue
        kept[int(k)] = tmpl
    if not kept:
        kept = {int(k): np.median(waveforms[labels == k], axis=0) for k in np.unique(labels)}
        dropped = {}
    return kept, dropped


def shifted_template(template: np.ndarray, shift: float) -> np.ndarray:
    """The template delayed by `shift` samples: out[i] = template(i - shift).

    Cubic spline, so a spike that starts between two samples can be
    subtracted where it actually is. Subtracting at the nearest integer
    sample instead leaves a residual shaped like the template's derivative,
    which is large for a sharp spike and gets re-fitted as garbage.
    """
    n_t, n_ch = template.shape
    coords = np.arange(n_t, dtype=np.float64) - shift
    out = np.empty((n_t, n_ch), dtype=np.float64)
    for c in range(n_ch):
        out[:, c] = ndimage.map_coordinates(template[:, c].astype(np.float64), [coords],
                                            order=3, mode="nearest")
    return out


class ShiftBank:
    """Pre-shifted copies of each template on a fine grid of sub-sample shifts.

    `shifted_template` costs a spline evaluation per channel. Thousands of
    spikes times several refinement passes made that the bottleneck, so the
    shifts are quantised to `steps` points across one sample and looked up.
    With 20 steps the worst quantisation error is 0.025 samples.
    """

    def __init__(self, templates: np.ndarray, steps: int = 20):
        self.steps = steps
        self.grid = np.linspace(-0.5, 0.5, steps + 1)
        self.bank = np.stack([
            np.stack([shifted_template(t, g) for g in self.grid]) for t in templates
        ])  # (K, steps + 1, n_t, n_ch)

    def __call__(self, k: int, shift: float) -> np.ndarray:
        i = int(round((float(np.clip(shift, -0.5, 0.5)) + 0.5) * self.steps))
        return self.bank[k, i]


def correlate_templates(x: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """C[k, t] = sum over samples i and channels c of x[t + i, c] * T[k, i, c].

    One FFT of the trace per channel, reused for every template.
    """
    n, n_ch = x.shape
    k_n, n_t, _ = templates.shape
    n_valid = n - n_t + 1
    nfft = fft.next_fast_len(n + n_t - 1)
    spectrum = fft.rfft(x, n=nfft, axis=0)
    out = np.empty((k_n, n_valid), dtype=x.dtype)
    for k in range(k_n):
        kernel = fft.rfft(templates[k, ::-1, :].astype(x.dtype), n=nfft, axis=0)
        full = fft.irfft((spectrum * kernel).sum(axis=1), n=nfft)
        out[k] = full[n_t - 1:n]
    return out


def _recompute(corr: np.ndarray, residual: np.ndarray, flat_templates: np.ndarray,
               lo: int, hi: int) -> None:
    """Recompute corr[:, lo:hi] directly from the residual.

    `flat_templates` is (K, n_t * n_ch) with the (sample, channel) order
    of the residual, so each window flattens to match and the update is one
    matrix product.
    """
    n_t = flat_templates.shape[1] // residual.shape[1]
    windows = sliding_window_view(residual[lo:hi + n_t - 1], n_t, axis=0)  # (hi - lo, n_ch, n_t)
    flat = windows.transpose(0, 2, 1).reshape(hi - lo, -1)
    corr[:, lo:hi] = (flat @ flat_templates.T).T


def match_templates(
    x: np.ndarray,
    templates: dict[int, np.ndarray] | list[np.ndarray],
    amp_range: tuple[float, float] = (0.8, 1.25),
    threshold: float = 0.1,
    min_z: float = 5.0,
    max_rounds: int = 10,
) -> MatchResult:
    """Explain `x` (n_samples, n_channels) as a sum of shifted, scaled templates.

    Parameters
    ----------
    templates
        Templates of equal length, shape (n_t, n_channels) each. Labels in
        the result index this list (or the sorted keys of the dict).
    amp_range
        Allowed scale factor for each fitted template. Tight limits stop a
        large template from shrinking to explain a small unit's spike.
    threshold
        A fit is accepted only if it removes at least this fraction of the
        template's own energy from the residual, i.e. the correlation must
        exceed (1 + threshold) / 2 of the template energy. Keep it low: in
        a collision the second spike's trough often sits on the first one's
        rebound, which pulls the correlation of both well below their
        template energy, and a strict threshold loses both spikes.
    min_z
        Noise floor. The correlation must also exceed `min_z` times the
        matched-filter noise level, estimated per template as the MAD of
        its correlation with the trace. This is what keeps a low
        `threshold` from fitting noise for small templates at high noise.
    max_rounds
        Each round fits one non-overlapping set of spikes. Isolated spikes
        go in round one, the second spike of a collision in round two.
    """
    if isinstance(templates, dict):
        templates = [templates[k] for k in sorted(templates)]
    if len(templates) == 0:
        raise ValueError("no templates to match")
    temps = np.stack([np.asarray(t, dtype=np.float64) for t in templates])
    k_n, n_t, n_ch = temps.shape
    flat_temps = temps.reshape(k_n, -1).astype(np.float32)
    shifted = ShiftBank(temps)
    residual = np.array(x, dtype=np.float32, copy=True)
    n = residual.shape[0]
    n_valid = n - n_t + 1
    amin, amax = amp_range

    norm2 = (temps ** 2).sum(axis=(1, 2))
    trough = temps.min(axis=2).argmin(axis=1)
    norm2_32 = norm2.astype(np.float32)[:, None]
    floor = np.zeros(k_n)

    # fits: [start, template, amplitude, shift], or None once dropped
    fits: list[list | None] = []
    rounds = 0
    if n_valid > 0:
        corr = correlate_templates(residual, temps).astype(np.float32)
        positions = np.arange(n_valid)
        # Matched-filter noise per template. Spikes are sparse, so the MAD
        # of the correlation over time is the noise, not the signal.
        med = np.median(corr, axis=1, keepdims=True)
        floor = min_z * np.median(np.abs(corr - med), axis=1) / 0.6745
        floor_32 = floor.astype(np.float32)[:, None]

        def update(t: int) -> None:
            _recompute(corr, residual, flat_temps, max(0, t - n_t + 1), min(n_valid, t + n_t))

        def fit_at(k: int, t: int, free_amp: bool) -> tuple[float, float, float]:
            """Sub-sample shift, amplitude and energy gain for T_k near t.

            With `free_amp` False the amplitude is 1 and the gain is
            2 C - |T|^2. With it True the amplitude is C / |T|^2 clipped to
            `amp_range` and the gain is 2 a C - a^2 |T|^2.
            """
            c0 = float(corr[k, t])
            shift, c_peak = 0.0, c0
            if 0 < t < n_valid - 1:
                cm, cp = float(corr[k, t - 1]), float(corr[k, t + 1])
                den = cm - 2.0 * c0 + cp
                if den < 0:
                    shift = float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
                    c_peak = c0 - 0.25 * (cm - cp) * shift
            a = float(np.clip(c_peak / norm2[k], amin, amax)) if free_amp else 1.0
            gain = a * (2.0 * c_peak - a * norm2[k])
            if c_peak < floor[k]:
                gain = -np.inf
            return shift, a, gain

        def refine(indices: list[int], passes: int, free_amp: bool) -> None:
            """Re-fit each listed spike with all the others subtracted.

            A greedy fit made next to an unexplained neighbour carries its
            influence: the position is pulled towards it, with a free
            amplitude the scale absorbs part of it, and sometimes the wrong
            template wins outright. Once the neighbour is fitted too,
            restoring a spike and fitting it again against the cleaner
            residual corrects all three: the re-fit considers every
            template within two samples of the old position and keeps the
            best. A spike whose re-fit no longer clears the threshold is
            dropped: it was a fragment of something else. Two passes let a
            pair of overlapping spikes settle.

            Re-selecting the template matters for a specific failure: a
            large spike explained as two smaller templates stacked on top of
            each other, both pinned at the low end of the amplitude range.
            Restoring either one leaves a residual that the right template
            fits better than the wrong one, and the other then falls under
            threshold and is dropped.
            """
            for _ in range(passes):
                for i in indices:
                    f = fits[i]
                    if f is None:
                        continue
                    t, k, a, sh = f
                    residual[t:t + n_t] += a * shifted(k, sh)
                    update(t)
                    lo, hi = max(0, t - 2), min(n_valid, t + 3)
                    best = None
                    for k2 in range(k_n):
                        t2 = lo + int(np.argmax(corr[k2, lo:hi]))
                        sh2, a2, g = fit_at(k2, t2, free_amp)
                        if g >= threshold * norm2[k2] and (best is None or g > best[0]):
                            best = (g, t2, k2, a2, sh2)
                    if best is None:
                        fits[i] = None
                        continue
                    _, t2, k2, a2, sh2 = best
                    residual[t2:t2 + n_t] -= a2 * shifted(k2, sh2)
                    fits[i] = [t2, k2, a2, sh2]
                    update(t2)
                    if t2 != t:
                        update(t)

        def neighbours(indices: list[int]) -> list[int]:
            """Indices of all live fits within 2 n_t of any of the given fits."""
            live = [j for j, f in enumerate(fits) if f is not None]
            starts = np.array([fits[j][0] for j in live])
            order = np.argsort(starts)
            starts, live_arr = starts[order], np.array(live)[order]
            near: set[int] = set()
            for i in indices:
                t = fits[i][0]
                lo = np.searchsorted(starts, t - 2 * n_t, side="right")
                hi = np.searchsorted(starts, t + 2 * n_t, side="left")
                near.update(live_arr[lo:hi].tolist())
            return sorted(near)

        for _ in range(max_rounds):
            # Detection uses unit amplitude: the energy removed by
            # subtracting T_k at t is 2 C - |T|^2. A free amplitude here
            # lets the first spike of a collision swallow part of the
            # second, which then falls under the threshold and is lost.
            # Amplitudes are estimated afterwards, with neighbours removed.
            # Candidates are ranked by absolute energy removed, not by the
            # fraction of their own template's energy, so a big spike is
            # always fitted before a small one that overlaps it.
            gain = 2.0 * corr - norm2_32
            gain[corr < floor_32] = -np.inf
            best_k = gain.argmax(axis=0)
            best = gain[best_k, positions]
            peaks = best == ndimage.maximum_filter1d(best, 2 * n_t + 1)
            cand = np.flatnonzero(peaks & (best > threshold * norm2_32[best_k, 0]))
            if cand.size == 0:
                break
            cand = _enforce_dead_time(cand, -best.astype(np.float64), n_t)
            new = []
            for t in cand:
                k = int(best_k[t])
                shift, a, _ = fit_at(k, int(t), free_amp=False)
                residual[t:t + n_t] -= a * shifted(k, shift)
                fits.append([int(t), k, a, shift])
                new.append(len(fits) - 1)
            for t in cand:
                update(int(t))
            rounds += 1
            if rounds > 1:
                refine(neighbours(new), passes=2, free_amp=False)

        # Final pass: positions once more, and amplitudes. Isolated spikes
        # settle in one pass; spikes with a neighbour get a second.
        live = [i for i, f in enumerate(fits) if f is not None]
        refine(live, passes=1, free_amp=True)
        live = [i for i in live if fits[i] is not None]
        starts = np.array([fits[i][0] for i in live])
        order = np.argsort(starts)
        gaps_prev = np.diff(starts[order], prepend=-np.inf)
        gaps_next = np.diff(starts[order], append=np.inf)
        crowded = [live[j] for j in order[(gaps_prev < 2 * n_t) | (gaps_next < 2 * n_t)]]
        refine(crowded, passes=1, free_amp=True)

    fits = [f for f in fits if f is not None]
    if not fits:
        empty = np.empty(0, dtype=np.int64)
        return MatchResult(empty, empty.copy(), np.empty(0), empty.copy(), np.empty(0), residual, rounds)

    starts = np.array([f[0] for f in fits], dtype=np.int64)
    labels = np.array([f[1] for f in fits], dtype=np.int64)
    amps = np.array([f[2] for f in fits])
    shifts = np.array([f[3] for f in fits])
    times = np.round(starts + shifts + trough[labels]).astype(np.int64)
    order = np.argsort(times, kind="stable")
    return MatchResult(times[order], labels[order], amps[order], starts[order], shifts[order],
                       residual, rounds)
