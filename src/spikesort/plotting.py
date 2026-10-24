"""Figures for a sorting run."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import Comparison
from .pipeline import Sorting

PALETTE = ["#2a6fdb", "#e0782b", "#2f9e6b", "#c4405a", "#7e57c2",
           "#1a9aa0", "#b08a1e", "#8a5a44", "#5c6b7a", "#d16fb0"]


def _color(k: int) -> str:
    return PALETTE[k % len(PALETTE)]


def _feature_name(i: int, n_pcs: int, per_channel: bool) -> str:
    return f"ch{i // n_pcs} PC{i % n_pcs + 1}" if per_channel else f"PC{i + 1}"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_traces(s: Sorting, start_s: float = 1.0, dur_s: float = 0.25, ax=None):
    """Filtered traces with sorted spikes marked by unit colour."""
    ax = ax or plt.gca()
    a, b = int(start_s * s.fs), int((start_s + dur_s) * s.fs)
    seg = s.filtered[a:b]
    t = np.arange(a, b) / s.fs * 1e3
    offset = 0.55 * float(np.percentile(seg.max(axis=0) - seg.min(axis=0), 75))
    for ch in range(seg.shape[1]):
        ax.plot(t, seg[:, ch] - ch * offset, color="#444", lw=0.5)
    sel = (s.times >= a) & (s.times < b)
    for tt, k in zip(s.times[sel], s.labels[sel], strict=True):
        ax.axvline(tt / s.fs * 1e3, color=_color(k), alpha=0.35, lw=2)
    ax.set_yticks([])
    ax.set_xlabel("time (ms)", fontsize=9)
    ax.set_title("Filtered data, spikes coloured by sorted unit", fontsize=10, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_features(s: Sorting, dims: tuple[int, int] = (0, 1), ax=None, max_points: int = 4000):
    """Scatter of two feature dimensions (default: PC1 of channels 0 and 1)."""
    ax = ax or plt.gca()
    rng = np.random.default_rng(0)
    idx = rng.permutation(s.features.shape[0])[:max_points]
    f, lab = s.features[idx], s.labels[idx]
    for k in np.unique(lab):
        m = lab == k
        ax.scatter(f[m, dims[0]], f[m, dims[1]], s=3, alpha=0.6, color=_color(k), label=f"unit {k}")
    ax.set_xlabel(_feature_name(dims[0], s.n_pcs, s.per_channel_pca), fontsize=9)
    ax.set_ylabel(_feature_name(dims[1], s.n_pcs, s.per_channel_pca), fontsize=9)
    ax.set_title("PCA feature space", fontsize=10, loc="left")
    _style(ax)


def plot_templates(s: Sorting, fig=None, n_show: int = 50):
    """One column per unit, one row per channel: sample waveforms plus median template."""
    units = sorted(s.templates)
    n_ch = s.waveforms.shape[2]
    fig = fig or plt.figure(figsize=(1.6 * len(units), 1.2 * n_ch))
    axes = fig.subplots(n_ch, len(units), sharex=True, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    t = np.arange(s.waveforms.shape[1]) / s.fs * 1e3
    for j, k in enumerate(units):
        idx = np.flatnonzero(s.labels == k)
        idx = rng.permutation(idx)[:n_show]
        for ch in range(n_ch):
            ax = axes[ch, j]
            ax.plot(t, s.waveforms[idx, :, ch].T, color=_color(k), alpha=0.08, lw=0.6)
            ax.plot(t, s.templates[k][:, ch], color="black", lw=1.2)
            ax.axis("off")
        axes[0, j].set_title(f"unit {k}\nn={(s.labels == k).sum()}", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, wspace=0.08, hspace=0.05)
    fig.suptitle("Waveforms by unit (rows are channels, black is the median)", fontsize=10,
                 x=0.02, ha="left")
    return fig


def plot_bic(s: Sorting, ax=None):
    ax = ax or plt.gca()
    ks, bic = s.bic["k_range"], s.bic["bic"]
    ax.plot(ks, bic, "o-", color="#2a6fdb", ms=4)
    ax.axvline(s.bic["k"], color="#c4405a", ls="--", lw=1)
    ax.set_xlabel("number of GMM components", fontsize=9)
    ax.set_ylabel("BIC", fontsize=9)
    ax.set_title(f"Model selection (BIC picks k={s.bic['k']})", fontsize=10, loc="left")
    _style(ax)


def plot_comparison(c: Comparison, ax=None):
    """Precision, recall and accuracy per ground-truth unit."""
    ax = ax or plt.gca()
    units = np.arange(c.accuracy.size)
    w = 0.27
    ax.bar(units - w, c.precision, w, label="precision", color="#2a6fdb")
    ax.bar(units, c.recall, w, label="recall", color="#e0782b")
    ax.bar(units + w, c.accuracy, w, label="accuracy", color="#2f9e6b")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(units)
    ax.set_xlabel("ground-truth unit", fontsize=9)
    ax.set_title(f"Against ground truth (mean accuracy {c.accuracy.mean():.2f})",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    _style(ax)


def summary_figure(s: Sorting, c: Comparison | None = None):
    fig = plt.figure(figsize=(11, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    plot_traces(s, ax=fig.add_subplot(gs[0, :2]))
    plot_bic(s, ax=fig.add_subplot(gs[0, 2]))
    plot_features(s, ax=fig.add_subplot(gs[1, 0]))
    plot_features(s, dims=(0, 2), ax=fig.add_subplot(gs[1, 1]))
    if c is not None:
        plot_comparison(c, ax=fig.add_subplot(gs[1, 2]))
    return fig


def find_collision(s: Sorting, max_gap_ms: float = 1.0, after_s: float = 0.5) -> int | None:
    """Sample index of the first pair of different-unit spikes closer than `max_gap_ms`."""
    gap = int(round(max_gap_ms * 1e-3 * s.fs))
    t, lab = s.times, s.labels
    close = (np.diff(t) < gap) & (lab[1:] != lab[:-1]) & (t[:-1] > after_s * s.fs)
    hits = np.flatnonzero(close)
    return int(t[hits[0]]) if hits.size else None


def plot_deconvolution(s: Sorting, centre: int | None = None, window_ms: float = 12.0, fig=None):
    """Filtered trace, the fitted templates, and what is left over.

    Centred on a resolved collision by default. Each row is one channel.
    """
    if s.residual is None:
        raise ValueError("this sorting has no residual; template matching was off")
    if centre is None:
        centre = find_collision(s)
    if centre is None:
        centre = int(s.times[len(s.times) // 2]) if s.times.size else s.filtered.shape[0] // 2
    half = int(round(window_ms * 1e-3 * s.fs / 2))
    a, b = max(0, centre - half), min(s.filtered.shape[0], centre + half)
    t = (np.arange(a, b) - centre) / s.fs * 1e3
    filt = s.filtered[a:b]
    resid = s.residual[a:b]
    fit = filt - resid
    n_ch = filt.shape[1]

    fig = fig or plt.figure(figsize=(10, 1.1 * n_ch + 1.6), constrained_layout=True)
    axes = fig.subplots(1, 3, sharey=True)
    offset = 1.1 * float(np.abs(filt).max())
    for ax, seg, title in [(axes[0], filt, "Filtered trace"),
                           (axes[1], fit, "Fitted templates"),
                           (axes[2], resid, "Residual")]:
        for ch in range(n_ch):
            ax.plot(t, seg[:, ch] - ch * offset, color="#444", lw=0.7)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("time from collision (ms)", fontsize=9)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=8)
    sel = (s.times >= a) & (s.times < b)
    for tt, k in zip(s.times[sel], s.labels[sel], strict=True):
        for ax in axes[:2]:
            ax.axvline((tt - centre) / s.fs * 1e3, color=_color(k), alpha=0.5, lw=2, label=f"unit {k}")
    handles, labels = axes[1].get_legend_handles_labels()
    seen = dict(zip(labels, handles, strict=True))
    axes[1].legend(seen.values(), seen.keys(), fontsize=8, frameon=False, loc="upper right")
    return fig


def plot_drift(s: Sorting, true_displacement=None, fig=None, max_points: int = 20000):
    """Spike depth against time, before and after drift correction.

    Left: every detected spike at its estimated depth, with the estimated
    displacement drawn through the middle of the array (and the true one,
    if given). Right: the same spikes with the estimate subtracted, which
    is what the sorter sees after correction.

    `true_displacement` is a callable from times in seconds to microns, in
    the same convention as the estimate (the apparent shift of spikes).
    """
    if s.drift is None:
        raise ValueError("this sorting has no drift estimate")
    d = s.drift
    rng = np.random.default_rng(0)
    idx = rng.permutation(d.times_s.size)[:max_points]
    t, depth, amp = d.times_s[idx], d.depths[idx], d.amplitudes[idx]
    shade = np.clip(amp / np.percentile(amp, 95), 0.15, 1.0)
    grey = 0.75 * (1 - shade)
    colors = np.column_stack([grey, grey, grey, np.full_like(shade, 0.7)])
    mid = 0.5 * (d.depths.min() + d.depths.max())

    fig = fig or plt.figure(figsize=(11, 3.8), constrained_layout=True)
    axes = fig.subplots(1, 2, sharey=True)
    axes[0].scatter(t, depth, s=2, c=colors, lw=0)
    axes[0].plot(d.bin_centres_s, mid + d.displacement_um, color="#2a6fdb", lw=1.6, label="estimated")
    if true_displacement is not None:
        tt = np.linspace(0, d.bin_centres_s[-1] + d.bin_centres_s[0], 400)
        true = np.asarray(true_displacement(tt), dtype=float)
        true = true - true.mean()
        axes[0].plot(tt, mid + true, color="#e0782b", lw=1.2, ls="--", label="true")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_title("Spike depths, raw (darker = larger spike)", fontsize=10, loc="left")
    axes[1].scatter(t, depth - d.at(t), s=2, c=colors, lw=0)
    axes[1].set_title("After subtracting the estimated drift", fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("time (s)", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("depth along probe (um)", fontsize=9)
    return fig
