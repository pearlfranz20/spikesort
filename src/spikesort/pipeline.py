"""End-to-end sorting.

Stage one: filter, detect, align, featurise, cluster, split, merge.
Stage two: template matching on the filtered trace with the stage-one
templates, which resolves overlapping spikes (see `matching.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cluster import gmm_cluster, merge_similar, split_bimodal, templates_from_labels
from .detect import detect_spikes, detect_spikes_combined, extract_waveforms
from .drift import DriftEstimate, estimate_drift, shift_channels, spike_depths
from .evaluate import isi_violation_rate, isolation_distance, snr
from .features import apply_features, waveform_features
from .matching import match_templates, select_templates
from .preprocess import bandpass, common_median_reference, mad_noise


@dataclass
class SortConfig:
    low_hz: float = 300.0
    high_hz: float = 6000.0
    use_cmr: bool = False
    detect_method: str = "combined"  # "combined", "energy" or "threshold"
    threshold: float = 5.0
    dead_time_ms: float = 1.0  # for the single detectors; "combined" uses 1 ms and 2 ms
    pre_ms: float = 0.6
    post_ms: float = 1.2
    align: str = "com"  # "com", "parabola" or "none"
    n_pcs: int = 6
    per_channel_pca: bool = False
    k_max: int = 20
    merge_corr: float = 0.97
    merge_amp_ratio: float = 0.85
    merge_dprime: float = 2.5
    split_dprime: float | None = 4.0  # held-out d' to split a cluster hiding two units; None disables
    template_matching: bool = True
    match_threshold: float = 0.1
    match_min_z: float = 5.0
    match_amp_range: tuple[float, float] = (0.8, 1.25)
    match_max_rounds: int = 10
    template_post_ms: float = 2.0  # stage-two templates run longer, to cover the rebound
    min_template_spikes: int = 10
    max_outlier_frac: float = 0.5
    drift_correction: bool = False  # needs channel_positions, see drift.py
    drift_axis: tuple[float, float] = (0.0, 1.0)
    drift_bin_s: float = 2.0
    drift_max_um: float = 60.0
    drift_kernel_um: float | None = None  # default: nearest-channel spacing
    drift_passes: int = 2  # re-estimate on corrected data; the centre of mass under-reads motion
    seed: int = 0


@dataclass
class Sorting:
    """Output of the sorter.

    `times`, `labels`, `waveforms` and `features` describe the final
    spikes. With template matching on, the waveforms are cut from the
    filtered trace at the matched times, so a resolved collision shows up
    as a waveform with another unit's spike riding on it, and `features`
    are those waveforms projected onto the stage-one PCA axes.
    """

    times: np.ndarray
    labels: np.ndarray
    waveforms: np.ndarray
    features: np.ndarray
    templates: dict[int, np.ndarray]
    noise: np.ndarray
    filtered: np.ndarray
    bic: dict = field(default_factory=dict)
    fs: float = 30_000.0
    n_pcs: int = 6
    per_channel_pca: bool = False
    amplitudes: np.ndarray | None = None  # template scale per spike (matching only)
    residual: np.ndarray | None = None  # filtered minus every fitted template
    stage1: dict = field(default_factory=dict)  # detection and clustering before matching
    matching: dict = field(default_factory=dict)
    drift: DriftEstimate | None = None  # set when drift correction ran; `filtered` is then corrected

    @property
    def n_units(self) -> int:
        return len(self.templates)

    def quality_table(self) -> list[dict]:
        """GT-free quality metrics for each sorted unit."""
        rows = []
        for k, tmpl in self.templates.items():
            t = self.times[self.labels == k]
            rows.append({
                "unit": k,
                "n_spikes": int(t.size),
                "rate_hz": t.size / (self.filtered.shape[0] / self.fs),
                "snr": snr(tmpl, self.noise),
                "isi_violations": isi_violation_rate(t, self.fs),
                "isolation_distance": isolation_distance(self.features, self.labels, k),
            })
        return rows


def sort(data: np.ndarray, fs: float, config: SortConfig | None = None,
         channel_positions: np.ndarray | None = None) -> Sorting:
    """Sort a (n_samples, n_channels) recording.

    `channel_positions` (n_channels, 2) in microns is only needed for
    drift correction.
    """
    cfg = config or SortConfig()

    x = bandpass(data, fs, cfg.low_hz, cfg.high_hz)
    if cfg.use_cmr:
        x = common_median_reference(x)

    drift = None
    if cfg.drift_correction:
        if channel_positions is None:
            raise ValueError("drift correction needs channel_positions")
        x, drift = correct_drift(x, fs, np.asarray(channel_positions, dtype=float), cfg)
    noise = mad_noise(x)

    times = _detect(x, fs, noise, cfg)
    wf, times = extract_waveforms(x, times, fs, cfg.pre_ms, cfg.post_ms, align=cfg.align)
    feats, models = waveform_features(wf, cfg.n_pcs, per_channel=cfg.per_channel_pca)

    labels, bic = gmm_cluster(feats, range(1, cfg.k_max + 1), seed=cfg.seed)
    if cfg.split_dprime is not None:
        labels = split_bimodal(wf, labels, feats, cfg.split_dprime, seed=cfg.seed)
    labels = merge_similar(wf, labels, cfg.merge_corr, cfg.merge_amp_ratio, cfg.merge_dprime)
    cluster_templates = templates_from_labels(wf, labels)
    stage1 = {"times": times, "labels": labels, "n_clusters": len(cluster_templates)}

    common = dict(noise=noise, filtered=x, bic=bic, fs=fs, n_pcs=cfg.n_pcs,
                  per_channel_pca=cfg.per_channel_pca, stage1=stage1, drift=drift)
    if not cfg.template_matching or times.size == 0:
        return Sorting(times, labels, wf, feats, cluster_templates, **common)

    # Longer waveforms for the templates, so the whole spike is subtracted.
    wf_long, _, ok = extract_waveforms(x, times, fs, cfg.pre_ms, cfg.template_post_ms,
                                       align=cfg.align, return_mask=True)
    kept, dropped = select_templates(wf_long, labels[ok], noise, cfg.min_template_spikes,
                                     max_outlier_frac=cfg.max_outlier_frac)
    stage1["dropped"] = dropped
    by_size = sorted(kept, key=lambda k: -int((labels == k).sum()))
    dictionary = [kept[k] for k in by_size]
    m = match_templates(x, dictionary, cfg.match_amp_range, cfg.match_threshold,
                        cfg.match_min_z, cfg.match_max_rounds)

    wf2, times2, ok = extract_waveforms(x, m.times, fs, cfg.pre_ms, cfg.post_ms,
                                        align="none", return_mask=True)
    labels2, amps = m.labels[ok], m.amplitudes[ok]
    # Renumber the surviving units 0..K-1 by spike count.
    ks, counts = np.unique(labels2, return_counts=True)
    order = ks[np.argsort(-counts, kind="stable")]
    remap = {int(k): i for i, k in enumerate(order)}
    labels2 = np.array([remap[int(k)] for k in labels2], dtype=np.int64)
    n_short = wf2.shape[1] if wf2.size else wf.shape[1]
    templates2 = {remap[int(k)]: dictionary[int(k)][:n_short] for k in order}
    feats2 = apply_features(wf2, models, cfg.per_channel_pca) if wf2.size else np.empty((0, feats.shape[1]))

    info = {"rounds": m.rounds, "n_templates": len(dictionary), "n_stage1": int(times.size),
            "n_matched": int(times2.size)}
    return Sorting(times2, labels2, wf2, feats2, templates2, amplitudes=amps,
                   residual=m.residual.astype(np.float32), matching=info, **common)


def _detect(x: np.ndarray, fs: float, noise: np.ndarray | None, cfg: SortConfig) -> np.ndarray:
    if cfg.detect_method == "combined":
        return detect_spikes_combined(x, fs, cfg.threshold, noise)
    return detect_spikes(x, fs, cfg.threshold, cfg.dead_time_ms, noise=noise, method=cfg.detect_method)


def correct_drift(x: np.ndarray, fs: float, channel_positions: np.ndarray,
                  cfg: SortConfig) -> tuple[np.ndarray, DriftEstimate]:
    """Estimate probe drift from a detection pass and undo it in `x`.

    The depth estimator under-reads motion by about 10%, so a second pass
    on the corrected data picks up most of the remainder. The passes'
    estimates are summed and the original data shifted once by the total.
    """
    axis = np.asarray(cfg.drift_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    n = x.shape[0]
    t_all = np.arange(n) / fs
    total = np.zeros(n)
    corrected = x
    est = None
    for _ in range(max(1, cfg.drift_passes)):
        times = _detect(corrected, fs, None, cfg)
        wf, times = extract_waveforms(corrected, times, fs, cfg.pre_ms, cfg.post_ms, align="none")
        depths, amps = spike_depths(wf, channel_positions, axis)
        this = estimate_drift(times / fs, depths, amps, n / fs, bin_s=cfg.drift_bin_s,
                              max_shift_um=cfg.drift_max_um)
        total += this.at(t_all)
        if est is None:
            est = this
        else:
            est.displacement_um = est.displacement_um + this.displacement_um
        corrected = shift_channels(x, channel_positions, total, axis, cfg.drift_kernel_um)
    return corrected, est
