"""End-to-end sorting: filter, detect, extract, featurise, cluster, split, merge."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cluster import gmm_cluster, merge_similar, split_bimodal, templates_from_labels
from .detect import detect_spikes, extract_waveforms
from .evaluate import isi_violation_rate, isolation_distance, snr
from .features import waveform_features
from .preprocess import bandpass, common_median_reference, mad_noise


@dataclass
class SortConfig:
    low_hz: float = 300.0
    high_hz: float = 6000.0
    use_cmr: bool = False
    threshold: float = 5.0
    dead_time_ms: float = 1.0
    pre_ms: float = 0.6
    post_ms: float = 1.2
    align: str = "com"  # "com", "parabola" or "none"
    n_pcs: int = 6
    per_channel_pca: bool = False
    k_max: int = 20
    merge_corr: float = 0.97
    merge_amp_ratio: float = 0.85
    merge_dprime: float = 2.5
    split_dprime: float | None = None  # opt-in, see README benchmarks
    seed: int = 0


@dataclass
class Sorting:
    """Output of the sorter."""

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


def sort(data: np.ndarray, fs: float, config: SortConfig | None = None) -> Sorting:
    """Sort a (n_samples, n_channels) recording."""
    cfg = config or SortConfig()

    x = bandpass(data, fs, cfg.low_hz, cfg.high_hz)
    if cfg.use_cmr:
        x = common_median_reference(x)
    noise = mad_noise(x)

    times = detect_spikes(x, fs, threshold=cfg.threshold,
                          dead_time_ms=cfg.dead_time_ms, noise=noise)
    wf, times = extract_waveforms(x, times, fs, cfg.pre_ms, cfg.post_ms, align=cfg.align)
    feats, _ = waveform_features(wf, cfg.n_pcs, per_channel=cfg.per_channel_pca)

    labels, bic = gmm_cluster(feats, range(1, cfg.k_max + 1), seed=cfg.seed)
    if cfg.split_dprime is not None:
        labels = split_bimodal(wf, labels, feats, cfg.split_dprime, seed=cfg.seed)
    labels = merge_similar(wf, labels, cfg.merge_corr, cfg.merge_amp_ratio, cfg.merge_dprime)

    return Sorting(
        times=times,
        labels=labels,
        waveforms=wf,
        features=feats,
        templates=templates_from_labels(wf, labels),
        noise=noise,
        filtered=x,
        bic=bic,
        fs=fs,
        n_pcs=cfg.n_pcs,
        per_channel_pca=cfg.per_channel_pca,
    )
