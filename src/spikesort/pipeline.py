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
from .localize import (
    Geometry,
    build_unit,
    cluster_trajectories,
    drift_shape,
    link_trajectories,
    render_template,
    rigid_registration,
    trough_profiles,
)
from .matching import match_in_chunks, match_templates, select_templates
from .preprocess import bandpass, common_median_reference, mad_noise
from .tracking import track_templates


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
    drift_correction: bool = False  # needs channel_positions, see drift.py and localize.py
    drift_method: str = "interpolate"  # "interpolate" (re-sample channels) or "positions" (localize.py)
    drift_axis: tuple[float, float] = (0.0, 1.0)
    drift_bin_s: float = 2.0
    drift_max_um: float = 60.0
    drift_kernel_um: float | None = None  # default: nearest-channel spacing
    drift_passes: int = 3  # re-estimate on corrected data; the centre of mass under-reads motion
    drift_falloff: float = 1.5  # point-source amplitude falls off as distance^-falloff ("positions")
    drift_fixed_z_um: float = 15.0  # source height assumed on a degenerate geometry such as a tetrode
    link_max_profile_dist: float = 0.4  # relative profile mismatch where two trajectories meet
    link_min_shape_corr: float = 0.97  # waveform time course agreement on the loudest channel
    link_max_gap_bins: int = 2  # trajectories further apart in time than this never meet
    link_max_coactivity: float = 0.45  # bands of one neuron rarely fire in the same bins
    traj_pool_bins: int = 3  # a sparse cluster pools up to this many bins per trajectory point
    tracking: bool = False  # chunked stage two with templates that follow the units, see tracking.py
    track_chunk_s: float = 5.0
    track_alpha: float = 0.5  # how far a template moves towards each chunk's median
    track_min_spikes: int = 10  # spikes in a chunk needed to update a template
    track_discovery: bool = True
    track_discovery_min: int = 100  # unexplained events to pile up before clustering them
    track_merge_amp_ratio: float = 0.4  # drift can halve a unit's amplitude between its bands
    track_merge_corr: float = 0.6  # loose shape gate for the merge test; the timing test decides
    track_ccg_min_expected: float = 10.0  # chance-level close pairs needed to trust the timing test
    track_min_unit_spikes: int = 20  # units with fewer spikes at the end are dropped
    track_init_s: float | None = None  # build the initial templates from this much of the start only
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
    drift: DriftEstimate | None = None  # set when drift correction ran ("interpolate" also corrects filtered)
    tracking: dict = field(default_factory=dict)  # chunk edges and template history when tracking ran
    localization: dict = field(default_factory=dict)  # units, trajectories and rigid fit ("positions" method)

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
        if cfg.drift_method == "interpolate":
            x, drift = correct_drift(x, fs, np.asarray(channel_positions, dtype=float), cfg)
        elif cfg.drift_method != "positions":
            raise ValueError(f"unknown drift_method {cfg.drift_method!r}")
    noise = mad_noise(x)

    times = _detect(x, fs, noise, cfg)
    wf, times = extract_waveforms(x, times, fs, cfg.pre_ms, cfg.post_ms, align=cfg.align)
    feats, models = waveform_features(wf, cfg.n_pcs, per_channel=cfg.per_channel_pca)

    labels, bic = cluster_stage_one(wf, feats, cfg)
    cluster_templates = templates_from_labels(wf, labels)
    stage1 = {"times": times, "labels": labels, "n_clusters": len(cluster_templates)}

    common = dict(noise=noise, filtered=x, bic=bic, fs=fs, n_pcs=cfg.n_pcs,
                  per_channel_pca=cfg.per_channel_pca, stage1=stage1, drift=drift)
    if not cfg.template_matching or times.size == 0:
        return Sorting(times, labels, wf, feats, cluster_templates, **common)

    # Longer waveforms for the templates, so the whole spike is subtracted.
    wf_long, _, ok = extract_waveforms(x, times, fs, cfg.pre_ms, cfg.template_post_ms,
                                       align=cfg.align, return_mask=True)
    positions_method = cfg.drift_correction and cfg.drift_method == "positions"
    # With drift, judge each spike against its neighbours in time, so a
    # drifting unit is not mistaken for a collision cluster (select_templates).
    kept, dropped = select_templates(wf_long, labels[ok], noise, cfg.min_template_spikes,
                                     max_outlier_frac=cfg.max_outlier_frac,
                                     times=times[ok] if positions_method else None)
    stage1["dropped"] = dropped
    stage1["kept_templates"] = kept
    by_size = sorted(kept, key=lambda k: -int((labels == k).sum()))
    dictionary = [kept[k] for k in by_size]
    tracking_info: dict = {}
    localization_info: dict = {}
    if positions_method:
        m_times, m_labels, m_amps, residual, unit_templates, drift, localization_info = positions_stage_two(
            x, fs, times[ok], labels[ok], wf_long, kept, np.asarray(channel_positions, dtype=float), cfg)
        rounds = int(localization_info["n_bins"])
    elif cfg.tracking:
        seeds = [wf_long[labels[ok] == k] for k in by_size]
        if cfg.track_init_s is not None:
            # Kilosort 1 style: templates from the start of the recording,
            # before drift has had time to split anything. Later units are
            # left to discovery.
            n_init = int(round(cfg.track_init_s * fs))
            head = times < n_init
            t0 = times[head]
            wf0, _, ok0 = extract_waveforms(x, t0, fs, cfg.pre_ms, cfg.template_post_ms,
                                            align=cfg.align, return_mask=True)
            f0, _ = waveform_features(wf0[:, :wf.shape[1]], cfg.n_pcs, per_channel=cfg.per_channel_pca)
            l0, _ = gmm_cluster(f0, range(1, cfg.k_max + 1), seed=cfg.seed)
            if cfg.split_dprime is not None:
                l0 = split_bimodal(wf0, l0, f0, cfg.split_dprime, seed=cfg.seed)
            l0 = merge_similar(wf0, l0, cfg.merge_corr, cfg.merge_amp_ratio, cfg.merge_dprime)
            kept0, _ = select_templates(wf0, l0, noise, cfg.min_template_spikes,
                                        max_outlier_frac=cfg.max_outlier_frac)
            order0 = sorted(kept0, key=lambda k: -int((l0 == k).sum()))
            dictionary = [kept0[k] for k in order0]
            seeds = [wf0[l0 == k] for k in order0]
        tr = track_templates(x, fs, dictionary, cfg, seed_waveforms=seeds)
        m_times, m_labels, m_amps, residual = tr.times, tr.labels, tr.amplitudes, tr.residual
        unit_templates = tr.templates
        tracking_info = {"chunk_edges": tr.chunk_edges, "history": tr.history,
                         "discovered": tr.n_discovered, "merged": tr.n_merged, "log": tr.log}
        rounds = int(tr.chunk_edges.size - 1)
    else:
        m = match_templates(x, dictionary, cfg.match_amp_range, cfg.match_threshold,
                            cfg.match_min_z, cfg.match_max_rounds)
        m_times, m_labels, m_amps, residual = m.times, m.labels, m.amplitudes, m.residual
        unit_templates = dict(enumerate(dictionary))
        rounds = m.rounds

    wf2, times2, ok = extract_waveforms(x, m_times, fs, cfg.pre_ms, cfg.post_ms,
                                        align="none", return_mask=True)
    labels2, amps = m_labels[ok], m_amps[ok]
    # Renumber the surviving units 0..K-1 by spike count.
    ks, counts = np.unique(labels2, return_counts=True)
    order = ks[np.argsort(-counts, kind="stable")]
    remap = {int(k): i for i, k in enumerate(order)}
    labels2 = np.array([remap[int(k)] for k in labels2], dtype=np.int64)
    n_short = wf2.shape[1] if wf2.size else wf.shape[1]
    templates2 = {remap[int(k)]: unit_templates[int(k)][:n_short] for k in order}
    feats2 = apply_features(wf2, models, cfg.per_channel_pca) if wf2.size else np.empty((0, feats.shape[1]))
    if tracking_info:
        tracking_info["history"] = {remap[int(k)]: h for k, h in tracking_info["history"].items()
                                    if int(k) in remap}

    info = {"rounds": rounds, "n_templates": len(dictionary), "n_stage1": int(times.size),
            "n_matched": int(times2.size), "mode": "tracking" if cfg.tracking else "matching"}
    if localization_info:
        info["mode"] = "positions"
        common["drift"] = drift
        # Final unit id of every linked unit (units with no spikes left have none).
        localization_info["final_ids"] = {i: remap.get(i) for i in range(len(localization_info["units"]))}
    return Sorting(times2, labels2, wf2, feats2, templates2, amplitudes=amps,
                   residual=residual.astype(np.float32), matching=info, tracking=tracking_info,
                   localization=localization_info, **common)


def cluster_stage_one(wf: np.ndarray, feats: np.ndarray, cfg: SortConfig) -> tuple[np.ndarray, dict]:
    """Gaussian mixture with BIC, then the split and merge passes."""
    labels, bic = gmm_cluster(feats, range(1, cfg.k_max + 1), seed=cfg.seed)
    if cfg.split_dprime is not None:
        labels = split_bimodal(wf, labels, feats, cfg.split_dprime, seed=cfg.seed)
    labels = merge_similar(wf, labels, cfg.merge_corr, cfg.merge_amp_ratio, cfg.merge_dprime)
    return labels, bic


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


def positions_stage_two(x, fs, times, labels, wf_long, kept, channel_positions, cfg):
    """Stage two for the "positions" drift method. See localize.py.

    Fits every kept cluster's trajectory, links clusters whose trajectories
    meet into units, renders each unit's template for every bin along its
    own trajectory, and matches bin by bin.
    """
    geom = Geometry.from_positions(channel_positions)
    n = x.shape[0]
    chunk = max(1, int(round(cfg.drift_bin_s * fs)))
    edges = np.append(np.arange(0, n, chunk), n)
    n_bins = edges.size - 1
    centres = 0.5 * (edges[:-1] + edges[1:]) / fs

    mask = np.isin(labels, list(kept))
    profiles = trough_profiles(wf_long[mask])
    traj = cluster_trajectories(times[mask], labels[mask], profiles, edges, geom, cfg.drift_falloff,
                                cfg.drift_fixed_z_um, max_pool_bins=cfg.traj_pool_bins)
    rigid = rigid_registration(traj, n_bins, geom.dim)
    frame = None if geom.degenerate else rigid.displacement
    groups = link_trajectories(traj, kept, geom, n_bins, cfg.link_max_profile_dist, cfg.link_min_shape_corr,
                               cfg.link_max_gap_bins, cfg.link_max_coactivity, cfg.drift_falloff, frame)
    units = [build_unit(g, traj, kept, n_bins, geom, None if geom.degenerate else rigid, cfg.drift_falloff)
             for g in groups]
    templates_per_bin = [[render_template(u.reference, u.reference_position, u.positions[b], u.z, geom,
                                          cfg.drift_falloff, u.reference_amplitude, u.amplitudes[b])
                          for u in units] for b in range(n_bins)]
    ids = [list(range(len(units)))] * n_bins
    m_times, m_units, m_amps, residual = match_in_chunks(x, edges, templates_per_bin, ids, cfg)

    axis = np.asarray(cfg.drift_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    wf_short = wf_long[mask][:, :int(round((cfg.pre_ms + cfg.post_ms) * 1e-3 * fs))]
    depths, amps = spike_depths(wf_short, channel_positions, axis)
    which = np.clip(np.searchsorted(edges, times[mask], "right") - 1, 0, n_bins - 1)
    shape = drift_shape(units, geom, axis)
    drift = None
    if not geom.degenerate:
        # The rigid fit is a real displacement in microns only where the
        # fitted positions share a frame (module docstring of localize.py).
        along = rigid.displacement @ (geom.basis.T @ axis)
        drift = DriftEstimate(centres, along, np.bincount(which, minlength=n_bins),
                              depths=depths, amplitudes=amps, times_s=times[mask] / fs)
    unit_templates = {i: u.reference for i, u in enumerate(units)}
    info = {"n_bins": n_bins, "groups": groups, "trajectories": traj, "rigid_residual_um": rigid.residual_um,
            "degenerate": geom.degenerate, "units": units, "edges": edges, "bin_centres_s": centres,
            "drift_shape": shape, "templates_per_bin": templates_per_bin}
    return m_times, m_units, m_amps, residual, unit_templates, drift, info
