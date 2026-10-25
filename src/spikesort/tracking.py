"""Stage two with templates that follow the neurons over time.

Probe drift changes every unit's waveform slowly. Plain matching fits one
fixed template per unit to the whole recording, which is wrong whenever
the recording is long enough to drift. This stage processes the recording
in chunks and lets each template follow its unit:

    pass one, chunk by chunk
        match the chunk with the current templates
        for each unit with enough spikes in the chunk, move its template
            towards the median of those spikes, taken with every other
            unit's spikes already subtracted
        merge templates that have converged onto the same neuron
        collect what the templates could not explain; when enough has
            piled up, cluster it and add any new unit as a template
    pass two
        re-match every chunk with the templates recorded for that chunk,
        so a unit discovered late is also fitted early, and a unit merged
        late is one unit throughout

Nothing here interpolates across channels, so it works for any geometry,
including a tetrode. It is the idea behind Kilosort 1's batch-wise
template updates, with a discovery step added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cluster import gmm_cluster, merge_similar, separation
from .detect import detect_spikes, extract_waveforms
from .features import waveform_features
from .matching import match_chunk, select_templates
from .preprocess import mad_noise


@dataclass
class TrackResult:
    times: np.ndarray  # trough sample of each fitted spike, sorted
    labels: np.ndarray  # final unit id per spike, 0..K-1
    amplitudes: np.ndarray
    residual: np.ndarray
    templates: dict[int, np.ndarray]  # each unit's template at the end
    chunk_edges: np.ndarray  # sample index of every chunk boundary
    history: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)  # unit -> chunk -> template
    n_discovered: int = 0
    n_merged: int = 0
    log: list = field(default_factory=list)  # (chunk, event, details) for debugging


def _cleaned(residual, starts, labels, amps, shifts, bank, n_t):
    """Each fitted spike's waveform with every other fit subtracted."""
    out = np.empty((starts.size, n_t, residual.shape[1]), dtype=np.float64)
    for i, (s, k, a, sh) in enumerate(zip(starts, labels, amps, shifts, strict=True)):
        out[i] = residual[s:s + n_t] + a * bank(int(k), float(sh))
    return out


def track_templates(x: np.ndarray, fs: float, templates: list[np.ndarray], cfg,
                    seed_waveforms: list[np.ndarray] | None = None) -> TrackResult:
    """Chunked matching with template updates, merges and discovery. See the module docstring.

    `cfg` is a `SortConfig`; the fields it reads are the matching ones
    plus `track_chunk_s`, `track_alpha`, `track_min_spikes`,
    `track_discovery`, `track_discovery_min`. `seed_waveforms` are the
    stage-one waveforms behind each template, used to seed the rolling
    buffers that the merge test works on.
    """
    n, n_ch = x.shape
    n_t = templates[0].shape[0]
    chunk = max(n_t * 4, int(round(cfg.track_chunk_s * fs)))
    edges = np.arange(0, n, chunk)
    edges = np.append(edges, n)
    n_chunks = edges.size - 1
    margin = 2 * n_t
    buffer_size = 200
    rng = np.random.default_rng(cfg.seed)

    # Unit bookkeeping. Ids never change; merged units point at a survivor.
    temps: dict[int, np.ndarray] = {i: np.asarray(t, dtype=np.float64).copy()
                                    for i, t in enumerate(templates)}
    alias: dict[int, int] = {}
    merged_at: dict[int, int] = {}
    counts: dict[int, int] = {i: 0 for i in temps}
    history: dict[int, dict[int, np.ndarray]] = {i: {} for i in temps}
    # Recent cleaned waveforms per unit. A unit that has stopped winning
    # spikes (because a tracked twin now fits better) keeps its last ones,
    # so the pair can still be tested for being the same neuron.
    recent: dict[int, np.ndarray] = {}
    for i in temps:
        if seed_waveforms is not None and seed_waveforms[i].shape[0]:
            w = seed_waveforms[i]
            recent[i] = w[rng.permutation(w.shape[0])[:buffer_size]].astype(np.float64)
        else:
            recent[i] = np.empty((0, n_t, n_ch))
    spikes: dict[int, list[int]] = {i: [] for i in temps}
    discovered = merged = 0

    def resolve(k: int) -> int:
        while k in alias:
            k = alias[k]
        return k

    log: list = []

    def same_neuron(a: int, b: int, cleaned: np.ndarray, unit: np.ndarray) -> bool:
        """Is this pair one neuron? Three sources of evidence, in order.

        Timing, when there is enough of it: a neuron cannot fire twice
        within its refractory period. Close pairs across the two trains at
        the chance rate means two neurons, whatever the waveforms say, and
        outranks everything below. Close pairs at a tenth of chance or
        less means one neuron. This decides the case of a neuron split
        into interleaved bands.

        Same-chunk overlap: if both fired in this chunk, cross-fitted d'
        between their spikes here. A template that has tracked its neuron
        onto a stale twin shares the twin's spikes for a chunk or two,
        and that is when the two look alike.

        Handover: if only one of them fired here, the other has stopped
        winning spikes because this one fits them better. If their
        templates are now the same waveform (the stage-one merge
        criteria on shape and amplitude), that is what a tracked template
        catching up with a band looks like, and they are merged.
        """
        close, expected = refractory_overlap(np.array(spikes[a]), np.array(spikes[b]), fs)
        if expected >= 2.0 and close >= 0.5 * expected:
            log.append(("veto", a, b, close, round(expected, 1)))
            return False
        if expected >= cfg.track_ccg_min_expected and close <= 0.1 * expected:
            log.append(("merge-timing", a, b, close, round(expected, 1)))
            return True
        ta, tb = temps[a], temps[b]
        r = np.corrcoef(ta.ravel(), tb.ravel())[0, 1]
        pa, pb = -ta.min(), -tb.min()
        ratio = min(pa, pb) / max(pa, pb)
        if r <= cfg.merge_corr or ratio <= cfg.merge_amp_ratio:
            return False
        na, nb = int((unit == a).sum()), int((unit == b).sum())
        if na >= 10 and nb >= 10:
            both = np.concatenate([cleaned[unit == a], cleaned[unit == b]])
            lab = np.repeat([0, 1], [na, nb])
            dp = separation(both, lab, 0, 1)
            log.append(("overlap-test", a, b, round(float(dp), 2), bool(dp < cfg.merge_dprime)))
            return bool(dp < cfg.merge_dprime)
        verdict = bool(r > cfg.merge_corr and ratio > cfg.merge_amp_ratio)
        log.append(("handover-test", a, b, round(float(r), 3), round(float(ratio), 2), verdict))
        return verdict

    def try_merges(cleaned: np.ndarray, unit: np.ndarray) -> int:
        """Merge pairs that look like one neuron."""
        done = 0
        ids = sorted(temps)
        for ai, a in enumerate(ids):
            for b in ids[ai + 1:]:
                if a not in temps or b not in temps:
                    continue
                ta, tb = temps[a], temps[b]
                r = np.corrcoef(ta.ravel(), tb.ravel())[0, 1]
                pa, pb = -ta.min(), -tb.min()
                if r <= cfg.track_merge_corr or min(pa, pb) / max(pa, pb) <= cfg.track_merge_amp_ratio:
                    continue
                if not same_neuron(a, b, cleaned, unit):
                    continue
                keep, drop = (a, b) if counts[a] >= counts[b] else (b, a)
                alias[drop] = keep
                counts[keep] += counts[drop]
                recent[keep] = np.concatenate([recent[keep], recent[drop]])[-buffer_size:]
                spikes[keep] = sorted(spikes[keep] + spikes[drop])
                unit[unit == drop] = keep
                del temps[drop]
                del recent[drop]
                del spikes[drop]
                done += 1
        return done

    # Pass one.
    residual = np.array(x, dtype=np.float32, copy=True)
    buffer_wf: list[np.ndarray] = []
    for c in range(n_chunks):
        c0, c1 = int(edges[c]), int(edges[c + 1])
        ids = sorted(temps)
        starts, lab, amps, shifts, t_fit, bank = match_chunk(
            residual, c0, c1, margin, [temps[i] for i in ids], cfg)
        unit = np.array([ids[int(j)] for j in lab], dtype=np.int64)
        cleaned = _cleaned(residual, starts, lab, amps, shifts, bank, n_t)

        # Update every unit that fired enough, and remember its spikes.
        for i in ids:
            m = unit == i
            counts[i] += int(m.sum())
            if m.any():
                recent[i] = np.concatenate([recent[i], cleaned[m]])[-buffer_size:]
                spikes[i].extend(t_fit[m].tolist())
            if m.sum() >= cfg.track_min_spikes:
                new = np.median(cleaned[m], axis=0)
                # A neuron changes slowly. A chunk whose median looks
                # nothing like the template is another neuron's spikes
                # landing on it, and must not drag the template along.
                r = np.corrcoef(new.ravel(), temps[i].ravel())[0, 1]
                ratio = -new.min() / max(-temps[i].min(), 1e-9)
                if r > 0.9 and 0.7 < ratio < 1.4:
                    temps[i] = (1 - cfg.track_alpha) * temps[i] + cfg.track_alpha * new

        # Merge templates that have become the same neuron.
        before = set(temps)
        merged += try_merges(cleaned, unit)
        for i in before - set(temps):
            merged_at[i] = c

        # Discovery: what the templates could not explain. Only events away
        # from every fitted spike count; the rest are misfit residues of
        # spikes we did explain, and those cluster all too well.
        if cfg.track_discovery:
            seg = residual[c0:c1]
            ev = detect_spikes(seg, fs, cfg.threshold, 2.0, method="energy")
            if ev.size and t_fit.size:
                fitted = np.sort(t_fit - c0)
                i = np.clip(np.searchsorted(fitted, ev), 1, fitted.size - 1)
                gap = np.minimum(np.abs(fitted[i] - ev), np.abs(fitted[i - 1] - ev))
                ev = ev[gap > int(round(1.5e-3 * fs))]
            wf, _ = extract_waveforms(seg, ev, fs, cfg.pre_ms, cfg.template_post_ms, align=cfg.align)
            if wf.size:
                buffer_wf.append(wf)
            n_buf = sum(w.shape[0] for w in buffer_wf)
            last = c == n_chunks - 1
            if n_buf >= cfg.track_discovery_min or (last and n_buf >= 2 * cfg.min_template_spikes):
                wf = np.concatenate(buffer_wf)
                buffer_wf = []
                noise = mad_noise(residual[max(0, c1 - 10 * chunk):c1])
                new_units = _discover(wf, noise, list(temps.values()), cfg)
                for t, w in new_units:
                    nid = max(list(temps) + list(alias)) + 1
                    temps[nid] = t
                    counts[nid] = 0
                    history[nid] = {}
                    recent[nid] = w[-buffer_size:].astype(np.float64)
                    spikes[nid] = []
                    discovered += 1

        for i in temps:
            history[i][c] = temps[i].copy()

    # Pass two: fixed per-chunk templates, every unit fitted in every chunk.
    final_ids = sorted(temps)
    residual = np.array(x, dtype=np.float32, copy=True)
    all_times, all_labels, all_amps = [], [], []
    for c in range(n_chunks):
        c0, c1 = int(edges[c]), int(edges[c + 1])
        ids_c, temps_c = [], []
        for i in list(history):
            h = history[i]
            if not h or c > merged_at.get(i, n_chunks):
                continue
            cs = sorted(h)
            # The template recorded for this chunk, or the earliest one
            # after it for a unit discovered later.
            at = c if c in h else next((k for k in cs if k > c), cs[-1])
            ids_c.append(i)
            temps_c.append(h[at])
        starts, lab, amps, _, times, _ = match_chunk(residual, c0, c1, margin, temps_c, cfg)
        all_times.append(times)
        all_labels.append(np.array([resolve(ids_c[int(j)]) for j in lab], dtype=np.int64))
        all_amps.append(amps)

    times = np.concatenate(all_times) if all_times else np.empty(0, dtype=np.int64)
    labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int64)
    amps = np.concatenate(all_amps) if all_amps else np.empty(0)
    order = np.argsort(times, kind="stable")
    times, labels, amps = times[order], labels[order], amps[order]

    # Units that ended up with almost no spikes are leftovers of discovery
    # (a cluster of collision residues that explained a handful of events);
    # drop them and their spikes.
    sizes = {i: int((labels == i).sum()) for i in final_ids}
    present = [i for i in final_ids if sizes[i] >= cfg.track_min_unit_spikes]
    keep = np.isin(labels, present)
    times, labels, amps = times[keep], labels[keep], amps[keep]
    remap = {i: j for j, i in enumerate(sorted(present, key=lambda i: -sizes[i]))}
    labels = np.array([remap[int(i)] for i in labels], dtype=np.int64)
    final_templates = {remap[i]: temps[i] for i in present}
    hist = {remap[i]: history[i] for i in present}
    return TrackResult(times, labels, amps, residual, final_templates, edges, hist, discovered, merged, log)


def refractory_overlap(ta: np.ndarray, tb: np.ndarray, fs: float, refractory_ms: float = 1.5,
                       baseline_ms: tuple[float, float] = (3.0, 20.0)) -> tuple[int, float]:
    """Spike pairs across two trains closer than the refractory period, and how
    many chance would predict from the pair density at 3 to 20 ms.

    A neuron split in two gives (close, expected) like (0, 6). Two
    independent neurons give close about equal to expected.
    """
    if ta.size == 0 or tb.size == 0:
        return 0, 0.0
    ta, tb = np.sort(ta), np.sort(tb)
    r = refractory_ms * 1e-3 * fs
    lo_b, hi_b = baseline_ms[0] * 1e-3 * fs, baseline_ms[1] * 1e-3 * fs
    close = 0
    base = 0
    for t in ta:
        i0 = np.searchsorted(tb, t - hi_b, side="left")
        i1 = np.searchsorted(tb, t + hi_b, side="right")
        d = np.abs(tb[i0:i1] - t)
        close += int((d < r).sum())
        base += int(((d >= lo_b) & (d < hi_b)).sum())
    expected = base * (2 * r) / (2 * (hi_b - lo_b))
    return close, float(expected)


def _discover(wf: np.ndarray, noise: np.ndarray, existing: list[np.ndarray], cfg) -> list:
    """Cluster unexplained waveforms and return (template, waveforms) of any new unit.

    Discovery is a backup for units stage one missed, so it is
    conservative: a candidate must pass the usual template selection with
    at least 50 spikes, have a peak-to-trough above 5 sigma on its best
    channel, and not look like a unit we already have (template
    correlation above 0.95 with amplitude within 30%), which would be the
    misfit left behind by that unit rather than a neuron. Noise events
    that cluster together are the main thing this has to reject.
    """
    if wf.shape[0] < 2 * cfg.min_template_spikes:
        return []
    feats, _ = waveform_features(wf, cfg.n_pcs, per_channel=cfg.per_channel_pca)
    labels, _ = gmm_cluster(feats, range(1, 9), seed=cfg.seed)
    labels = merge_similar(wf, labels, cfg.merge_corr, cfg.merge_amp_ratio, cfg.merge_dprime)
    kept, _ = select_templates(wf, labels, noise, max(cfg.min_template_spikes, 50),
                               max_outlier_frac=cfg.max_outlier_frac)
    out = []
    for k, tmpl in kept.items():
        ch = int(np.argmin(tmpl.min(axis=0)))
        if (tmpl[:, ch].max() - tmpl[:, ch].min()) / noise[ch] < 5.0:
            continue
        dup = False
        for t in existing:
            r = np.corrcoef(tmpl.ravel(), t.ravel())[0, 1]
            ratio = -tmpl.min() / -t.min()
            if r > 0.95 and 0.7 < ratio < 1.4:
                dup = True
                break
        if not dup:
            out.append((tmpl, wf[labels == k]))
    return out
