"""Clustering: Gaussian mixture with BIC model selection, then split and merge passes."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def gmm_cluster(
    features: np.ndarray,
    k_range: range = range(1, 13),
    n_init: int = 3,
    max_fit_spikes: int = 20_000,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Fit GMMs over `k_range` and keep the one with the lowest BIC.

    Features are standardised first. For speed, models are fit on at most
    `max_fit_spikes` randomly chosen spikes, then every spike is labelled.

    Returns (labels, info) where info has the BIC curve and the chosen k.
    """
    rng = np.random.default_rng(seed)
    mu, sd = features.mean(axis=0), features.std(axis=0) + 1e-12
    X = (features - mu) / sd
    fit_idx = rng.permutation(X.shape[0])[:max_fit_spikes]
    Xfit = X[fit_idx]

    bics, models = [], []
    for k in k_range:
        if k > Xfit.shape[0]:
            break
        gm = GaussianMixture(k, covariance_type="full", n_init=n_init,
                             reg_covar=1e-4, random_state=seed).fit(Xfit)
        bics.append(gm.bic(Xfit))
        models.append(gm)
    best = int(np.argmin(bics))
    labels = models[best].predict(X)
    return labels, {"k_range": list(k_range)[: len(bics)], "bic": bics,
                    "k": models[best].n_components}


def templates_from_labels(waveforms: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Median waveform per cluster. Median, not mean, so overlaps barely move it."""
    return {int(k): np.median(waveforms[labels == k], axis=0) for k in np.unique(labels)}


def separation(waveforms: np.ndarray, labels: np.ndarray, a: int, b: int, seed: int = 0) -> float:
    """d' between two clusters along the line joining their mean waveforms.

    Each spike is projected onto the unit vector between the two cluster
    means, and d' = |mean_a - mean_b| / sqrt((var_a + var_b) / 2) on that
    axis. Two halves of one neuron give small d' (the clouds touch). Two
    neurons with similar shapes but tight, distinct clouds give large d',
    even when their templates correlate highly.

    The score is cross-fitted: each cluster is halved at random, the axis is
    estimated from one half of each and d' is scored on the other halves,
    then the roles swap and the two scores are averaged. Scoring on the same
    spikes that defined the axis inflates d' in high dimensions (a random
    split of a single neuron reads as d' of about 1 with 100 spikes per half
    in 60 dimensions), which would block legitimate merges of small clusters.
    """
    wa = waveforms[labels == a].reshape(np.sum(labels == a), -1)
    wb = waveforms[labels == b].reshape(np.sum(labels == b), -1)
    if wa.shape[0] < 4 or wb.shape[0] < 4:
        return _dprime(wa, wb, wa, wb)
    rng = np.random.default_rng(seed)
    ia, ib = rng.permutation(wa.shape[0]), rng.permutation(wb.shape[0])
    a1, a2 = wa[ia[: ia.size // 2]], wa[ia[ia.size // 2:]]
    b1, b2 = wb[ib[: ib.size // 2]], wb[ib[ib.size // 2:]]
    return (_dprime(a1, b1, a2, b2) + _dprime(a2, b2, a1, b1)) / 2


def _dprime(fit_a: np.ndarray, fit_b: np.ndarray, test_a: np.ndarray, test_b: np.ndarray) -> float:
    """d' of the test sets along the axis between the fit sets' means."""
    axis = fit_b.mean(axis=0) - fit_a.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return 0.0
    axis /= norm
    pa, pb = test_a @ axis, test_b @ axis
    pooled = np.sqrt((pa.var() + pb.var()) / 2) + 1e-12
    return float(abs(pb.mean() - pa.mean()) / pooled)


def gap_ratio(waveforms: np.ndarray, labels: np.ndarray, a: int, b: int, seed: int = 0) -> float:
    """How deep the density valley is between two clusters, 0 (empty gap) to about 1 (none).

    Both clusters are projected onto the axis joining their means (axis
    from one random half of each, projection of the other halves, so the
    projection is not flattered by the axis having been fitted to it).
    A kernel density of the pooled projection is averaged over the middle
    third of the stretch between the two medians and over its outer
    thirds; the ratio is middle over outer.

    Two neurons give a valley. One neuron whose amplitude slid
    continuously during the recording, which a Gaussian mixture cuts into
    two or three bands, gives a flat top: the bands are adjacent slices of
    a continuum with no gap between them. In practice, with 300 spikes per
    cluster, two Gaussians at d' = 3 read 0.75 to 1.1, at d' = 4 about
    0.55, and a continuum cut in half 0.9 to 1.1, so the test separates
    clearly distinct neurons from a continuum but not marginal ones.
    """
    wa = waveforms[labels == a].reshape(np.sum(labels == a), -1)
    wb = waveforms[labels == b].reshape(np.sum(labels == b), -1)
    if wa.shape[0] < 8 or wb.shape[0] < 8:
        return 0.0
    rng = np.random.default_rng(seed)
    ia, ib = rng.permutation(wa.shape[0]), rng.permutation(wb.shape[0])
    fa, ta = wa[ia[: ia.size // 2]], wa[ia[ia.size // 2:]]
    fb, tb = wb[ib[: ib.size // 2]], wb[ib[ib.size // 2:]]
    axis = fb.mean(axis=0) - fa.mean(axis=0)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return 1.0
    axis /= norm
    pa, pb = ta @ axis, tb @ axis
    pooled = np.concatenate([pa, pb])
    sd = np.sqrt((pa.var() + pb.var()) / 2) + 1e-9
    bw = sd / 2
    ma, mb = np.median(pa), np.median(pb)
    if mb < ma:
        ma, mb = mb, ma
    span = mb - ma
    grid = ma + span * np.linspace(0.0, 1.0, 21)
    dens = np.array([np.exp(-0.5 * ((pooled - g) / bw) ** 2).sum() for g in grid])
    # Mean density over the middle third of the segment between the two
    # medians, against the mean over the outer thirds. Averaging over a
    # stretch rather than reading single points keeps kernel noise from
    # inventing or hiding a valley.
    middle = dens[7:14].mean()
    outer = np.concatenate([dens[:7], dens[14:]]).mean()
    return float(middle / max(outer, 1e-12))


def merge_similar(
    waveforms: np.ndarray,
    labels: np.ndarray,
    corr_threshold: float = 0.97,
    amp_ratio_threshold: float = 0.85,
    dprime_threshold: float = 2.5,
    gap_threshold: float | None = None,
    gap_amp_ratio_threshold: float = 0.5,
) -> np.ndarray:
    """Merge clusters that look like the same neuron.

    BIC tends to over-split: one neuron whose amplitude varies a bit, or whose
    cloud is not quite Gaussian, gets two components. Two clusters merge when
    all three hold:

    * their flattened templates correlate above `corr_threshold`,
    * their peak amplitudes are within `amp_ratio_threshold` of each other,
    * their spike clouds overlap, i.e. `separation` is below `dprime_threshold`.

    The last check matters at low noise, where two different neurons can have
    near-identical templates yet perfectly separable clusters.

    A second route, off by default, is meant for a neuron that drifted:
    same shape, amplitudes within `gap_amp_ratio_threshold` (drift can
    change the amplitude a lot), and no density gap between the clouds
    (`gap_ratio` above `gap_threshold`, 0.85 is a reasonable value). It is
    off because with a few hundred spikes per cluster the gap ratio of two
    neurons at d' of 3 overlaps that of a continuum, and because a neuron
    under oscillating drift dwells at the extremes and has a real gap.

    Repeats until nothing else merges. Returned labels are renumbered 0..K-1 by size.
    """
    labels = labels.copy()
    while True:
        temps = templates_from_labels(waveforms, labels)
        keys = sorted(temps)
        best = None
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ta, tb = temps[a].ravel(), temps[b].ravel()
                r = np.corrcoef(ta, tb)[0, 1]
                pa, pb = -temps[a].min(), -temps[b].min()
                ratio = min(pa, pb) / max(pa, pb)
                if r <= corr_threshold or ratio <= gap_amp_ratio_threshold:
                    continue
                overlap = (ratio > amp_ratio_threshold
                           and separation(waveforms, labels, a, b) < dprime_threshold)
                continuum = (gap_threshold is not None
                             and gap_ratio(waveforms, labels, a, b) > gap_threshold)
                if not (overlap or continuum):
                    continue
                if best is None or r > best[0]:
                    best = (r, a, b)
        if best is None:
            break
        _, a, b = best
        labels[labels == b] = a
    return relabel_by_size(labels)


def split_bimodal(
    waveforms: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    dprime_threshold: float = 4.0,
    min_size: int = 30,
    max_rounds: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Split clusters that hide two neurons.

    The mirror image of `merge_similar`. BIC sometimes under-splits, most
    often at low noise, where real clusters are so tight that the scattered
    collision spikes look like better value for an extra component.

    For every cluster, the spikes are divided at random into a fit half and a
    test half. A two-component GMM is fit on the fit half, and the axis
    between its two mean waveforms is taken from the fit half too. The test
    half is then labelled by that GMM and projected onto that axis, and d'
    is computed there. Held-out scoring matters: a split chosen to separate a
    set of spikes will always look separated when scored on those same
    spikes, especially in a few hundred waveform dimensions. The split is
    kept if held-out d' exceeds `dprime_threshold` and both sides have at
    least `min_size` spikes. Repeats until no cluster splits.
    """
    rng = np.random.default_rng(seed)
    labels = labels.copy()
    for _ in range(max_rounds):
        changed = False
        for k in np.unique(labels):
            idx = np.flatnonzero(labels == k)
            if idx.size < 4 * min_size:
                continue
            X = features[idx]
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
            W = waveforms[idx].reshape(idx.size, -1)

            perm = rng.permutation(idx.size)
            fit, test = perm[: idx.size // 2], perm[idx.size // 2:]
            gm = GaussianMixture(2, covariance_type="full", n_init=3, reg_covar=1e-3,
                                 random_state=seed).fit(X[fit])
            sub_fit, sub_test = gm.predict(X[fit]), gm.predict(X[test])
            if min(np.bincount(sub_fit, minlength=2)) < min_size // 2:
                continue
            if min(np.bincount(sub_test, minlength=2)) < min_size // 2:
                continue

            axis = W[fit][sub_fit == 1].mean(axis=0) - W[fit][sub_fit == 0].mean(axis=0)
            axis /= np.linalg.norm(axis) + 1e-12
            p0, p1 = W[test][sub_test == 0] @ axis, W[test][sub_test == 1] @ axis
            d = abs(p1.mean() - p0.mean()) / (np.sqrt((p0.var() + p1.var()) / 2) + 1e-12)
            if d <= dprime_threshold:
                continue

            sub_all = gm.predict(X)
            if min(np.bincount(sub_all, minlength=2)) < min_size:
                continue
            labels[idx[sub_all == 1]] = labels.max() + 1
            changed = True
        if not changed:
            break
    return labels


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 0..K-1, largest first."""
    ks, counts = np.unique(labels, return_counts=True)
    order = ks[np.argsort(-counts, kind="stable")]
    remap = {int(k): i for i, k in enumerate(order)}
    return np.array([remap[int(k)] for k in labels], dtype=np.int64)
