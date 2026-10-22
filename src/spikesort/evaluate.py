"""Score a sorting against ground truth, plus ground-truth-free quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Comparison:
    """Result of matching sorted units to ground-truth units."""

    match_counts: np.ndarray  # (n_gt_units, n_sorted_units) matched spike counts
    gt_counts: np.ndarray  # spikes per ground-truth unit
    sorted_counts: np.ndarray  # spikes per sorted unit
    assignment: dict[int, int | None]  # gt unit -> sorted unit (or None)
    accuracy: np.ndarray  # per gt unit
    precision: np.ndarray
    recall: np.ndarray

    def summary(self) -> str:
        lines = ["unit  matched  n_gt   n_sorted  precision  recall  accuracy"]
        for u, k in self.assignment.items():
            n_sorted = self.sorted_counts[k] if k is not None else 0
            lines.append(
                f"{u:>4}  {str(k):>7}  {self.gt_counts[u]:>5}  {n_sorted:>8}  "
                f"{self.precision[u]:>9.3f}  {self.recall[u]:>6.3f}  {self.accuracy[u]:>8.3f}"
            )
        lines.append(f"mean accuracy: {self.accuracy.mean():.3f}")
        return "\n".join(lines)


def _count_matches(gt: np.ndarray, sorted_: np.ndarray, delta: int) -> int:
    """Number of ground-truth spikes with a sorted spike within +/- delta samples.

    Each sorted spike can be used once, so a burst cannot match one event twice.
    """
    if gt.size == 0 or sorted_.size == 0:
        return 0
    used = np.zeros(sorted_.size, dtype=bool)
    n = 0
    lo_all = np.searchsorted(sorted_, gt - delta, side="left")
    hi_all = np.searchsorted(sorted_, gt + delta, side="right")
    for g, lo, hi in zip(gt, lo_all, hi_all, strict=True):
        if lo >= hi:
            continue
        cand = np.arange(lo, hi)
        cand = cand[~used[cand]]
        if cand.size:
            j = cand[np.argmin(np.abs(sorted_[cand] - g))]
            used[j] = True
            n += 1
    return n


def compare_to_ground_truth(
    gt_times: np.ndarray,
    gt_labels: np.ndarray,
    sorted_times: np.ndarray,
    sorted_labels: np.ndarray,
    fs: float,
    delta_ms: float = 0.4,
) -> Comparison:
    """Match sorted units to ground-truth units with the Hungarian algorithm.

    Per-pair accuracy follows the SpikeForest definition,
    n_match / (n_gt + n_sorted - n_match), which penalises misses and false
    positives together. The assignment maximises total accuracy.
    """
    delta = int(round(delta_ms * 1e-3 * fs))
    gt_units = np.unique(gt_labels)
    sorted_units = np.unique(sorted_labels)
    gt_trains = {u: np.sort(gt_times[gt_labels == u]) for u in gt_units}
    so_trains = {k: np.sort(sorted_times[sorted_labels == k]) for k in sorted_units}

    n_gt = np.array([gt_trains[u].size for u in gt_units])
    n_so = np.array([so_trains[k].size for k in sorted_units])
    M = np.zeros((gt_units.size, sorted_units.size), dtype=np.int64)
    for i, u in enumerate(gt_units):
        for j, k in enumerate(sorted_units):
            M[i, j] = _count_matches(gt_trains[u], so_trains[k], delta)

    acc_matrix = M / (n_gt[:, None] + n_so[None, :] - M)
    rows, cols = linear_sum_assignment(-acc_matrix)

    assignment: dict[int, int | None] = {int(u): None for u in gt_units}
    acc = np.zeros(gt_units.size)
    prec = np.zeros(gt_units.size)
    rec = np.zeros(gt_units.size)
    for i, j in zip(rows, cols, strict=True):
        if M[i, j] == 0:
            continue
        assignment[int(gt_units[i])] = int(sorted_units[j])
        acc[i] = acc_matrix[i, j]
        prec[i] = M[i, j] / n_so[j]
        rec[i] = M[i, j] / n_gt[i]

    return Comparison(M, n_gt, n_so, assignment, acc, prec, rec)


# Quality metrics that do not need ground truth. These are what you would
# actually report on real data.

def isi_violation_rate(times: np.ndarray, fs: float, refractory_ms: float = 1.5) -> float:
    """Fraction of inter-spike intervals shorter than the refractory period.

    A well-isolated single neuron should be near zero.
    """
    if times.size < 2:
        return 0.0
    isi = np.diff(np.sort(times)) / fs * 1e3
    return float((isi < refractory_ms).mean())


def snr(template: np.ndarray, noise: np.ndarray) -> float:
    """Peak-to-trough amplitude on the best channel divided by that channel's noise."""
    ch = int(np.argmin(template.min(axis=0)))
    return float((template[:, ch].max() - template[:, ch].min()) / noise[ch])


def isolation_distance(features: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Harris et al. 2001.

    Squared Mahalanobis distance (in the cluster's own metric) of the n-th
    closest non-member spike, where n is the cluster size. Larger means
    better separated. Undefined (nan) if there are fewer non-members than members.
    """
    inside = features[labels == k]
    outside = features[labels != k]
    n = inside.shape[0]
    if n <= features.shape[1] or outside.shape[0] < n:
        return float("nan")
    mu = inside.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(inside, rowvar=False))
    d = outside - mu
    d2 = np.einsum("ij,jk,ik->i", d, cov_inv, d)
    return float(np.partition(d2, n - 1)[n - 1])
