"""Benchmark: how does sorting accuracy degrade with noise and with unit count?

Also checks where the misses come from by splitting ground-truth spikes into
"isolated" and "collision" (another unit fired within 1 ms).

    python benchmarks/sweep.py --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spikesort import compare_to_ground_truth, simulate_recording, sort


def collision_mask(times: np.ndarray, labels: np.ndarray, fs: float, window_ms: float = 1.0) -> np.ndarray:
    """True for ground-truth spikes with a spike from a different unit within the window."""
    w = int(round(window_ms * 1e-3 * fs))
    order = np.argsort(times)
    t, lab = times[order], labels[order]
    out = np.zeros(t.size, dtype=bool)
    lo = np.searchsorted(t, t - w, side="left")
    hi = np.searchsorted(t, t + w, side="right")
    for i in range(t.size):
        neigh = lab[lo[i]:hi[i]]
        out[i] = np.any(neigh != lab[i])
    res = np.empty_like(out)
    res[order] = out
    return res


def detected_mask(gt_times: np.ndarray, sorted_times: np.ndarray, delta: int) -> np.ndarray:
    """True for ground-truth spikes that have any sorted spike within delta."""
    s = np.sort(sorted_times)
    lo = np.searchsorted(s, gt_times - delta, side="left")
    hi = np.searchsorted(s, gt_times + delta, side="right")
    return hi > lo


def correctly_sorted_mask(rec, sorting, comp, delta: int) -> np.ndarray:
    """True for GT spikes matched to a sorted spike in the unit assigned to their GT unit."""
    out = np.zeros(rec.spike_times.size, dtype=bool)
    for u, k in comp.assignment.items():
        if k is None:
            continue
        idx = np.flatnonzero(rec.spike_labels == u)
        out[idx] = detected_mask(rec.spike_times[idx], sorting.times[sorting.labels == k], delta)
    return out


def run_one(n_units: int, noise: float, seed: int, duration: float) -> dict:
    rec = simulate_recording(n_units=n_units, noise_uv=noise, duration=duration, seed=seed)
    s = sort(rec.data, rec.fs)
    c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
    delta = int(round(0.4e-3 * rec.fs))
    coll = collision_mask(rec.spike_times, rec.spike_labels, rec.fs)
    ok = correctly_sorted_mask(rec, s, c, delta)
    return {
        "n_units": n_units, "noise": noise, "seed": seed,
        "accuracy": float(c.accuracy.mean()),
        "precision": float(c.precision.mean()),
        "recall": float(c.recall.mean()),
        "n_found": s.n_units,
        "collision_frac": float(coll.mean()),
        "recall_isolated": float(ok[~coll].mean()),
        "recall_collision": float(ok[coll].mean()) if coll.any() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--plot-only", action="store_true", help="redraw from saved sweep.json")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    noise_levels = [5, 10, 15, 20, 25, 30]
    unit_counts = [2, 4, 6, 8, 10]
    if args.plot_only:
        plot(json.loads((args.out / "sweep.json").read_text()), noise_levels, unit_counts, args.out)
        return
    rows = []
    for nz in noise_levels:
        for seed in range(args.seeds):
            r = run_one(5, nz, seed, args.duration); r["sweep"] = "noise"; rows.append(r)
            print(f"noise={nz:>3} seed={seed} acc={r['accuracy']:.3f}")
    for nu in unit_counts:
        for seed in range(args.seeds):
            r = run_one(nu, 10, seed, args.duration); r["sweep"] = "units"; rows.append(r)
            print(f"units={nu:>3} seed={seed} acc={r['accuracy']:.3f}")
    (args.out / "sweep.json").write_text(json.dumps(rows, indent=2))
    plot(rows, noise_levels, unit_counts, args.out)


def plot(rows, noise_levels, unit_counts, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    def agg(sweep, key, xs, xkey):
        m = [np.mean([r[key] for r in rows if r["sweep"] == sweep and r[xkey] == x]) for x in xs]
        sd = [np.std([r[key] for r in rows if r["sweep"] == sweep and r[xkey] == x]) for x in xs]
        return np.array(m), np.array(sd)

    for ax, sweep, xs, xkey, xlabel in [
        (axes[0], "noise", noise_levels, "noise", "noise SD (microvolts)"),
        (axes[1], "units", unit_counts, "n_units", "ground-truth units"),
    ]:
        for key, color in [("precision", "#2a6fdb"), ("recall", "#e0782b"), ("accuracy", "#2f9e6b")]:
            m, sd = agg(sweep, key, xs, xkey)
            ax.plot(xs, m, "o-", color=color, ms=4, label=key)
            ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.15, lw=0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_title("Accuracy vs noise (5 units)", fontsize=10, loc="left")
    axes[1].set_title("Accuracy vs unit count (noise 10)", fontsize=10, loc="left")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[2]
    xs = unit_counts
    iso, _ = agg("units", "recall_isolated", xs, "n_units")
    col, _ = agg("units", "recall_collision", xs, "n_units")
    ax.plot(xs, iso, "o-", color="#2a6fdb", ms=4, label="isolated spikes")
    ax.plot(xs, col, "o-", color="#c4405a", ms=4, label="collisions (<1 ms)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ground-truth units", fontsize=9)
    ax.set_title("Recall: isolated vs overlapping spikes", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.savefig(out / "sweep.png", dpi=150)
    print(f"saved {out}/sweep.png")


if __name__ == "__main__":
    main()
