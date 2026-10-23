"""Benchmark: accuracy against noise and unit count, with and without template matching.

Three sorters per condition:

* cluster   stage one only (detect, align, cluster, merge)
* matching  the full pipeline (stage one, then template matching)
* oracle    template matching with templates built from the ground-truth
            labels, an upper bound on what matching can do with perfect
            stage-one output

Ground-truth spikes are also split into "isolated" and "collision" (another
unit fired within 1 ms) to show where each sorter loses spikes.

    python benchmarks/sweep.py --out benchmarks/results
    python benchmarks/sweep.py --plot-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spikesort import SortConfig, compare_to_ground_truth, simulate_recording, sort
from spikesort.cluster import templates_from_labels
from spikesort.detect import extract_waveforms
from spikesort.evaluate import collision_mask, hit_mask
from spikesort.matching import match_templates
from spikesort.preprocess import bandpass

MODES = ["cluster", "matching", "oracle"]
COLORS = {"cluster": "#8a8f98", "matching": "#2a6fdb", "oracle": "#e0782b"}
LABELS = {"cluster": "clustering only", "matching": "with template matching",
          "oracle": "matching, true templates"}


def oracle_sort(rec, cfg: SortConfig):
    """Template matching with templates from the ground-truth labels."""
    x = bandpass(rec.data, rec.fs, cfg.low_hz, cfg.high_hz)
    wf, _, ok = extract_waveforms(x, rec.spike_times, rec.fs, cfg.pre_ms, cfg.template_post_ms,
                                  align="none", return_mask=True)
    temps = templates_from_labels(wf, rec.spike_labels[ok])
    m = match_templates(x, temps, cfg.match_amp_range, cfg.match_threshold,
                        cfg.match_min_z, cfg.match_max_rounds)
    return m.times, m.labels


def score(rec, times, labels, coll) -> dict:
    c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, times, labels, rec.fs)
    hit = hit_mask(rec.spike_times, rec.spike_labels, times, labels, c, rec.fs)
    return {
        "accuracy": float(c.accuracy.mean()),
        "precision": float(c.precision.mean()),
        "recall": float(c.recall.mean()),
        "n_found": int(np.unique(labels).size),
        "recall_isolated": float(hit[~coll].mean()),
        "recall_collision": float(hit[coll].mean()) if coll.any() else float("nan"),
    }


def run_one(n_units: int, noise: float, seed: int, duration: float) -> list[dict]:
    rec = simulate_recording(n_units=n_units, noise_uv=noise, duration=duration, seed=seed)
    coll = collision_mask(rec.spike_times, rec.spike_labels, rec.fs)
    base = {"n_units": n_units, "noise": noise, "seed": seed, "collision_frac": float(coll.mean())}
    rows = []
    s = sort(rec.data, rec.fs, SortConfig(template_matching=False))
    rows.append({**base, "mode": "cluster", **score(rec, s.times, s.labels, coll)})
    s = sort(rec.data, rec.fs, SortConfig())
    rows.append({**base, "mode": "matching", **score(rec, s.times, s.labels, coll)})
    t, lab = oracle_sort(rec, SortConfig())
    rows.append({**base, "mode": "oracle", **score(rec, t, lab, coll)})
    return rows


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
        rows = json.loads((args.out / "sweep.json").read_text())
        plot(rows, noise_levels, unit_counts, args.out)
        table(rows, noise_levels, unit_counts)
        return
    rows = []
    for nz in noise_levels:
        for seed in range(args.seeds):
            for r in run_one(5, nz, seed, args.duration):
                r["sweep"] = "noise"
                rows.append(r)
                print(f"noise={nz:>3} seed={seed} {r['mode']:>8} acc={r['accuracy']:.3f}", flush=True)
    for nu in unit_counts:
        for seed in range(args.seeds):
            for r in run_one(nu, 10, seed, args.duration):
                r["sweep"] = "units"
                rows.append(r)
                print(f"units={nu:>3} seed={seed} {r['mode']:>8} acc={r['accuracy']:.3f}", flush=True)
    (args.out / "sweep.json").write_text(json.dumps(rows, indent=2))
    plot(rows, noise_levels, unit_counts, args.out)
    table(rows, noise_levels, unit_counts)


def _agg(rows, sweep, mode, key, xs, xkey):
    vals = [[r[key] for r in rows if r["sweep"] == sweep and r["mode"] == mode and r[xkey] == x]
            for x in xs]
    return np.array([np.nanmean(v) for v in vals]), np.array([np.nanstd(v) for v in vals])


def plot(rows, noise_levels, unit_counts, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, sweep, xs, xkey, xlabel, title in [
        (axes[0], "noise", noise_levels, "noise", "noise SD (microvolts)", "Accuracy vs noise (5 units)"),
        (axes[1], "units", unit_counts, "n_units", "ground-truth units", "Accuracy vs unit count (noise 10)"),
    ]:
        for mode in MODES:
            m, sd = _agg(rows, sweep, mode, "accuracy", xs, xkey)
            ax.plot(xs, m, "o-", color=COLORS[mode], ms=4, label=LABELS[mode])
            ax.fill_between(xs, m - sd, m + sd, color=COLORS[mode], alpha=0.12, lw=0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[2]
    for mode in MODES:
        m, _ = _agg(rows, "units", mode, "recall_collision", unit_counts, "n_units")
        ax.plot(unit_counts, m, "o-", color=COLORS[mode], ms=4, label=LABELS[mode])
    m, _ = _agg(rows, "units", "matching", "recall_isolated", unit_counts, "n_units")
    ax.plot(unit_counts, m, "--", color=COLORS["matching"], lw=1, label="isolated spikes, matching")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ground-truth units", fontsize=9)
    ax.set_title("Recall on overlapping spikes (<1 ms apart)", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.savefig(out / "sweep.png", dpi=150)
    print(f"saved {out}/sweep.png")


def table(rows, noise_levels, unit_counts):
    """Markdown summary for the README."""
    print("\n| Condition | Clustering only | With matching | True templates |")
    print("|---|---|---|---|")
    for nz in noise_levels:
        cells = [f"{_agg(rows, 'noise', m, 'accuracy', [nz], 'noise')[0][0]:.2f}" for m in MODES]
        print(f"| 5 units, noise {nz} uV | {' | '.join(cells)} |")
    for nu in unit_counts:
        cells = [f"{_agg(rows, 'units', m, 'accuracy', [nu], 'n_units')[0][0]:.2f}" for m in MODES]
        print(f"| {nu} units, noise 10 uV | {' | '.join(cells)} |")
    print("\n| Condition | Clustering only | With matching | True templates |")
    print("|---|---|---|---|")
    print("| recall, isolated spikes | " + " | ".join(
        f"{np.nanmean(_agg(rows, 'units', m, 'recall_isolated', unit_counts, 'n_units')[0]):.2f}"
        for m in MODES) + " |")
    print("| recall, overlapping spikes | " + " | ".join(
        f"{np.nanmean(_agg(rows, 'units', m, 'recall_collision', unit_counts, 'n_units')[0]):.2f}"
        for m in MODES) + " |")
    print("| precision | " + " | ".join(
        f"{np.nanmean(_agg(rows, 'units', m, 'precision', unit_counts, 'n_units')[0]):.2f}"
        for m in MODES) + " |")


if __name__ == "__main__":
    main()
