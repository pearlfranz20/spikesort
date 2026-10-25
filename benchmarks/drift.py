"""Benchmark: accuracy under probe drift, with and without correction.

Two probes. A tetrode with 5 units, and a 16-channel linear probe (20 um
pitch) with 8 units placed inside the array, away from its ends. Motion is
a slow sinusoid along the probe axis (one and a half cycles over the
recording) at increasing amplitude. "Corrected" estimates the drift from
spike depths and re-interpolates the channels before sorting (drift.py).
"Tracked" is the experimental template-tracking stage two (tracking.py),
run on the tetrode, where there is nothing to interpolate along: once with
templates from the whole recording and once with templates from the first
10 s only.

    python benchmarks/drift.py --out benchmarks/results
    python benchmarks/drift.py --plot-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spikesort import (
    SortConfig,
    compare_to_ground_truth,
    linear_probe_geometry,
    simulate_recording,
    sort,
    tetrode_geometry,
)
from spikesort.plotting import plot_drift

PROBES = {
    "tetrode": dict(geometry=tetrode_geometry(), n_units=5, margin=15.0),
    "linear16": dict(geometry=linear_probe_geometry(16), n_units=8, margin=-30.0),
}
MOTIONS = [0, 10, 20, 30, 40]
MODES = {
    "tetrode": [("plain", SortConfig()), ("corrected", SortConfig(drift_correction=True)),
                ("tracked", SortConfig(tracking=True)),
                ("tracked_init10", SortConfig(tracking=True, track_init_s=10.0))],
    "linear16": [("plain", SortConfig()), ("corrected", SortConfig(drift_correction=True))],
}
STYLE = {("tetrode", "plain"): ("#8a8f98", "--", "tetrode, uncorrected"),
         ("tetrode", "corrected"): ("#8a8f98", "-", "tetrode, interpolation"),
         ("tetrode", "tracked"): ("#c4405a", ":", "tetrode, tracking"),
         ("tetrode", "tracked_init10"): ("#c4405a", "-.", "tetrode, tracking from first 10 s"),
         ("linear16", "plain"): ("#2a6fdb", "--", "16-ch linear, uncorrected"),
         ("linear16", "corrected"): ("#2a6fdb", "-", "16-ch linear, interpolation")}


def run_one(probe: str, motion: float, seed: int, duration: float) -> list[dict]:
    spec = PROBES[probe]
    rec = simulate_recording(n_units=spec["n_units"], duration=duration, seed=seed,
                             channel_positions=spec["geometry"], motion_um=motion,
                             unit_margin_um=spec["margin"])
    rows = []
    for mode, cfg in MODES[probe]:
        s = sort(rec.data, rec.fs, cfg, channel_positions=spec["geometry"])
        c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
        row = {"probe": probe, "motion": motion, "seed": seed, "mode": mode,
               "accuracy": float(c.accuracy.mean()), "precision": float(c.precision.mean()),
               "recall": float(c.recall.mean()), "n_found": s.n_units, "n_units": spec["n_units"]}
        if s.drift is not None:
            true = -rec.motion_at(s.drift.bin_centres_s)
            true = true - true.mean()
            row["drift_rms_error_um"] = float(np.sqrt(np.mean((s.drift.displacement_um - true) ** 2)))
        rows.append(row)
    return rows


def example_figure(out: Path, duration: float):
    spec = PROBES["linear16"]
    rec = simulate_recording(n_units=spec["n_units"], duration=duration, seed=0,
                             channel_positions=spec["geometry"], motion_um=20,
                             unit_margin_um=spec["margin"])
    s = sort(rec.data, rec.fs, SortConfig(drift_correction=True), channel_positions=spec["geometry"])
    fig = plot_drift(s, true_displacement=lambda t: -rec.motion_at(t))
    fig.savefig(out / "drift_map.png", dpi=150)
    print(f"saved {out}/drift_map.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        rows = []
        for probe in PROBES:
            for motion in MOTIONS:
                for seed in range(args.seeds):
                    for r in run_one(probe, motion, seed, args.duration):
                        rows.append(r)
                        print(f"{probe:>8} motion={motion:>2} seed={seed} {r['mode']:>14} "
                              f"acc={r['accuracy']:.3f} prec={r['precision']:.3f} found={r['n_found']}",
                              flush=True)
        (args.out / "drift.json").write_text(json.dumps(rows, indent=2))
        example_figure(args.out, args.duration)
    rows = json.loads((args.out / "drift.json").read_text())
    plot(rows, args.out)
    table(rows)


def _mean(rows, probe, mode, motion, key):
    v = [r[key] for r in rows if r["probe"] == probe and r["mode"] == mode
         and r["motion"] == motion and key in r]
    return float(np.mean(v)) if v else float("nan")


def plot(rows, out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for (probe, mode), (color, ls, label) in STYLE.items():
        m = [_mean(rows, probe, mode, mo, "accuracy") for mo in MOTIONS]
        ax.plot(MOTIONS, m, "o", ls=ls, color=color, ms=4, label=label)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("peak probe motion (um)", fontsize=9)
    ax.set_ylabel("accuracy", fontsize=9)
    ax.set_title("Accuracy under drift", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.savefig(out / "drift.png", dpi=150)
    print(f"saved {out}/drift.png")


def table(rows):
    cols = [("tetrode", "plain"), ("tetrode", "corrected"), ("tetrode", "tracked"),
            ("tetrode", "tracked_init10"), ("linear16", "plain"), ("linear16", "corrected")]
    print("\n| Peak motion (um) | Tetrode | Tetrode, interpolation | Tetrode, tracking "
          "| Tetrode, tracking from first 10 s | Linear 16-ch | Linear, interpolation "
          "| Drift estimate error (um) |")
    print("|---|---|---|---|---|---|---|---|")
    for mo in MOTIONS:
        cells = [f"{_mean(rows, p, m, mo, 'accuracy'):.2f}" for p, m in cols]
        err = _mean(rows, "linear16", "corrected", mo, "drift_rms_error_um")
        print(f"| {mo} | {' | '.join(cells)} | {err:.1f} |")
    print("\nprecision:")
    for mo in MOTIONS:
        cells = [f"{_mean(rows, p, m, mo, 'precision'):.2f}" for p, m in cols]
        print(f"| {mo} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
