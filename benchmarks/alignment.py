"""Benchmark: how much does spike alignment matter?

Compares integer alignment, parabolic sub-sample refinement and
centre-of-mass alignment across noise levels and unit counts. Stage one
only (no template matching, no split pass) with the classic threshold
detector, since alignment is a stage-one question and the table predates
the energy detector and the split pass.

    python benchmarks/alignment.py --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spikesort import SortConfig, compare_to_ground_truth, simulate_recording, sort

MODES = ["none", "parabola", "com"]
CONDITIONS = [(5, 5), (5, 10), (5, 15), (5, 20), (5, 25), (2, 10), (8, 10)]  # (units, noise)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--duration", type=float, default=60.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_units, noise in CONDITIONS:
        for seed in range(args.seeds):
            rec = simulate_recording(n_units=n_units, noise_uv=noise, duration=args.duration, seed=seed)
            for mode in MODES:
                cfg = SortConfig(align=mode, template_matching=False, detect_method="threshold",
                                 dead_time_ms=1.0, split_dprime=None)
                s = sort(rec.data, rec.fs, cfg)
                c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
                rows.append({"n_units": n_units, "noise": noise, "seed": seed, "align": mode,
                             "accuracy": float(c.accuracy.mean()), "found": s.n_units})
    (args.out / "alignment.json").write_text(json.dumps(rows, indent=2))

    def mean(mode, **cond):
        return np.mean([r["accuracy"] for r in rows
                        if r["align"] == mode and all(r[k] == v for k, v in cond.items())])

    print("| units | noise (uV) | integer | parabola | centre of mass |")
    print("|---|---|---|---|---|")
    for n_units, noise in CONDITIONS:
        cells = " | ".join(f"{mean(m, n_units=n_units, noise=noise):.2f}" for m in MODES)
        print(f"| {n_units} | {noise} | {cells} |")
    print(f"| **all** | | {' | '.join(f'**{mean(m):.2f}**' for m in MODES)} |")


if __name__ == "__main__":
    main()
