"""Command line entry point.

    spikesort demo --units 5 --duration 60 --seed 0 --out results/
    spikesort sort recording.npy --fs 30000 --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import compare_to_ground_truth
from .pipeline import SortConfig, sort
from .simulate import linear_probe_geometry, simulate_recording, tetrode_geometry


def _save(sorting, out: Path, comparison=None):
    from . import plotting

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "sorting.npz", times=sorting.times, labels=sorting.labels)
    quality = sorting.quality_table()
    (out / "quality.json").write_text(json.dumps(quality, indent=2, default=float))

    fig = plotting.summary_figure(sorting, comparison)
    fig.savefig(out / "summary.png", dpi=150)
    fig = plotting.plot_templates(sorting)
    fig.savefig(out / "templates.png", dpi=150, bbox_inches="tight")

    print(f"{sorting.times.size} spikes in {sorting.n_units} units")
    print("unit  n_spikes  rate_hz   snr  isi_viol  iso_dist")
    for q in quality:
        print(f"{q['unit']:>4}  {q['n_spikes']:>8}  {q['rate_hz']:>7.2f}  {q['snr']:>4.1f}  "
              f"{q['isi_violations']:>8.4f}  {q['isolation_distance']:>8.1f}")
    print(f"saved results to {out}/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="spikesort")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="simulate a recording, sort it, score it")
    d.add_argument("--units", type=int, default=5)
    d.add_argument("--duration", type=float, default=60.0)
    d.add_argument("--noise", type=float, default=10.0, help="noise SD in microvolts")
    d.add_argument("--probe", choices=["tetrode", "linear"], default="tetrode")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--split", action="store_true", help="enable the bimodal split pass")
    d.add_argument("--out", type=Path, default=Path("results"))

    s = sub.add_parser("sort", help="sort a .npy array of shape (n_samples, n_channels)")
    s.add_argument("path", type=Path)
    s.add_argument("--fs", type=float, required=True)
    s.add_argument("--threshold", type=float, default=5.0)
    s.add_argument("--cmr", action="store_true", help="common median reference")
    s.add_argument("--out", type=Path, default=Path("results"))

    args = p.parse_args(argv)

    if args.cmd == "demo":
        geom = tetrode_geometry() if args.probe == "tetrode" else linear_probe_geometry()
        rec = simulate_recording(n_units=args.units, duration=args.duration,
                                 noise_uv=args.noise, channel_positions=geom, seed=args.seed)
        sorting = sort(rec.data, rec.fs, SortConfig(seed=args.seed, split_dprime=4.0 if args.split else None))
        comp = compare_to_ground_truth(rec.spike_times, rec.spike_labels,
                                       sorting.times, sorting.labels, rec.fs)
        _save(sorting, args.out, comp)
        print()
        print(comp.summary())
    else:
        data = np.load(args.path)
        cfg = SortConfig(threshold=args.threshold, use_cmr=args.cmr)
        _save(sort(data, args.fs, cfg), args.out)


if __name__ == "__main__":
    main()
