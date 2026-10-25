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
    if sorting.residual is not None:
        fig = plotting.plot_deconvolution(sorting)
        fig.savefig(out / "deconvolution.png", dpi=150)

    s1 = sorting.stage1
    if sorting.matching:
        dropped = s1.get("dropped", {})
        drop_txt = f", {len(dropped)} dropped as composite" if dropped else ""
        print(f"stage 1: {s1['times'].size} events in {s1['n_clusters']} clusters{drop_txt}")
        if sorting.tracking:
            print(f"stage 2: tracking over {sorting.matching['rounds']} chunks, "
                  f"{sorting.tracking['merged']} merges, {sorting.tracking['discovered']} discovered, "
                  f"{sorting.times.size} spikes in {sorting.n_units} units")
        elif sorting.localization:
            loc = sorting.localization
            n_linked = sum(len(g) for g in loc["groups"])
            print(f"stage 2: {n_linked} clusters linked into {len(loc['groups'])} units over "
                  f"{loc['n_bins']} bins, {sorting.times.size} spikes in {sorting.n_units} units")
        else:
            print(f"stage 2: {sorting.matching['rounds']} matching rounds, "
                  f"{sorting.times.size} spikes in {sorting.n_units} units")
    else:
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
    d.add_argument("--no-split", action="store_true", help="disable the bimodal split pass")
    d.add_argument("--motion", type=float, default=0.0, help="peak probe motion in microns")
    d.add_argument("--drift-correction", action="store_true", help="estimate and undo probe drift")
    d.add_argument("--drift-method", choices=["interpolate", "positions"], default="interpolate",
                   help="re-sample channels (interpolate) or follow each unit in position space (positions)")
    d.add_argument("--tracking", action="store_true", help="experimental: templates follow their units")
    d.add_argument("--no-matching", action="store_true", help="stop after clustering (stage 1)")
    d.add_argument("--out", type=Path, default=Path("results"))

    s = sub.add_parser("sort", help="sort a .npy array of shape (n_samples, n_channels)")
    s.add_argument("path", type=Path)
    s.add_argument("--fs", type=float, required=True)
    s.add_argument("--threshold", type=float, default=5.0)
    s.add_argument("--cmr", action="store_true", help="common median reference")
    s.add_argument("--no-matching", action="store_true", help="stop after clustering (stage 1)")
    s.add_argument("--positions", type=Path, default=None,
                   help=".npy of channel positions (n_channels, 2) in microns, enables drift correction")
    s.add_argument("--drift-method", choices=["interpolate", "positions"], default="interpolate",
                   help="re-sample channels (interpolate) or follow each unit in position space (positions)")
    s.add_argument("--tracking", action="store_true", help="experimental: templates follow their units")
    s.add_argument("--out", type=Path, default=Path("results"))

    args = p.parse_args(argv)

    if args.cmd == "demo":
        geom = tetrode_geometry() if args.probe == "tetrode" else linear_probe_geometry(16)
        margin = 15.0 if args.probe == "tetrode" else -30.0
        rec = simulate_recording(n_units=args.units, duration=args.duration, noise_uv=args.noise,
                                 channel_positions=geom, seed=args.seed, motion_um=args.motion,
                                 unit_margin_um=margin)
        cfg = SortConfig(seed=args.seed, split_dprime=None if args.no_split else 4.0,
                         template_matching=not args.no_matching, drift_correction=args.drift_correction,
                         drift_method=args.drift_method, tracking=args.tracking)
        sorting = sort(rec.data, rec.fs, cfg, channel_positions=geom)
        if sorting.drift is not None or sorting.localization:
            from . import plotting

            args.out.mkdir(parents=True, exist_ok=True)
            if sorting.drift is not None:
                fig = plotting.plot_drift(sorting, true_displacement=lambda t: -rec.motion_at(t))
                fig.savefig(args.out / "drift_map.png", dpi=150)
            if sorting.localization:
                fig = plotting.plot_trajectories(sorting, true_displacement=lambda t: -rec.motion_at(t))
                fig.savefig(args.out / "trajectories.png", dpi=150)
        comp = compare_to_ground_truth(rec.spike_times, rec.spike_labels,
                                       sorting.times, sorting.labels, rec.fs)
        _save(sorting, args.out, comp)
        print()
        print(comp.summary())
    else:
        data = np.load(args.path)
        positions = np.load(args.positions) if args.positions is not None else None
        cfg = SortConfig(threshold=args.threshold, use_cmr=args.cmr,
                         template_matching=not args.no_matching,
                         drift_correction=positions is not None, drift_method=args.drift_method,
                         tracking=args.tracking)
        sorting = sort(data, args.fs, cfg, channel_positions=positions)
        if sorting.localization:
            from . import plotting

            args.out.mkdir(parents=True, exist_ok=True)
            plotting.plot_trajectories(sorting).savefig(args.out / "trajectories.png", dpi=150)
        _save(sorting, args.out)


if __name__ == "__main__":
    main()
