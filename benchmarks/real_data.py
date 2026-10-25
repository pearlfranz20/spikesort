"""Run spikesort on a SpikeInterface recording and score it against ground truth.

This is the bridge to data that did not come from our own simulator:
SpikeInterface reads most formats, and hybrid and synthetic datasets
(MEArec, SpikeForest, the SpikeInterface generators) carry a ground-truth
sorting. Three ways in:

    python benchmarks/real_data.py --generate --units 8 --channels 16 --duration 60
    python benchmarks/real_data.py --generate-drift --motion 20 --duration 60
    python benchmarks/real_data.py --mearec path/to/recording.h5
    python benchmarks/real_data.py --recording path/to/rec --sorting path/to/gt_sorting

The first two use SpikeInterface's own generators (ellipsoid templates,
spatially correlated noise, 5% amplitude jitter, zigzag drift: all
different from ours, so they are an independent check). `--generate-drift`
makes a 16-channel linear probe recording that drifts and sorts it with
and without drift correction. The third reads a MEArec file, which
bundles recording and ground truth. The last loads anything saved with
SpikeInterface (`recording.save(...)`, or a folder `si.load` understands).

Needs `pip install spikeinterface` (and `MEArec` for --mearec). Data
downloads do not work from every environment, so this is meant to be run
locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spikesort import SortConfig, compare_to_ground_truth, sort


def generate_drifting(args):
    """A drifting 16-channel linear-probe recording from SpikeInterface's generator."""
    import spikeinterface.generation as sg

    rng = np.random.default_rng(args.seed)
    n_ch, pitch = 16, 20.0
    span = (n_ch - 1) * pitch
    locs = np.column_stack([rng.uniform(-15, 15, args.units),
                            rng.uniform(30, span - 30, args.units),
                            rng.uniform(5, 30, args.units)])
    probe = dict(num_columns=1, num_contact_per_column=n_ch, xpitch=pitch, ypitch=pitch,
                 contact_shapes="circle", contact_shape_params={"radius": 5})
    motion = dict(displacement_sampling_frequency=5.0, drift_start_um=[0, args.motion],
                  drift_stop_um=[0, -args.motion], drift_step_um=1,
                  motion_list=[dict(drift_mode="zigzag", non_rigid_gradient=None, t_start_drift=0.0,
                                    t_end_drift=None, period_s=args.duration / 1.5)])
    static, drifting, gt = sg.generate_drifting_recording(
        num_units=args.units, duration=args.duration, sampling_frequency=30_000.0,
        generate_probe_kwargs=probe, unit_locations=locs, generate_displacement_vector_kwargs=motion,
        generate_sorting_kwargs=dict(firing_rates=(2.0, 10.0), refractory_period_ms=2.0),
        generate_noise_kwargs=dict(noise_levels=(8.0, 10.0), spatial_decay=25.0), seed=args.seed)
    return static, drifting, gt


def load(args):
    import spikeinterface as si

    if args.generate:
        rec, gt = si.generate_ground_truth_recording(
            durations=[args.duration], num_channels=args.channels, num_units=args.units,
            sampling_frequency=30_000.0, seed=args.seed)
        return rec, gt
    if args.mearec:
        import spikeinterface.extractors as se

        rec, gt = se.read_mearec(args.mearec)
        return rec, gt
    rec = si.load(args.recording)
    gt = si.load(args.sorting) if args.sorting else None
    return rec, gt


def traces_in_uv(rec, start_s: float, duration_s: float | None, max_channels: int | None):
    fs = rec.get_sampling_frequency()
    start = int(round(start_s * fs))
    end = None if duration_s is None else start + int(round(duration_s * fs))
    channel_ids = rec.channel_ids[:max_channels] if max_channels else rec.channel_ids
    scaled = rec.has_scaleable_traces() if hasattr(rec, "has_scaleable_traces") else False
    kwargs = dict(start_frame=start, end_frame=end, channel_ids=channel_ids)
    if scaled:
        # The keyword was renamed in SpikeInterface 0.102; support both.
        try:
            x = rec.get_traces(return_in_uV=True, **kwargs)
        except TypeError:
            x = rec.get_traces(return_scaled=True, **kwargs)
    else:
        x = rec.get_traces(**kwargs)
    x = np.asarray(x, dtype=np.float32)
    positions = rec.get_channel_locations(channel_ids=channel_ids)
    return x, fs, start, positions


def numpy_sorting(si, samples, labels, fs):
    """A SpikeInterface sorting from spike sample indices and labels, across API versions.

    Newer versions take samples through `from_samples_and_labels` and
    seconds through `from_times_and_labels`; older ones take samples
    through `from_times_labels`.
    """
    if hasattr(si.NumpySorting, "from_samples_and_labels"):
        return si.NumpySorting.from_samples_and_labels(samples, labels, fs)
    return si.NumpySorting.from_times_labels(samples, labels, fs)


def ground_truth_arrays(gt, start: int, n_samples: int):
    """Spike times (relative to `start`) and labels of every ground-truth unit."""
    times, labels = [], []
    for i, uid in enumerate(gt.unit_ids):
        t = gt.get_unit_spike_train(uid) - start
        t = t[(t >= 0) & (t < n_samples)]
        times.append(t)
        labels.append(np.full(t.size, i))
    times = np.concatenate(times)
    labels = np.concatenate(labels)
    order = np.argsort(times, kind="stable")
    return times[order], labels[order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate", action="store_true", help="SpikeInterface's synthetic ground truth")
    p.add_argument("--generate-drift", action="store_true",
                   help="SpikeInterface's drifting generator on a 16-channel linear probe")
    p.add_argument("--motion", type=float, default=20.0, help="peak drift in microns for --generate-drift")
    p.add_argument("--mearec", type=Path, default=None, help="MEArec .h5 recording with ground truth")
    p.add_argument("--recording", type=Path, default=None, help="folder loadable with si.load")
    p.add_argument("--sorting", type=Path, default=None, help="ground-truth sorting loadable with si.load")
    p.add_argument("--units", type=int, default=8)
    p.add_argument("--channels", type=int, default=16, help="channels to generate")
    p.add_argument("--duration", type=float, default=60.0, help="seconds to generate or to read")
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
    p.add_argument("--max-channels", type=int, default=None, help="use only the first N channels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drift-correction", action="store_true")
    p.add_argument("--drift-method", choices=["interpolate", "positions"], default="interpolate")
    p.add_argument("--tracking", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/real_data"))
    args = p.parse_args()
    if not (args.generate or args.generate_drift or args.mearec or args.recording):
        p.error("give --generate, --generate-drift, --mearec or --recording")

    import spikeinterface as si
    import spikeinterface.comparison as sc

    if args.generate_drift:
        static, drifting, gt = generate_drifting(args)
        results = {}
        runs = [("static", static, SortConfig()), ("drifting", drifting, SortConfig()),
                ("drifting, interpolation", drifting, SortConfig(drift_correction=True)),
                ("drifting, positions", drifting,
                 SortConfig(drift_correction=True, drift_method="positions"))]
        for name, rec, cfg in runs:
            x, fs, start, positions = traces_in_uv(rec, 0.0, None, None)
            s = sort(x, fs, cfg, channel_positions=positions)
            gt_times, gt_labels = ground_truth_arrays(gt, start, x.shape[0])
            comp = compare_to_ground_truth(gt_times, gt_labels, s.times, s.labels, fs)
            results[name] = {"accuracy": float(comp.accuracy.mean()),
                             "precision": float(comp.precision.mean()),
                             "recall": float(comp.recall.mean()), "n_units_found": int(s.n_units)}
            line = (f"{name:<24} accuracy {comp.accuracy.mean():.3f} precision {comp.precision.mean():.3f} "
                    f"units {s.n_units}/{len(gt.unit_ids)}")
            if s.drift is not None:
                est = s.drift.displacement_um
                line += f"  estimated drift range {est.min():+.1f}..{est.max():+.1f} um"
            print(line, flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "drift_scores.json").write_text(json.dumps(results, indent=2))
        return

    rec, gt = load(args)
    duration = args.duration if (args.mearec or args.recording) else None
    x, fs, start, positions = traces_in_uv(rec, args.start, duration, args.max_channels)
    print(f"{x.shape[1]} channels, {x.shape[0] / fs:.1f} s at {fs:.0f} Hz, "
          f"{'with' if gt is not None else 'without'} ground truth")

    cfg = SortConfig(drift_correction=args.drift_correction, drift_method=args.drift_method,
                     tracking=args.tracking)
    s = sort(x, fs, cfg, channel_positions=positions)
    print(f"stage 1: {s.stage1['times'].size} events in {s.stage1['n_clusters']} clusters; "
          f"stage 2: {s.times.size} spikes in {s.n_units} units")

    args.out.mkdir(parents=True, exist_ok=True)
    ours = numpy_sorting(si, s.times + start, s.labels, fs)
    ours.save(folder=args.out / "sorting", overwrite=True)

    if gt is None:
        print("no ground truth: sorting saved, nothing to score")
        return
    gt_times, gt_labels = ground_truth_arrays(gt, start, x.shape[0])
    ours_comp = compare_to_ground_truth(gt_times, gt_labels, s.times, s.labels, fs)
    print("\nspikesort's own scoring:")
    print(ours_comp.summary())

    gt_window = gt.frame_slice(start, start + x.shape[0]) if hasattr(gt, "frame_slice") else gt
    comp = sc.compare_sorter_to_ground_truth(gt_window, numpy_sorting(si, s.times, s.labels, fs),
                                             delta_time=0.4, exhaustive_gt=True)
    perf = comp.get_performance(method="by_unit")
    print("\nSpikeInterface's scoring (delta 0.4 ms):")
    print(perf[["accuracy", "recall", "precision"]].round(3).to_string())
    pooled = comp.get_performance(method="pooled_with_average")
    print(f"mean accuracy {float(pooled['accuracy']):.3f}, recall {float(pooled['recall']):.3f}, "
          f"precision {float(pooled['precision']):.3f}")
    (args.out / "scores.json").write_text(json.dumps({
        "ours": {"accuracy": float(ours_comp.accuracy.mean()),
                 "precision": float(ours_comp.precision.mean()),
                 "recall": float(ours_comp.recall.mean())},
        "spikeinterface": {k: float(pooled[k]) for k in ["accuracy", "recall", "precision"]},
        "n_units_found": int(s.n_units), "n_units_true": int(len(gt.unit_ids)),
    }, indent=2))


if __name__ == "__main__":
    main()
