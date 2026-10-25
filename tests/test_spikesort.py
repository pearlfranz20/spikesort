import numpy as np
import pytest

from spikesort import SortConfig, compare_to_ground_truth, linear_probe_geometry, simulate_recording, sort
from spikesort.cluster import merge_similar, relabel_by_size, separation, split_bimodal
from spikesort.detect import _enforce_dead_time, detect_spikes, extract_waveforms
from spikesort.evaluate import _count_matches, isi_violation_rate, isolation_distance
from spikesort.features import PCA
from spikesort.matching import correlate_templates, match_templates, select_templates, shifted_template
from spikesort.preprocess import bandpass, mad_noise
from spikesort.simulate import spike_waveform

FS = 30_000.0


@pytest.fixture(scope="module")
def rec():
    return simulate_recording(n_units=4, duration=20, seed=3)


# Simulator

def test_simulation_is_reproducible():
    a = simulate_recording(n_units=3, duration=2, seed=7)
    b = simulate_recording(n_units=3, duration=2, seed=7)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_array_equal(a.spike_times, b.spike_times)


def test_simulated_trains_respect_refractory_period(rec):
    for u in range(rec.n_units):
        assert isi_violation_rate(rec.spike_times[rec.spike_labels == u], rec.fs, 2.0) == 0.0


def test_spike_times_point_at_troughs(rec):
    x = bandpass(rec.data, rec.fs)
    big = np.argmax(-rec.templates.min(axis=(1, 2)))
    t = rec.spike_times[rec.spike_labels == big][:50]
    ch = int(np.argmin(rec.templates[big].min(axis=0)))
    offsets = [int(np.argmin(x[s - 5:s + 6, ch])) - 5 for s in t]
    assert abs(np.median(offsets)) <= 1


# Preprocessing and detection

def test_bandpass_removes_drift():
    t = np.arange(int(FS)) / FS
    slow = 100 * np.sin(2 * np.pi * 2 * t)[:, None]
    assert np.abs(bandpass(slow, FS)).max() < 1.0


def test_mad_is_robust_to_spikes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100_000, 1))
    x[::500] -= 50
    assert mad_noise(x)[0] == pytest.approx(1.0, rel=0.05)
    assert x.std() > 2.0  # the plain SD is badly inflated


def test_detects_isolated_spikes_at_the_trough():
    x = np.zeros((3000, 2), dtype=np.float32)
    x += np.random.default_rng(1).standard_normal(x.shape).astype(np.float32)
    for t in (500, 1500, 2500):
        x[t - 2:t + 3, 1] -= np.array([10, 20, 30, 20, 10])
    found = detect_spikes(x, FS, threshold=5, method="threshold")
    np.testing.assert_array_equal(found, [500, 1500, 2500])
    found = detect_spikes(x, FS, threshold=5, method="energy")
    assert found.size == 3
    assert np.abs(found - [500, 1500, 2500]).max() <= 1


def test_dead_time_keeps_deepest_event():
    zmin = np.zeros(100)
    zmin[[10, 15, 60]] = [-6, -9, -7]
    kept = _enforce_dead_time(np.array([10, 15, 60]), zmin, dead=10)
    np.testing.assert_array_equal(kept, [15, 60])


def test_extract_waveforms_drops_edge_spikes():
    x = np.arange(100, dtype=float)[:, None]
    wf, t = extract_waveforms(x, np.array([2, 50, 98]), FS, pre_ms=0.1, post_ms=0.1)
    np.testing.assert_array_equal(t, [50])
    assert wf.shape == (1, 6, 1)
    assert wf[0, 3, 0] == 50


# Features and clustering

def test_pca_matches_numpy_variance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 10)) @ np.diag(np.arange(10, 0, -1))
    pca = PCA(3).fit(X)
    Z = pca.transform(X)
    top3 = np.sort(np.linalg.eigvalsh(np.cov(X.T, ddof=0)))[::-1][:3]
    np.testing.assert_allclose(Z.var(axis=0, ddof=0), top3, rtol=1e-6)
    assert np.all(np.diff(pca.explained_variance_ratio_) <= 0)


def test_merge_joins_identical_templates_only():
    t = np.linspace(0, 1, 30)
    a = -np.exp(-((t - 0.3) / 0.05) ** 2)
    b = -np.exp(-((t - 0.6) / 0.05) ** 2)
    rng = np.random.default_rng(0)
    clean = np.stack([a] * 20 + [a * 1.02] * 20 + [b] * 20)
    wf = (clean + 0.05 * rng.standard_normal(clean.shape))[:, :, None]
    labels = np.repeat([0, 1, 2], 20)
    merged = merge_similar(wf, labels)
    assert len(np.unique(merged)) == 2
    assert merged[0] == merged[25] != merged[50]


def _noisy_cluster(template, n, sd, rng):
    return template[None, :] + sd * rng.standard_normal((n, template.size))


def test_split_leaves_a_single_neuron_alone():
    # The held-out d' is what prevents this: an in-sample score on a 2-way
    # GMM split of one Gaussian cloud in 60 dims looks well separated.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 60)
    tmpl = -np.exp(-((t - 0.3) / 0.05) ** 2) * 100
    wf = _noisy_cluster(tmpl, 400, 5.0, rng)[:, :, None]
    feats = PCA(6).fit_transform(wf.reshape(400, -1))
    labels = split_bimodal(wf, np.zeros(400, dtype=int), feats)
    assert len(np.unique(labels)) == 1


def test_split_separates_two_hidden_neurons():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 60)
    a = -np.exp(-((t - 0.3) / 0.05) ** 2) * 100
    b = -np.exp(-((t - 0.3) / 0.08) ** 2) * 100  # same peak, wider
    wf = np.vstack([_noisy_cluster(a, 200, 3.0, rng), _noisy_cluster(b, 200, 3.0, rng)])[:, :, None]
    truth = np.repeat([0, 1], 200)
    feats = PCA(6).fit_transform(wf.reshape(400, -1))
    labels = split_bimodal(wf, np.zeros(400, dtype=int), feats)
    assert len(np.unique(labels)) == 2
    agree = max((labels == truth).mean(), (labels != truth).mean())
    assert agree > 0.95


def test_separation_small_within_unit_large_between():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 60)
    a = -np.exp(-((t - 0.3) / 0.05) ** 2) * 100
    b = -np.exp(-((t - 0.3) / 0.08) ** 2) * 100
    wf = np.vstack([_noisy_cluster(a, 200, 3.0, rng), _noisy_cluster(b, 200, 3.0, rng)])[:, :, None]
    assert separation(wf, np.repeat([0, 1], 200), 0, 1) > 5
    within = rng.integers(0, 2, 400)  # neuron a split at random into 0 and 1
    within[200:] = 2
    assert separation(wf, within, 0, 1) < 0.5


def test_relabel_by_size():
    np.testing.assert_array_equal(relabel_by_size(np.array([5, 5, 2, 2, 2, 9])), [1, 1, 0, 0, 0, 2])


# Evaluation

def test_count_matches_uses_each_sorted_spike_once():
    assert _count_matches(np.array([100, 101]), np.array([100]), delta=5) == 1
    assert _count_matches(np.array([100, 200]), np.array([102, 300]), delta=5) == 1


def test_perfect_sorting_scores_one():
    t = np.array([10, 50, 90, 130])
    lab = np.array([0, 1, 0, 1])
    c = compare_to_ground_truth(t, lab, t, 1 - lab, FS)  # labels swapped on purpose
    assert c.accuracy.tolist() == [1.0, 1.0]
    assert c.assignment == {0: 1, 1: 0}


def test_isolation_distance_grows_with_separation():
    rng = np.random.default_rng(0)
    near = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(2, 1, (200, 2))])
    far = np.vstack([rng.normal(0, 1, (200, 2)), rng.normal(10, 1, (200, 2))])
    lab = np.repeat([0, 1], 200)
    assert isolation_distance(far, lab, 0) > 10 * isolation_distance(near, lab, 0)


# End to end

def test_end_to_end_accuracy(rec):
    s = sort(rec.data, rec.fs)
    c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
    assert c.accuracy.mean() > 0.85
    assert c.precision.min() > 0.85


def test_linear_probe_runs():
    r = simulate_recording(n_units=3, duration=10, channel_positions=linear_probe_geometry(8), seed=1)
    s = sort(r.data, r.fs)
    c = compare_to_ground_truth(r.spike_times, r.spike_labels, s.times, s.labels, r.fs)
    assert c.accuracy.mean() > 0.8


# Template matching (stage two)

def _two_templates():
    n_t = 54
    a = spike_waveform(n_t, FS)
    b = spike_waveform(n_t, FS, width_ms=0.2, repol_ratio=0.5)
    return (100 * np.stack([a, 0.5 * a], axis=1), 70 * np.stack([0.4 * b, b], axis=1))


def test_correlate_templates_matches_direct_sum():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3))
    temps = rng.standard_normal((2, 20, 3))
    corr = correlate_templates(x, temps)
    direct = np.array([[(x[t:t + 20] * temps[k]).sum() for t in range(281)] for k in range(2)])
    np.testing.assert_allclose(corr, direct, atol=1e-9)


def test_shifted_template_shifts_the_trough():
    t_a, _ = _two_templates()
    # A full-sample shift must equal an index shift, away from the edges.
    np.testing.assert_allclose(shifted_template(t_a, 1.0)[2:-2], t_a[1:-3], atol=1e-6)
    # Half a sample later, the sample after the trough reads a value between
    # the trough and its old neighbour.
    half = shifted_template(t_a, 0.5)
    trough = int(np.argmin(t_a[:, 0]))
    assert t_a[trough, 0] < half[trough + 1, 0] < t_a[trough + 1, 0]


def test_matching_recovers_overlapping_spikes():
    rng = np.random.default_rng(0)
    t_a, t_b = _two_templates()
    n_t = t_a.shape[0]
    x = 5 * rng.standard_normal((20_000, 2))
    truth = [(1000, 0, 0.0), (1012, 1, 0.0), (3000, 0, 0.0), (3006, 1, 0.0), (6000, 1, 0.0),
             (8000, 0, 0.3), (12000, 1, -0.4), (12003, 0, 0.2), (15000, 0, 0.0), (15000, 1, 0.0)]
    for start, k, shift in truth:
        x[start:start + n_t] += shifted_template([t_a, t_b][k], shift)
    m = match_templates(x, [t_a, t_b])
    assert m.starts.size == len(truth)
    got = sorted(zip(m.starts + m.shifts, m.labels, strict=True))
    want = sorted((s + sh, k) for s, k, sh in truth)
    for (gs, gk), (ws, wk) in zip(got, want, strict=True):
        assert gk == wk
        assert abs(gs - ws) < 0.5
    assert np.all(np.abs(m.amplitudes - 1) < 0.1)
    assert m.residual.std() < 5.3  # noise was 5


def test_matching_fits_nothing_on_noise():
    rng = np.random.default_rng(1)
    t_a, t_b = _two_templates()
    x = 5 * rng.standard_normal((60_000, 2))
    m = match_templates(x, [t_a, t_b])
    assert m.starts.size == 0
    np.testing.assert_allclose(m.residual, x, rtol=1e-6)  # float32 copy, untouched


def test_select_templates_drops_composite_cluster():
    rng = np.random.default_rng(0)
    t_a, t_b = _two_templates()
    noise = np.array([4.0, 4.0])
    clean = t_a[None] + 4 * rng.standard_normal((100, *t_a.shape))
    shifts = rng.integers(-12, 13, size=100)
    composite = np.stack([t_a + np.roll(t_b, s, axis=0) for s in shifts])
    composite = composite + 4 * rng.standard_normal(clean.shape)
    wf = np.vstack([clean, composite])
    labels = np.repeat([0, 1], 100)
    kept, dropped = select_templates(wf, labels, noise)
    assert list(kept) == [0]
    assert dropped == {1: 100}


def test_matching_improves_collision_recall():
    from spikesort.evaluate import collision_mask, hit_mask

    rec = simulate_recording(n_units=5, duration=30, seed=0)
    coll = collision_mask(rec.spike_times, rec.spike_labels, rec.fs)
    assert coll.sum() > 20  # otherwise the test is meaningless
    results = {}
    for name, cfg in [("cluster", SortConfig(template_matching=False)), ("match", SortConfig())]:
        s = sort(rec.data, rec.fs, cfg)
        c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
        hit = hit_mask(rec.spike_times, rec.spike_labels, s.times, s.labels, c, rec.fs)
        results[name] = (c.accuracy.mean(), c.precision.min(), hit[~coll].mean(), hit[coll].mean())
    acc, prec, iso, col = results["match"]
    assert acc > results["cluster"][0]
    assert acc > 0.95
    assert prec > 0.9
    assert iso > 0.97
    assert col > results["cluster"][3] + 0.3


# Drift

def test_static_recording_is_unchanged_by_motion_support():
    import hashlib

    rec = simulate_recording(n_units=3, duration=2, seed=7)
    assert hashlib.md5(rec.data.tobytes()).hexdigest()[:12] == "9e9c9934af29"
    assert rec.motion_um is None
    np.testing.assert_allclose(rec.template_at(0, 1.0), rec.templates[0])


def test_motion_changes_amplitude_profile_over_time():
    geom = linear_probe_geometry(16)
    rec = simulate_recording(n_units=3, duration=10, seed=1, channel_positions=geom,
                             motion_um=20, unit_margin_um=-30)
    early, late = rec.template_at(0, 0.0), rec.template_at(0, 10.0 / 6)  # peak of the sine
    assert np.abs(rec.motion_at(10.0 / 6) - 20) < 0.5
    assert not np.allclose(early, late)
    # The unit's footprint slides along the shank: the deepest channel moves.
    assert early.min(axis=0).argmin() != late.min(axis=0).argmin()


def test_register_sign_and_subbin_precision():
    from spikesort.drift import _register

    g = np.exp(-0.5 * ((np.arange(200) - 80) / 5.0) ** 2)
    lag, corr = _register(g, np.roll(g, 7), 30)
    assert abs(lag - 7) < 0.05
    assert corr > 0.99


def test_shift_channels_undoes_a_shifted_field():
    from spikesort.drift import shift_channels

    pos = linear_probe_geometry(8, 20.0)
    y = pos[:, 1]
    bump = lambda y0: np.exp(-0.5 * ((y - y0) / 25.0) ** 2)  # noqa: E731
    drifted = bump(78.0)[None, :]
    back = shift_channels(drifted, pos, np.array([8.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(back[0], bump(70.0), atol=1e-3)


def test_estimate_drift_recovers_a_sine():
    from spikesort.drift import estimate_drift

    rng = np.random.default_rng(0)
    duration, n = 60.0, 3000
    t = np.sort(rng.uniform(0, duration, n))
    unit = rng.integers(0, 4, n)
    true = 15 * np.sin(2 * np.pi * 1.5 * t / duration)
    depth = np.array([30.0, 60.0, 95.0, 120.0])[unit] + true + rng.normal(0, 2.0, n)
    amp = np.array([80.0, 120.0, 60.0, 150.0])[unit] * rng.uniform(0.9, 1.1, n)
    est = estimate_drift(t, depth, amp, duration, bin_s=2.0)
    want = 15 * np.sin(2 * np.pi * 1.5 * est.bin_centres_s / duration)
    want -= want.mean()
    assert np.sqrt(np.mean((est.displacement_um - want) ** 2)) < 1.0


def test_drift_correction_restores_accuracy():
    geom = linear_probe_geometry(16)
    rec = simulate_recording(n_units=5, duration=16, seed=0, channel_positions=geom,
                             motion_um=20, unit_margin_um=-30)
    results = {}
    for corrected in [False, True]:
        s = sort(rec.data, rec.fs, SortConfig(drift_correction=corrected), channel_positions=geom)
        c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
        results[corrected] = c.accuracy.mean()
        if corrected:
            true = -rec.motion_at(s.drift.bin_centres_s)
            true = true - true.mean()
            assert np.sqrt(np.mean((s.drift.displacement_um - true) ** 2)) < 4.0
    assert results[True] > 0.9
    assert results[True] > results[False] + 0.15


# Tracking

def test_gap_ratio_separates_continuum_from_two_clusters():
    from spikesort.cluster import gap_ratio

    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 40))
    b = rng.standard_normal((300, 40))
    b[:, 0] += 4.0
    two = np.vstack([a, b])[:, :, None]
    assert gap_ratio(two, np.repeat([0, 1], 300), 0, 1) < 0.7
    u = rng.standard_normal((600, 40))
    u[:, 0] = rng.uniform(0, 8, 600)  # one neuron sliding along a continuum
    assert gap_ratio(u[:, :, None], (u[:, 0] > 4).astype(int), 0, 1) > 0.75  # noisy; see docstring


def test_refractory_overlap_tells_one_neuron_from_two():
    from spikesort.tracking import refractory_overlap

    rec = simulate_recording(n_units=2, duration=60, rates=(10.0, 12.0), seed=4)
    t0 = rec.spike_times[rec.spike_labels == 0]
    t1 = rec.spike_times[rec.spike_labels == 1]
    close, expected = refractory_overlap(t0[::2], t0[1::2], rec.fs)  # one neuron, split in two
    assert expected > 5 and close == 0
    close, expected = refractory_overlap(t0, t1, rec.fs)  # two independent neurons
    assert expected > 5 and close > 0.4 * expected


def test_tracking_matches_plain_matching_on_static_data():
    rec = simulate_recording(n_units=4, duration=20, seed=3)
    plain = sort(rec.data, rec.fs)
    tracked = sort(rec.data, rec.fs, SortConfig(tracking=True))
    c_plain = compare_to_ground_truth(rec.spike_times, rec.spike_labels, plain.times, plain.labels,
                                      rec.fs)
    c_track = compare_to_ground_truth(rec.spike_times, rec.spike_labels, tracked.times, tracked.labels,
                                      rec.fs)
    assert tracked.matching["mode"] == "tracking"
    assert tracked.tracking["chunk_edges"].size == 5  # 20 s in 5 s chunks
    assert c_track.accuracy.mean() > 0.95
    assert abs(c_track.accuracy.mean() - c_plain.accuracy.mean()) < 0.03


# Drift in position space (localize.py)

def test_geometry_knows_when_height_is_not_identifiable():
    from spikesort.localize import Geometry
    from spikesort.simulate import tetrode_geometry

    tet = Geometry.from_positions(tetrode_geometry())
    assert tet.degenerate and tet.dim == 2
    lin = Geometry.from_positions(linear_probe_geometry(16))
    assert not lin.degenerate and lin.dim == 1
    grid = Geometry.from_positions([[x, y] for x in (0, 20, 40) for y in (0, 20, 40)])
    assert not grid.degenerate and grid.dim == 2


def test_fit_position_recovers_a_source_on_a_probe_and_reproduces_a_tetrode_profile():
    from spikesort.localize import Geometry, fit_position
    from spikesort.simulate import tetrode_geometry

    lin = Geometry.from_positions(linear_probe_geometry(16))
    s_true, z_true, a_true = np.array([37.0]), 24.0, 300.0
    profile = a_true * lin.gains(s_true, z_true, 1.5)
    s, z, a, rms = fit_position(profile, lin, 1.5)
    assert abs(s[0] - s_true[0]) < 0.5 and abs(z - z_true) < 0.5 and abs(a / a_true - 1) < 0.02

    # On a tetrode a source at height 30 is fitted exactly at height 15 as
    # well: the profile does not say how high the source is.
    tet = Geometry.from_positions(tetrode_geometry())
    profile = 200.0 * tet.gains(np.array([10.0, -4.0]), 30.0, 1.5)
    _, _, _, rms = fit_position(profile, tet, 1.5, z=15.0)
    assert rms < 0.01


def test_select_templates_keeps_a_drifting_unit_when_judged_against_neighbours_in_time():
    rng = np.random.default_rng(1)
    t_a, t_b = _two_templates()
    noise = np.array([4.0, 4.0])
    scale = np.linspace(0.6, 1.4, 200)  # unit 0 grows over the recording, unit 1 does not
    drifting = scale[:, None, None] * t_a[None] + 4 * rng.standard_normal((200, *t_a.shape))
    static = t_b[None] + 4 * rng.standard_normal((100, *t_b.shape))
    wf = np.vstack([drifting, static])
    labels = np.repeat([0, 1], [200, 100])
    times = np.concatenate([np.arange(200) * 1000, np.arange(100) * 2000])
    kept, dropped = select_templates(wf, labels, noise)
    assert list(kept) == [1] and dropped == {0: 200}  # against its global median it looks composite
    kept, dropped = select_templates(wf, labels, noise, times=times)
    assert sorted(kept) == [0, 1] and dropped == {}


def test_link_trajectories_joins_bands_of_one_neuron_and_not_a_neighbour():
    from spikesort.localize import Geometry, cluster_trajectories, link_trajectories, trough_profiles

    rng = np.random.default_rng(0)
    geom = Geometry.from_positions(linear_probe_geometry(16))
    fs, n_bins, bin_s = 30_000.0, 30, 2.0
    edges = np.arange(n_bins + 1) * int(bin_s * fs)
    t_s = np.sort(rng.uniform(0, n_bins * bin_s, 1500))
    times = (t_s * fs).astype(np.int64)
    motion = 20 * np.sin(2 * np.pi * 1.5 * t_s / (n_bins * bin_s))
    # Neuron A drifts and was cut into two bands at position zero; neuron B
    # sits 60 um away and fires throughout.
    which = rng.integers(0, 2, t_s.size)
    pos = np.where(which == 0, motion, 60.0 + motion)
    labels = np.where(which == 1, 2, (pos > 0).astype(int))
    profiles = np.stack([150.0 * geom.gains(np.array([p]), 20.0, 1.5) for p in pos])
    profiles += rng.normal(0, 3.0, profiles.shape)
    traj = cluster_trajectories(times, labels, np.clip(profiles, 0, None), edges, geom)
    wave = np.zeros((60, 16))
    wave[20:30, :] = -np.hanning(10)[:, None]  # the same time course on every cluster
    templates = {k: wave * (profiles[labels == k].mean(axis=0))[None, :] for k in (0, 1, 2)}
    assert trough_profiles(templates[0][None]).shape == (1, 16)
    groups = link_trajectories(traj, templates, geom, n_bins)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_positions_method_sorts_a_drifting_tetrode():
    from spikesort.simulate import tetrode_geometry

    geom = tetrode_geometry()
    rec = simulate_recording(n_units=5, duration=60, seed=0, channel_positions=geom,
                             motion_um=20, unit_margin_um=15)
    s = sort(rec.data, rec.fs, SortConfig(drift_correction=True, drift_method="positions"),
             channel_positions=geom)
    c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
    assert s.matching["mode"] == "positions"
    assert s.drift is None and s.localization["degenerate"]  # no displacement in microns on a tetrode
    assert c.accuracy.mean() > 0.95 and s.n_units == 5
    true = -rec.motion_at(s.localization["bin_centres_s"])
    assert np.corrcoef(s.localization["drift_shape"], true)[0, 1] > 0.6  # but its time course


def test_positions_method_estimates_drift_in_microns_on_a_probe():
    geom = linear_probe_geometry(16)
    rec = simulate_recording(n_units=5, duration=16, seed=0, channel_positions=geom,
                             motion_um=20, unit_margin_um=-30)
    s = sort(rec.data, rec.fs, SortConfig(drift_correction=True, drift_method="positions"),
             channel_positions=geom)
    c = compare_to_ground_truth(rec.spike_times, rec.spike_labels, s.times, s.labels, rec.fs)
    assert c.accuracy.mean() > 0.85
    true = -rec.motion_at(s.drift.bin_centres_s)
    true = true - true.mean()
    assert np.sqrt(np.mean((s.drift.displacement_um - true) ** 2)) < 4.0
