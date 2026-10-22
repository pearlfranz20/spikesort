import numpy as np
import pytest

from spikesort import compare_to_ground_truth, linear_probe_geometry, simulate_recording, sort
from spikesort.cluster import merge_similar, relabel_by_size, separation, split_bimodal
from spikesort.detect import _enforce_dead_time, detect_spikes, extract_waveforms
from spikesort.evaluate import _count_matches, isi_violation_rate, isolation_distance
from spikesort.features import PCA
from spikesort.preprocess import bandpass, mad_noise

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
    found = detect_spikes(x, FS, threshold=5)
    np.testing.assert_array_equal(found, [500, 1500, 2500])


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
