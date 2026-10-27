"""Drift handling in position space, for any electrode geometry.

A spike's amplitudes across channels say where its neuron is. With a
point-source model of the extracellular field (amplitude falls off as a
power of the distance to each channel), a cluster's mean amplitude profile
in a time bin gives the cluster's position in that bin, and a sequence of
bins gives its trajectory. Three uses:

* Linking. A drifting neuron that a Gaussian mixture cut into amplitude
  bands is several clusters, each active in its own epoch. Their
  trajectories continue each other: extrapolate one across the gap and it
  predicts the other's amplitude profile. Two neurons do not continue each
  other, and, the stronger cue, a neuron is in one place at a time, so two
  bands of one neuron are never active in the same bins while two neurons
  that both fire are. Clusters are linked into units on continuity, with
  a veto on co-activity between the groups being joined.
* Rendering. Each unit's template is re-rendered for every time bin by
  scaling its channels with the model gains along the unit's own
  trajectory (the model sets the shape across channels, the measured
  amplitude in the bin sets the size), and stage two matches each bin
  with those templates.
* Drift estimate. A rigid displacement per bin that best explains all
  trajectories at once (weighted least squares), where the geometry
  allows it.

One geometric fact shapes all of this. On a tetrode every channel is the
same distance from the centre, and then the amplitude profile is invariant
to moving the source along the ray from the centre while adjusting its
height: the fit has an exact one-parameter degeneracy and cannot recover
the height, nor therefore the absolute scale of positions or of the
drift. Fitted positions on a tetrode move radially, by amounts that depend
on each neuron's height, so there is no frame they share: the rigid
registration is skipped, trajectories are extrapolated in time rather than
carried by a common drift, and only the time course of the drift is
reported (`drift_shape`), in arbitrary units. Linking and rendering do not
care: with a fixed nominal height the map from true position to fitted
position is a fixed smooth function, so two bands of one neuron still
continue each other, and a template rendered along its own fitted
trajectory still reproduces its own profiles. A linear probe has channels
at different distances from the centre, the degeneracy is absent, the
height is fitted, and the rigid registration is a displacement in microns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Geometry:
    """Channel positions and the subspace they span.

    A linear probe spans one dimension, a tetrode two. Positions are
    expressed as `centre + basis @ s` with `s` in the spanned subspace,
    plus a height `z` above the array plane.
    """

    positions: np.ndarray  # (n_ch, 2)
    centre: np.ndarray  # (2,)
    basis: np.ndarray  # (2, dim)
    extent: float  # largest distance from the centre to a channel
    degenerate: bool  # every channel equidistant from the centre: height not identifiable

    @classmethod
    def from_positions(cls, channel_positions: np.ndarray) -> "Geometry":
        pos = np.asarray(channel_positions, dtype=np.float64)
        centre = pos.mean(axis=0)
        _, sv, vt = np.linalg.svd(pos - centre, full_matrices=False)
        keep = sv > 1e-6 * max(sv.max(), 1e-12)
        basis = vt[keep].T  # (2, dim)
        radii = np.sqrt(((pos - centre) ** 2).sum(axis=1))
        degenerate = bool(radii.max() - radii.min() < 1e-6 * max(radii.max(), 1e-12))
        return cls(pos, centre, basis, float(radii.max()), degenerate)

    @property
    def dim(self) -> int:
        return self.basis.shape[1]

    def to_xy(self, s: np.ndarray) -> np.ndarray:
        return self.centre + np.asarray(s) @ self.basis.T

    def gains(self, s: np.ndarray, z: float, falloff: float) -> np.ndarray:
        """Point-source amplitude at every channel for a source at (s, z), unnormalised."""
        xy = self.to_xy(s)
        d2 = ((self.positions - xy[None, :]) ** 2).sum(axis=1) + z ** 2
        return d2 ** (-falloff / 2)


def fit_position(profile: np.ndarray, geom: Geometry, falloff: float = 1.5,
                 z: float | None = None, s0: np.ndarray | None = None):
    """Source position for one amplitude profile.

    Least squares on amplitude = A * gains(s, z), with the scale A solved in
    closed form inside the residual, so the search runs over `s` (and `z`
    unless fixed). The search starts under the loudest channel, and also
    from `s0` if given (the previous bin's position, say); the better of
    the two fits is kept. Returns (s, z, A, rms residual relative to the
    profile).
    """
    prof = np.asarray(profile, dtype=np.float64)
    scale = max(prof.max(), 1e-9)
    prof = prof / scale
    fit_z = z is None
    lo = np.full(geom.dim, -geom.extent - 60.0)
    hi = np.full(geom.dim, geom.extent + 60.0)
    if fit_z:
        lo, hi = np.append(lo, 3.0), np.append(hi, 80.0)

    def residual(x):
        g = geom.gains(x[:geom.dim], x[geom.dim] if fit_z else z, falloff)
        a = float(prof @ g / max(g @ g, 1e-12))
        return a * g - prof

    loudest = (geom.positions[int(np.argmax(prof))] - geom.centre) @ geom.basis
    starts = [loudest] if s0 is None else [loudest, np.asarray(s0, dtype=np.float64)]
    best = None
    for start in starts:
        x0 = np.clip(start, lo[:geom.dim] + 1e-3, hi[:geom.dim] - 1e-3)
        if fit_z:
            x0 = np.append(x0, 15.0)
        res = least_squares(residual, x0, bounds=(lo, hi), x_scale=10.0)
        if best is None or res.cost < best.cost:
            best = res
    s = best.x[:geom.dim]
    z_out = float(best.x[geom.dim]) if fit_z else float(z)
    g = geom.gains(s, z_out, falloff)
    a = float(prof @ g / max(g @ g, 1e-12)) * scale
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    return s, z_out, a, rms


def trough_profiles(waveforms: np.ndarray) -> np.ndarray:
    """Per-channel amplitude of each waveform at its trough (deepest sample on any channel, +-1)."""
    n, n_t, _ = waveforms.shape
    idx = np.argmin(waveforms.min(axis=2), axis=1)
    lo, hi = np.clip(idx - 1, 0, n_t - 1), np.clip(idx + 1, 0, n_t - 1)
    rows = np.arange(n)
    at = (waveforms[rows, lo] + waveforms[rows, idx] + waveforms[rows, hi]) / 3.0
    return np.clip(-at, 0, None).astype(np.float64)


@dataclass
class Trajectory:
    """Where one stage-one cluster sits in each time bin it fired in."""

    bins: np.ndarray  # bin indices with enough spikes
    positions: np.ndarray  # (n_bins, dim) fitted positions
    amplitudes: np.ndarray  # (n_bins,) fitted amplitude at the loudest channel
    counts: np.ndarray  # (n_bins,) spikes per bin
    z: float
    profiles: np.ndarray | None = None  # (n_bins, n_ch) mean measured profile per bin
    strengths: np.ndarray | None = None  # (n_bins,) fitted source strength A (amplitude = A * gains)


def cluster_trajectories(
    times: np.ndarray,
    labels: np.ndarray,
    profiles: np.ndarray,
    edges: np.ndarray,
    geom: Geometry,
    falloff: float = 1.5,
    fixed_z: float = 15.0,
    min_spikes: int = 6,
    max_pool_bins: int = 3,
) -> dict[int, Trajectory]:
    """Fit every cluster's position in every bin where it has `min_spikes` spikes.

    A cluster that fires too sparsely for that pools consecutive bins, up
    to `max_pool_bins` of them, until it has `min_spikes` (`_pool_bins`),
    so a unit at a few hertz still gets a point every few seconds instead
    of losing every bin where Poisson chance gave it one spike too few.

    The height is fitted once per cluster from its whole-recording mean
    profile (with 16 channels it is well determined), or held at
    `fixed_z` on a degenerate geometry such as a tetrode, where it is not
    identifiable at all (see the module docstring). Per bin only the
    in-plane position and the scale are fitted.
    """
    which = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, edges.size - 2)
    out: dict[int, Trajectory] = {}
    for k in np.unique(labels):
        m = labels == k
        if geom.degenerate:
            z = fixed_z
        else:
            _, z, _, _ = fit_position(profiles[m].mean(axis=0), geom, falloff)
        bins, pos, amps, counts, means, strengths = [], [], [], [], [], []
        s_prev = None
        for group in _pool_bins(which[m], min_spikes, max_pool_bins):
            mb = m & np.isin(which, group)
            n = int(mb.sum())
            mean = profiles[mb].mean(axis=0)
            s, _, a, _ = fit_position(mean, geom, falloff, z=z, s0=s_prev)
            s_prev = s
            # A pooled point sits at the bin holding its median spike.
            bins.append(int(np.median(which[mb])))
            pos.append(s)
            amps.append(a * geom.gains(s, z, falloff).max())
            counts.append(n)
            means.append(mean)
            strengths.append(a)
        out[int(k)] = Trajectory(np.array(bins, dtype=np.int64),
                                 np.array(pos).reshape(-1, geom.dim),
                                 np.array(amps), np.array(counts, dtype=np.int64), float(z),
                                 np.array(means).reshape(-1, profiles.shape[1]), np.array(strengths))
    return out


def _pool_bins(which: np.ndarray, min_spikes: int, max_pool_bins: int) -> list[list[int]]:
    """Consecutive bins grouped until each group has `min_spikes` spikes.

    `which` is the bin of every spike of one cluster. Groups span at most
    `max_pool_bins` bins. A run that cannot reach `min_spikes` within that
    span is abandoned and the search restarts at its later bins.
    """
    present, counts = np.unique(which, return_counts=True)
    groups, cur, n = [], [], 0
    for b, c in zip(present.tolist(), counts.tolist(), strict=True):
        cur.append(b)
        n += c
        while cur and cur[-1] - cur[0] + 1 > max_pool_bins:
            n -= int(counts[present == cur[0]][0])
            cur.pop(0)
        if n >= min_spikes:
            groups.append(cur)
            cur, n = [], 0
    return groups


def shape_correlation(ta: np.ndarray, tb: np.ndarray) -> float:
    """Correlation of the two templates' time courses on their own best channels.

    Drift changes which channel a neuron is loudest on and by how much,
    but not the time course of the spike itself, so this is the part of
    the template that identifies the neuron across epochs.
    """
    a = ta[:, int(np.argmin(ta.min(axis=0)))]
    b = tb[:, int(np.argmin(tb.min(axis=0)))]
    return float(np.corrcoef(a, b)[0, 1])


def _extrapolate(bins: np.ndarray, values: np.ndarray, b0: int, b1: int, window: int) -> np.ndarray:
    """Line through `values` over the bins within `window` of b0, read off at b1 (constant if too few)."""
    near = np.abs(bins - b0) <= window
    x = bins[near].astype(np.float64)
    if x.size >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, values[near], 1)
        return slope * b1 + intercept
    return values[bins == b0][0]


def profile_continuity(ta: Trajectory, tb: Trajectory, max_gap: int, geom: Geometry,
                       falloff: float = 1.5, displacement: np.ndarray | None = None,
                       window: int = 3) -> float:
    """How well each trajectory, extrapolated across the gap, predicts the other's amplitude profiles.

    Where the two come within `max_gap` bins of each other, one is
    extrapolated to the other's bin, the point-source model renders the
    profile it predicts there, and the relative difference from the
    profile actually measured there is taken (norm of the difference over
    norm of the mean). Both directions, every meeting, median. Two bands
    of one neuron continue each other and score low whatever the drift
    speed; two neurons do not. Infinite if they never meet.

    With `displacement` (the rigid drift per bin) the position is carried
    across the gap by the drift and the source strength held, which is
    exact for a rigid probe and lets a cluster seen in only a few bins
    follow the motion everyone else shows. Without it (a tetrode, where
    fitted positions move radially and by neuron-dependent amounts, so
    there is no common frame) position and log strength are extrapolated
    with a line in time over the `window` bins around the meeting.
    """
    out = []
    for first, second in ((ta, tb), (tb, ta)):
        if first.bins.size == 0 or second.bins.size == 0:
            continue
        log_strength = np.log(np.maximum(first.strengths, 1e-12))
        for i, b in enumerate(first.bins.tolist()):
            j = int(np.argmin(np.abs(second.bins - b)))
            bj = int(second.bins[j])
            if abs(bj - b) > max_gap:
                continue
            if displacement is not None:
                s_pred = first.positions[i] + displacement[bj] - displacement[b]
                log_a = log_strength[i]
            else:
                s_pred = _extrapolate(first.bins, first.positions, b, bj, window)
                log_a = _extrapolate(first.bins, log_strength, b, bj, window)
            pred = np.exp(log_a) * geom.gains(s_pred, first.z, falloff)
            target = second.profiles[j]
            out.append(np.linalg.norm(pred - target) / max(np.linalg.norm(0.5 * (pred + target)), 1e-9))
    return float(np.median(out)) if out else float("inf")


def coactivity(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """Share of the sparser unit's spikes fired in bins where the other was firing too.

    `counts_*` are spikes per bin. For every bin the smaller of the two
    counts is taken, and the total is divided by the smaller of the two
    totals. One neuron is in one place at a time, so two bands of the same
    neuron overlap only in the bins where it crosses between them, and the
    share is small. Two neurons that both fire throughout score near one.
    """
    both = float(np.minimum(counts_a, counts_b).sum())
    return both / max(min(float(counts_a.sum()), float(counts_b.sum())), 1.0)


def link_trajectories(
    traj: dict[int, Trajectory],
    templates: dict[int, np.ndarray],
    geom: Geometry,
    n_bins: int,
    max_profile_dist: float = 0.4,
    min_shape_corr: float = 0.97,
    max_gap_bins: int = 2,
    max_coactivity: float = 0.3,
    falloff: float = 1.5,
    displacement: np.ndarray | None = None,
) -> list[list[int]]:
    """Group clusters whose trajectories meet into units.

    Two clusters can be the same neuron when their trajectories come
    within `max_gap_bins` of each other, each one's amplitude profiles
    extrapolated across the gap predict the other's to within
    `max_profile_dist` (relative, see `profile_continuity`), their
    waveform time courses correlate above `min_shape_corr`, and they do
    not fire in the same bins more than `max_coactivity` of the time.
    `displacement` is passed on to `profile_continuity` and should be
    given only on a geometry where the fitted positions share a frame.

    Candidate pairs are linked best first (closest in profile, shape and
    co-activity together), and a link is refused when the two groups it
    would join fire together more than `max_coactivity` of the time. The
    best links are the true ones, and once a neuron's bands are together
    they cover the recording, so a neighbour that happens to meet one band
    is kept out by the whole. Returns groups, largest first.
    """
    ids = sorted(traj)
    parent = {k: k for k in ids}
    active = {}
    for k in ids:
        counts = np.zeros(n_bins)
        counts[traj[k].bins] = traj[k].counts
        active[k] = counts

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    candidates = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            dist = profile_continuity(traj[a], traj[b], max_gap_bins, geom, falloff, displacement)
            if dist > max_profile_dist:
                continue
            corr = shape_correlation(templates[a], templates[b])
            if corr < min_shape_corr:
                continue
            co = coactivity(active[a], active[b])
            if co > max_coactivity:
                continue
            score = dist / max_profile_dist + (1.0 - corr) / (1.0 - min_shape_corr) + co / max_coactivity
            candidates.append((score, a, b))
    for _, a, b in sorted(candidates):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if coactivity(active[ra], active[rb]) > max_coactivity:
            continue
        parent[ra] = rb
        active[rb] = active[ra] + active[rb]
    groups: dict[int, list[int]] = {}
    for k in ids:
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: -sum(int(traj[k].counts.sum()) for k in g))


@dataclass
class Unit:
    """A linked group of clusters with one trajectory and one reference template."""

    members: list[int]
    positions: np.ndarray  # (n_bins_total, dim), every bin, filled in where no member fired
    amplitudes: np.ndarray  # (n_bins_total,) amplitude at the loudest channel, likewise
    reference: np.ndarray  # template (n_t, n_ch) of the largest member
    reference_position: np.ndarray  # where that template was measured
    reference_amplitude: float  # its amplitude at the loudest channel
    z: float
    observed: np.ndarray  # (n_bins_total,) True where some member had data
    counts: np.ndarray  # (n_bins_total,) spikes the members had in each bin


def build_unit(members: list[int], traj: dict[int, Trajectory], templates: dict[int, np.ndarray],
               n_bins: int, geom: Geometry, rigid: "RigidDrift | None" = None,
               falloff: float = 1.5) -> Unit:
    """Merge the members' trajectories into one, fill the gaps, pick the reference.

    Bins where no member fired take their position from the neighbours.
    With a rigid drift estimate the unit is assumed to move with the array
    there: the position is interpolated relative to the rigid displacement
    and the displacement added back, so a unit seen only in the middle of
    the recording still follows the drift at both ends. The source
    strength is interpolated in the log domain and held at the ends, and
    the amplitude in every bin is the strength times the model gain of
    the loudest channel at that bin's position.
    """
    dim = geom.dim
    acc = np.zeros((n_bins, dim))
    log_a = np.zeros(n_bins)
    wsum = np.zeros(n_bins)
    for k in members:
        t = traj[k]
        for i, b in enumerate(t.bins):
            acc[b] += t.counts[i] * t.positions[i]
            log_a[b] += t.counts[i] * np.log(max(t.strengths[i], 1e-12))
            wsum[b] += t.counts[i]
    observed = wsum > 0
    largest = max(members, key=lambda k: int(traj[k].counts.sum()))
    t = traj[largest]
    positions = np.zeros((n_bins, dim))
    amplitudes = np.ones(n_bins)
    if observed.any():
        idx = np.flatnonzero(observed)
        d = rigid.displacement if rigid is not None else np.zeros((n_bins, dim))
        for j in range(dim):
            rel = acc[idx, j] / wsum[idx] - d[idx, j]
            positions[:, j] = np.interp(np.arange(n_bins), idx, rel) + d[:, j]
        strength = np.exp(np.interp(np.arange(n_bins), idx, log_a[idx] / wsum[idx]))
        amplitudes = np.array([strength[b] * geom.gains(positions[b], t.z, falloff).max()
                               for b in range(n_bins)])
    ref_pos = (t.positions * t.counts[:, None]).sum(axis=0) / max(t.counts.sum(), 1)
    ref_amp = float(trough_profiles(templates[largest][None])[0].max())
    return Unit(list(members), positions, amplitudes, templates[largest], ref_pos, ref_amp, t.z, observed,
                wsum.astype(np.int64))


def render_template(template: np.ndarray, s_ref: np.ndarray, s_now: np.ndarray, z: float,
                    geom: Geometry, falloff: float = 1.5, amp_ref: float | None = None,
                    amp_now: float | None = None) -> np.ndarray:
    """The template as it would look with the source moved from `s_ref` to `s_now`.

    Every channel is scaled by the ratio of the model gains at the two
    positions. The measured time courses are kept as they are, so the
    model only has to get the change in gains right, not the gains.

    With `amp_ref` and `amp_now` the model sets only the shape of the
    profile across channels and the overall size comes from the measured
    amplitudes: the rendered template's loudest channel is scaled by
    `amp_now / amp_ref`. On a degenerate geometry the model's gain ratios
    are right in shape but not in size (module docstring), so this is the
    form to use there.
    """
    g_now = geom.gains(s_now, z, falloff)
    g_ref = geom.gains(s_ref, z, falloff)
    ratio = g_now / g_ref
    if amp_ref is not None and amp_now is not None:
        ratio *= (amp_now / max(amp_ref, 1e-9)) * (g_ref.max() / g_now.max())
    return template * ratio[None, :]


@dataclass
class RigidDrift:
    displacement: np.ndarray  # (n_bins, dim), mean zero
    cluster_positions: np.ndarray  # (K, dim)
    residual_um: float = 0.0
    ids: list[int] = field(default_factory=list)


def rigid_registration(traj: dict[int, Trajectory], n_bins: int, dim: int = 1,
                       iterations: int = 3) -> RigidDrift:
    """One displacement per bin that best explains every cluster's trajectory.

    Solves y_kb = p_k + d_b by weighted least squares, one dimension at a
    time, displacements anchored to mean zero, with Huber re-weighting so
    a junk cluster whose "position" wanders does not bend the trajectory.
    Meaningful only where the fitted positions share a frame, which a
    degenerate geometry does not give (module docstring, `drift_shape`).
    """
    ids = [k for k in sorted(traj) if traj[k].bins.size]
    if not ids:
        return RigidDrift(np.zeros((n_bins, dim)), np.zeros((0, dim)))
    dim = traj[ids[0]].positions.shape[1]
    rows_k, rows_b, y, w0 = [], [], [], []
    for i, k in enumerate(ids):
        t = traj[k]
        for j, b in enumerate(t.bins):
            rows_k.append(i)
            rows_b.append(int(b))
            y.append(t.positions[j])
            w0.append(np.sqrt(t.counts[j]))
    rows_k, rows_b = np.array(rows_k), np.array(rows_b)
    y, w0 = np.array(y), np.array(w0)
    n_k, n_obs = len(ids), y.shape[0]
    w = w0.copy()
    p = np.zeros((n_k, dim))
    d = np.zeros((n_bins, dim))
    for _ in range(iterations):
        a_mat = np.zeros((n_obs + n_bins, n_k + n_bins))
        a_mat[np.arange(n_obs), rows_k] = w
        a_mat[np.arange(n_obs), n_k + rows_b] = w
        a_mat[n_obs:, n_k:] = 1e-3 * np.eye(n_bins)
        for j in range(dim):
            sol = np.linalg.lstsq(a_mat, np.concatenate([w * y[:, j], np.zeros(n_bins)]), rcond=None)[0]
            p[:, j], d[:, j] = sol[:n_k], sol[n_k:]
        seen = np.zeros(n_bins, dtype=bool)
        seen[rows_b] = True
        if not seen.all():
            idx = np.flatnonzero(seen)
            for j in range(dim):
                d[:, j] = np.interp(np.arange(n_bins), idx, d[idx, j])
        shift = d.mean(axis=0)
        d -= shift
        p += shift
        r = np.sqrt((((p[rows_k] + d[rows_b]) - y) ** 2).sum(axis=1))
        scale = max(2.0 * float(np.median(r)), 1.0)
        w = w0 * np.minimum(1.0, scale / np.maximum(r, 1e-9))
    r = np.sqrt((((p[rows_k] + d[rows_b]) - y) ** 2).sum(axis=1))
    return RigidDrift(d, p, float(np.sqrt(np.mean(r ** 2))), ids)


def drift_shape(units: list[Unit], geom: Geometry, axis: np.ndarray, min_bins: int = 3) -> np.ndarray:
    """The time course of the drift on a degenerate geometry, in arbitrary units.

    On a tetrode a source that moves relative to the array moves radially
    in fitted-position space, towards the centre when it approaches and
    away when it recedes, by an amount that depends on its height. So the
    direction of motion is read from which side of the drift `axis` each
    unit sits on, the unit's radial excursion over the bins it was seen in
    is flipped to that sign and scaled to unit variance, and the units are
    averaged (weighted by spike count) bin by bin. Units, not clusters: a
    band of a drifting neuron lives in one epoch and would lose the level
    of that epoch relative to the others. The result follows the drift in
    time but says nothing about how far the array moved; the sign follows
    the spike-displacement convention of `DriftEstimate`. Zero where no
    unit with at least `min_bins` observed bins had data.
    """
    u = np.asarray(axis, dtype=np.float64)
    u = u / max(np.linalg.norm(u), 1e-12)
    u_sub = geom.basis.T @ u  # the axis in the fitted subspace
    n_bins = units[0].positions.shape[0] if units else 0
    acc = np.zeros(n_bins)
    wsum = np.zeros(n_bins)
    for unit in units:
        seen = unit.observed
        if seen.sum() < min_bins:
            continue
        pos, w = unit.positions[seen], unit.counts[seen]
        radius = np.sqrt((pos ** 2).sum(axis=1))
        excursion = radius - np.average(radius, weights=w)
        if excursion.std() < 1e-9:
            continue
        side = np.sign(float(np.average(pos, axis=0, weights=w) @ u_sub)) or 1.0
        shape = side * excursion / excursion.std()
        acc[seen] += w * shape
        wsum[seen] += w
    out = np.zeros(n_bins)
    seen = wsum > 0
    out[seen] = acc[seen] / wsum[seen]
    if seen.any() and not seen.all():
        idx = np.flatnonzero(seen)
        out = np.interp(np.arange(n_bins), idx, out[idx])
    return out - out.mean() if n_bins else out
