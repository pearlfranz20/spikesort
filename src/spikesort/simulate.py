"""Synthetic extracellular recordings with known ground truth.

Each unit has a multichannel spike template (a biphasic waveform whose
amplitude decays with distance from the unit to each electrode). Spike
trains are Poisson with an absolute refractory period. Background noise is
band-limited Gaussian noise plus a low-frequency LFP-like offset so the
filtering step has something real to do.

Probe motion ("drift") is optional: a slow trajectory moves the electrodes
relative to the tissue, so each spike's per-channel amplitudes depend on
when it fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal


@dataclass
class Recording:
    """A simulated recording and its ground truth."""

    data: np.ndarray  # (n_samples, n_channels), microvolts
    fs: float
    spike_times: np.ndarray  # sample index of each spike trough
    spike_labels: np.ndarray  # unit id for each spike
    templates: np.ndarray  # (n_units, n_template_samples, n_channels), at zero motion
    channel_positions: np.ndarray  # (n_channels, 2), microns
    unit_positions: np.ndarray  # (n_units, 3), microns (x, y, height above the plane)
    params: dict = field(default_factory=dict)
    unit_waveforms: np.ndarray | None = None  # (n_units, n_template_samples), scaled base shapes
    motion_um: np.ndarray | None = None  # probe displacement along `motion_axis`, sampled at 1 kHz
    motion_axis: np.ndarray | None = None  # unit vector in the electrode plane

    @property
    def n_units(self) -> int:
        return self.templates.shape[0]

    @property
    def duration(self) -> float:
        return self.data.shape[0] / self.fs

    def motion_at(self, t_s: np.ndarray | float) -> np.ndarray | float:
        """Probe displacement (microns) at the given times in seconds."""
        if self.motion_um is None:
            return np.zeros_like(np.asarray(t_s, dtype=float))
        idx = np.clip((np.asarray(t_s) * 1000).astype(int), 0, self.motion_um.size - 1)
        return self.motion_um[idx]

    def gains_at(self, unit: int, displacement_um: float = 0.0) -> np.ndarray:
        """Per-channel amplitude factors for a unit with the probe displaced."""
        return _gains(self.channel_positions, self.unit_positions[unit],
                      displacement_um, self.motion_axis)

    def template_at(self, unit: int, t_s: float) -> np.ndarray:
        """The unit's template at time t, (n_template_samples, n_channels)."""
        g = self.gains_at(unit, float(self.motion_at(t_s)))
        return self.unit_waveforms[unit][:, None] * g[None, :]


def _gains(channel_positions: np.ndarray, unit_position: np.ndarray,
           displacement_um: float = 0.0, axis: np.ndarray | None = None) -> np.ndarray:
    """Monopole-like amplitude falloff, (d_min / d)^1.5, normalised to the
    nearest channel at zero displacement so that moving away from the whole
    array lowers every channel."""
    xy, z = unit_position[:2], unit_position[2]
    pos = channel_positions
    if displacement_um and axis is not None:
        pos = pos + displacement_um * axis[None, :]
    d0 = np.sqrt(((channel_positions - xy) ** 2).sum(axis=1) + z ** 2).min()
    d = np.sqrt(((pos - xy) ** 2).sum(axis=1) + z ** 2)
    return (d0 / d) ** 1.5


def probe_motion(duration: float, amplitude_um: float, kind: str = "sine",
                 rng: np.random.Generator | None = None, rate_hz: float = 1000.0) -> np.ndarray:
    """A slow displacement trajectory sampled at `rate_hz`, peak |value| = amplitude.

    ``"sine"``: one and a half cycles over the recording, starting at zero.
    ``"walk"``: a smoothed random walk (Gaussian smoothing, 2 s), rescaled.
    ``"ramp"``: linear from -amplitude to +amplitude.
    """
    n = int(round(duration * rate_hz))
    t = np.arange(n) / rate_hz
    if amplitude_um == 0:
        return np.zeros(n)
    if kind == "sine":
        return amplitude_um * np.sin(2 * np.pi * 1.5 * t / duration)
    if kind == "ramp":
        return np.linspace(-amplitude_um, amplitude_um, n)
    if kind == "walk":
        rng = rng or np.random.default_rng(0)
        steps = rng.standard_normal(n)
        walk = ndimage.gaussian_filter1d(np.cumsum(steps), 2.0 * rate_hz)
        walk -= walk.mean()
        return amplitude_um * walk / np.abs(walk).max()
    raise ValueError(f"unknown motion kind {kind!r}")


def tetrode_geometry(spacing: float = 25.0) -> np.ndarray:
    """Four channels on a square, like a tetrode."""
    h = spacing / 2
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=float)


def linear_probe_geometry(n_channels: int = 8, pitch: float = 20.0) -> np.ndarray:
    """A single column of channels, like a simple silicon probe shank."""
    y = np.arange(n_channels) * pitch
    return np.column_stack([np.zeros(n_channels), y])


def spike_waveform(
    n_samples: int,
    fs: float,
    trough_ms: float = 0.3,
    width_ms: float = 0.15,
    repol_ratio: float = 0.35,
    repol_delay_ms: float = 0.35,
) -> np.ndarray:
    """Biphasic extracellular spike: sharp negative trough, slower positive rebound.

    Normalised so the trough is -1.
    """
    t = np.arange(n_samples) / fs * 1e3  # ms
    trough = -np.exp(-0.5 * ((t - trough_ms) / width_ms) ** 2)
    rebound = repol_ratio * np.exp(
        -0.5 * ((t - trough_ms - repol_delay_ms) / (2.2 * width_ms)) ** 2
    )
    w = trough + rebound
    return w / -w.min()


def _poisson_train(rate: float, n_samples: int, fs: float, refractory_ms: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Spike times (samples) from a Poisson process with a dead time."""
    refractory = refractory_ms / 1e3
    duration = n_samples / fs
    times = []
    t = 0.0
    while True:
        t += refractory + rng.exponential(1.0 / rate)
        if t >= duration:
            break
        times.append(t)
    return (np.asarray(times) * fs).astype(np.int64)


def simulate_recording(
    n_units: int = 5,
    duration: float = 60.0,
    fs: float = 30_000.0,
    channel_positions: np.ndarray | None = None,
    rates: tuple[float, float] = (2.0, 15.0),
    amplitudes: tuple[float, float] = (60.0, 180.0),
    noise_uv: float = 10.0,
    refractory_ms: float = 2.0,
    template_ms: float = 2.0,
    seed: int | None = 0,
    motion_um: float = 0.0,
    motion_kind: str = "sine",
    motion_axis: tuple[float, float] = (0.0, 1.0),
    unit_margin_um: float = 15.0,
) -> Recording:
    """Simulate a recording.

    Parameters
    ----------
    n_units
        Number of ground-truth neurons.
    duration
        Seconds of data.
    fs
        Sampling rate in Hz.
    channel_positions
        (n_channels, 2) electrode positions in microns. Defaults to a tetrode.
    rates
        Uniform range of mean firing rates in Hz.
    amplitudes
        Uniform range of peak trough amplitude (microvolts) at the nearest channel.
    noise_uv
        Standard deviation of the band-limited background noise.
    refractory_ms
        Absolute refractory period.
    template_ms
        Template length in ms.
    seed
        RNG seed for reproducibility.
    motion_um
        Peak probe displacement in microns. Zero (default) reproduces the
        static recordings exactly, seed for seed.
    motion_kind
        "sine", "walk" or "ramp", see `probe_motion`.
    motion_axis
        Direction of motion in the electrode plane. (0, 1) is along a
        linear probe's shank.
    unit_margin_um
        Units are placed in the electrode bounding box grown by this much
        on every side. Negative values keep them inside the array, away
        from the ends, which matters for drift estimation on short probes.
    """
    rng = np.random.default_rng(seed)
    if channel_positions is None:
        channel_positions = tetrode_geometry()
    n_ch = channel_positions.shape[0]
    n_samples = int(duration * fs)
    n_t = int(round(template_ms * 1e-3 * fs))

    # Place units in a box around the electrodes, a little off the plane.
    lo = channel_positions.min(axis=0) - unit_margin_um
    hi = channel_positions.max(axis=0) + unit_margin_um
    centre = 0.5 * (lo + hi)
    lo, hi = np.minimum(lo, centre), np.maximum(hi, centre)  # a negative margin on a flat axis
    unit_xy = rng.uniform(lo, hi, size=(n_units, 2))
    unit_z = rng.uniform(10, 30, size=n_units)

    unit_positions = np.column_stack([unit_xy, unit_z])
    axis = np.asarray(motion_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    templates = np.zeros((n_units, n_t, n_ch))
    unit_waveforms = np.zeros((n_units, n_t))
    for u in range(n_units):
        base = spike_waveform(
            n_t, fs,
            width_ms=rng.uniform(0.10, 0.22),
            repol_ratio=rng.uniform(0.2, 0.6),
            repol_delay_ms=rng.uniform(0.25, 0.5),
        )
        amp = rng.uniform(*amplitudes)
        unit_waveforms[u] = amp * base
        templates[u] = unit_waveforms[u][:, None] * _gains(channel_positions, unit_positions[u])[None, :]

    # Spike trains.
    times, labels = [], []
    for u in range(n_units):
        rate = rng.uniform(*rates)
        st = _poisson_train(rate, n_samples - n_t, fs, refractory_ms, rng)
        times.append(st)
        labels.append(np.full(st.size, u, dtype=np.int64))
    times = np.concatenate(times)
    labels = np.concatenate(labels)
    order = np.argsort(times, kind="stable")
    times, labels = times[order], labels[order]

    # Background noise: white noise shaped into the spike band, correlated a
    # little across channels, plus slow drift (LFP-like) that filtering removes.
    white = rng.standard_normal((n_samples, n_ch))
    mix = 0.8 * np.eye(n_ch) + 0.2 / n_ch
    white = white @ mix
    b, a = signal.butter(2, [200, 8000], btype="band", fs=fs)
    noise = signal.lfilter(b, a, white, axis=0)
    noise *= noise_uv / noise.std()
    t = np.arange(n_samples) / fs
    lfp = 40.0 * np.sin(2 * np.pi * 3.0 * t)[:, None] + 20.0 * np.sin(2 * np.pi * 0.7 * t + 1)[:, None]
    data = noise + lfp

    # Probe motion. Only drawn from the RNG when asked for, so motion_um=0
    # leaves the static recordings bit-for-bit unchanged.
    motion = probe_motion(duration, motion_um, motion_kind, rng) if motion_um else None

    # Trough of the base waveform sits at sample `peak`; spike_times refer to it.
    peak = int(np.argmin(templates[0].min(axis=1)))
    for s, u in zip(times, labels, strict=True):
        if motion is None:
            data[s:s + n_t] += templates[u]
        else:
            disp = motion[min(motion.size - 1, int(s / fs * 1000))]
            g = _gains(channel_positions, unit_positions[u], disp, axis)
            data[s:s + n_t] += unit_waveforms[u][:, None] * g[None, :]
    spike_times = times + peak

    return Recording(
        data=data.astype(np.float32),
        fs=fs,
        spike_times=spike_times,
        spike_labels=labels,
        templates=templates,
        channel_positions=channel_positions,
        unit_positions=unit_positions,
        params=dict(n_units=n_units, duration=duration, fs=fs, noise_uv=noise_uv,
                    rates=rates, amplitudes=amplitudes, seed=seed, motion_um=motion_um,
                    motion_kind=motion_kind),
        unit_waveforms=unit_waveforms,
        motion_um=motion,
        motion_axis=axis,
    )
