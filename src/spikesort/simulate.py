"""Synthetic extracellular recordings with known ground truth.

Each unit has a multichannel spike template (a biphasic waveform whose
amplitude decays with distance from the unit to each electrode). Spike
trains are Poisson with an absolute refractory period. Background noise is
band-limited Gaussian noise plus low-frequency drift so the filtering step
has something real to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class Recording:
    """A simulated recording and its ground truth."""

    data: np.ndarray  # (n_samples, n_channels), microvolts
    fs: float
    spike_times: np.ndarray  # sample index of each spike trough
    spike_labels: np.ndarray  # unit id for each spike
    templates: np.ndarray  # (n_units, n_template_samples, n_channels)
    channel_positions: np.ndarray  # (n_channels, 2), microns
    unit_positions: np.ndarray  # (n_units, 2), microns
    params: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return self.templates.shape[0]

    @property
    def duration(self) -> float:
        return self.data.shape[0] / self.fs


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
    """
    rng = np.random.default_rng(seed)
    if channel_positions is None:
        channel_positions = tetrode_geometry()
    n_ch = channel_positions.shape[0]
    n_samples = int(duration * fs)
    n_t = int(round(template_ms * 1e-3 * fs))

    # Place units in a box around the electrodes, a little off the plane.
    lo = channel_positions.min(axis=0) - 15
    hi = channel_positions.max(axis=0) + 15
    unit_xy = rng.uniform(lo, hi, size=(n_units, 2))
    unit_z = rng.uniform(10, 30, size=n_units)

    templates = np.zeros((n_units, n_t, n_ch))
    for u in range(n_units):
        base = spike_waveform(
            n_t, fs,
            width_ms=rng.uniform(0.10, 0.22),
            repol_ratio=rng.uniform(0.2, 0.6),
            repol_delay_ms=rng.uniform(0.25, 0.5),
        )
        d = np.sqrt(((channel_positions - unit_xy[u]) ** 2).sum(axis=1) + unit_z[u] ** 2)
        gain = (d.min() / d) ** 1.5  # monopole-ish falloff, normalised to nearest channel
        amp = rng.uniform(*amplitudes)
        templates[u] = amp * base[:, None] * gain[None, :]

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
    drift = 40.0 * np.sin(2 * np.pi * 3.0 * t)[:, None] + 20.0 * np.sin(2 * np.pi * 0.7 * t + 1)[:, None]
    data = noise + drift

    # Trough of the base waveform sits at sample `peak`; spike_times refer to it.
    peak = int(np.argmin(templates[0].min(axis=1)))
    for s, u in zip(times, labels, strict=True):
        data[s:s + n_t] += templates[u]
    spike_times = times + peak

    return Recording(
        data=data.astype(np.float32),
        fs=fs,
        spike_times=spike_times,
        spike_labels=labels,
        templates=templates,
        channel_positions=channel_positions,
        unit_positions=np.column_stack([unit_xy, unit_z]),
        params=dict(n_units=n_units, duration=duration, fs=fs, noise_uv=noise_uv,
                    rates=rates, amplitudes=amplitudes, seed=seed),
    )
