# spikesort

A spike sorter written from scratch in NumPy and SciPy, benchmarked against simulated recordings where the true answer is known.

Spike sorting takes raw voltage from an extracellular electrode and works out which neuron fired each spike. Production sorters like Kilosort and MountainSort are large and hard to see inside. This one implements the classic pipeline in a few hundred readable lines, then measures exactly where it succeeds and where it fails.

![Demo run on a simulated tetrode](docs/img/summary.png)

## Pipeline

```
raw voltage
  -> bandpass 300-6000 Hz (zero-phase Butterworth)
  -> detect: negative threshold crossings at 5 x MAD noise, 1 ms dead time
  -> align: centre of mass of the trough, sub-sample, cubic spline resampling
  -> features: PCA on the concatenated multichannel waveform (6 components)
  -> cluster: Gaussian mixture, number of units chosen by BIC
  -> merge: join over-split clusters (template similarity + cross-fitted d')
  -> score: ground-truth matching (Hungarian) and GT-free quality metrics
```

| Module | What it does |
|---|---|
| `simulate.py` | Synthetic recordings: biphasic spike templates with distance-dependent amplitude across channels, Poisson spike trains with a refractory period, band-limited correlated noise, slow drift. Tetrode and linear probe geometries. |
| `preprocess.py` | Bandpass, common median reference, robust MAD noise estimate (Quiroga et al. 2004). |
| `detect.py` | Threshold detection, dead time, three alignment modes. |
| `features.py` | PCA via SVD, global or per-channel. |
| `cluster.py` | GMM with BIC, merge pass, optional split pass, cluster separation (d'). |
| `evaluate.py` | SpikeForest-style accuracy, Hungarian matching, ISI violations, SNR, isolation distance (Harris et al. 2001). |

## Quick start

```bash
pip install -e ".[dev]"
spikesort demo --units 5 --duration 60 --seed 0 --out results/
spikesort sort my_recording.npy --fs 30000 --out results/   # (n_samples, n_channels) array
pytest
```

The demo prints per-unit quality metrics and the score against ground truth:

```
unit  matched  n_gt   n_sorted  precision  recall  accuracy
   0        1    322       304      1.000   0.944     0.944
   1        4    181       172      1.000   0.950     0.950
   2        3    239       224      1.000   0.937     0.937
   3        2    264       249      1.000   0.943     0.943
   4        0    752       781      0.956   0.993     0.950
mean accuracy: 0.945
```

Accuracy follows the SpikeForest definition, matched / (ground truth + sorted - matched), so it penalises misses and false positives together.

## Results

Each point is the mean of 3 random seeds, 60 s recordings, simulated tetrode. Reproduce with `python benchmarks/sweep.py`.

![Benchmark sweep](docs/img/sweep.png)

| Condition | Accuracy | Precision | Recall |
|---|---|---|---|
| 5 units, noise 5 to 15 uV | 0.90 to 0.92 | 0.99 to 1.00 | 0.91 to 0.93 |
| 5 units, noise 20 uV | 0.87 | 1.00 | 0.87 |
| 5 units, noise 25 uV | 0.73 | 0.87 | 0.75 |
| 5 units, noise 30 uV | 0.55 | 0.86 | 0.57 |
| 2 units | 0.98 | 1.00 | 0.98 |
| 10 units | 0.81 | 1.00 | 0.82 |

### Finding 1: errors are almost all collisions

Splitting ground-truth spikes by whether another neuron fired within 1 ms (right panel above):

* isolated spikes: recall 0.95 to 1.00 at every unit count
* overlapping spikes: recall 0.02 to 0.26

Precision stays near 1.0, so the sorter rarely assigns a spike to the wrong neuron. It loses spikes when two neurons fire together, because the summed waveform matches neither template and lands in a junk cluster. As the unit count rises from 2 to 10, the collision fraction climbs from 2% to 15% of spikes, and that accounts for most of the accuracy drop. Threshold-and-cluster designs share this ceiling, which is why modern sorters add template matching to peel overlapping spikes apart (see Next steps).

### Finding 2: alignment is the biggest single lever

Reproduce with `python benchmarks/alignment.py`.

| Units | Noise (uV) | Integer | Parabola | Centre of mass |
|---|---|---|---|---|
| 5 | 5 | 0.69 | 0.91 | 0.90 |
| 5 | 10 | 0.85 | 0.92 | 0.92 |
| 5 | 15 | 0.82 | 0.85 | 0.92 |
| 5 | 20 | 0.82 | 0.82 | 0.87 |
| 5 | 25 | 0.61 | 0.64 | 0.73 |
| 2 | 10 | 0.65 | 0.98 | 0.98 |
| 8 | 10 | 0.76 | 0.83 | 0.84 |
| **all** | | **0.74** | **0.85** | **0.88** |

The first version aligned each spike to the sample holding its minimum. A filtered spike's true trough usually falls between two samples, so noise decides which one wins. Every neuron became two populations one sample apart, PCA separated them cleanly, and the clusterer reported each neuron twice. The giveaway was a clean 50/50 split of one neuron where the difference between the halves matched the waveform's time derivative.

Parabolic refinement fixes that at high SNR. For smaller spikes, the noisy minimum wanders several samples along a broad trough, and a parabola only corrects within half a sample. Aligning on the trough's centre of mass averages over many samples, so the noise mostly cancels.

Counterintuitively, integer alignment does worst at the *lowest* noise level. Clean data makes the two offset populations tight enough to look like separate neurons.

### Finding 3: global PCA beats per-channel PCA

The classic tetrode recipe runs PCA separately on each channel (3 components x 4 channels = 12 features). On an 8-channel probe that becomes 24 dimensions, and a full-covariance GMM needs about 300 parameters per component. With a few hundred spikes per neuron, BIC's penalty swamps the likelihood gain and similar neurons get merged. One PCA on the concatenated waveform, keeping 6 components, raised mean accuracy on an 8-channel probe from 0.49 to 0.90 and on a tetrode from 0.82 to 0.88 (before the alignment fix).

### Finding 4: score separation on held-out spikes

The merge pass joins two clusters only if their templates correlate above 0.97, their amplitudes are within 15%, and their spike clouds overlap (d' below 2.5 along the axis between their means). Measured naively, d' is inflated by double-dipping. A random split of one neuron scores d' of about 1 when the axis and the score come from the same 100 spikes in 60 dimensions. `separation()` is therefore cross-fitted: the axis comes from half the spikes and d' is scored on the other half.

There is also an optional split pass (`SortConfig(split_dprime=4.0)` or `spikesort demo --split`) that tests each cluster for a hidden second neuron, using the same held-out scoring. It helped with 8 to 10 units when BIC's search was capped at 12 components. Once the cap was raised to 20 it stopped firing, so it is off by default.

## Known limitations

* **Collisions** are the main loss of recall (Finding 1).
* **Near-identical neurons** cannot be separated by waveform alone. In one 4-unit seed, two neurons on the same side of the tetrode have templates correlating at 0.989 and ground-truth d' of 2.2. They are sorted as one unit, which explains the dip at 4 units in the sweep.
* **High noise.** Below about SNR 5 (noise 25 uV and up for these amplitudes), small units fall under the detection threshold and recall drops.
* **No drift correction.** Simulated neurons stay put. Real probes move, which is a major source of error in long recordings.
* **Simulated data only so far.** The simulator is simple: Gaussian-shaped spikes, a monopole-like falloff, no bursting with amplitude decay.

## Next steps

* Template matching to peel overlapping spikes apart (greedy subtraction or a Kilosort-style deconvolution). Aimed directly at Finding 1.
* Validation on real hybrid ground-truth data, for example SpikeForest recordings, via SpikeInterface for loading.
* Drift-aware clustering for long recordings.

## Project layout

```
src/spikesort/      the package
tests/              pytest suite (19 tests: unit tests per stage plus end-to-end)
benchmarks/         sweep.py and alignment.py, results in benchmarks/results/
docs/img/           figures used in this README
```

## References

* Quiroga, Nadasdy and Ben-Shaul (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation* 16(8).
* Harris, Hirase, Leinekugel, Henze and Buzsaki (2001). Temporal interaction between single spikes and complex spike bursts in hippocampal pyramidal cells. *Neuron* 32(1). (Isolation distance.)
* Schmitzer-Torbert, Jackson, Henze, Harris and Redish (2005). Quantitative measures of cluster quality for use in extracellular recordings. *Neuroscience* 131(1).
* Magland et al. (2020). SpikeForest, reproducible web-facing ground-truth validation of automated neural spike sorters. *eLife* 9.
* Pachitariu et al. (2024). Spike sorting with Kilosort4. *Nature Methods* 21.

## License

MIT
