# spikesort

A spike sorter written from scratch in NumPy and SciPy, benchmarked against simulated recordings where the true answer is known.

Spike sorting takes raw voltage from an extracellular electrode and works out which neuron fired each spike. Production sorters like Kilosort and MountainSort are large and hard to see inside. This one implements the classic two-stage design in a few hundred readable lines, then measures exactly where it succeeds and where it fails: stage one detects and clusters spikes to learn one template per neuron, and stage two goes back to the raw trace and explains it as a sum of those templates, which is what recovers spikes from two neurons that fired at the same time.

![Demo run on a simulated tetrode](docs/img/summary.png)

## Pipeline

```
raw voltage
  -> bandpass 300-6000 Hz (zero-phase Butterworth)
  stage one
  -> detect: negative threshold crossings at 5 x MAD noise, 1 ms dead time
  -> align: centre of mass of the trough, sub-sample, cubic spline resampling
  -> features: PCA on the concatenated multichannel waveform (6 components)
  -> cluster: Gaussian mixture, number of units chosen by BIC
  -> merge: join over-split clusters (template similarity + cross-fitted d')
  -> templates: median waveform per cluster, composite clusters dropped
  stage two
  -> match: greedy matching pursuit on the filtered trace, sub-sample shifts,
            positions and amplitudes refined with neighbours subtracted
  -> score: ground-truth matching (Hungarian) and GT-free quality metrics
```

| Module | What it does |
|---|---|
| `simulate.py` | Synthetic recordings: biphasic spike templates with distance-dependent amplitude across channels, Poisson spike trains with a refractory period, band-limited correlated noise, slow drift. Tetrode and linear probe geometries. |
| `preprocess.py` | Bandpass, common median reference, robust MAD noise estimate (Quiroga et al. 2004). |
| `detect.py` | Threshold detection, dead time, three alignment modes. |
| `features.py` | PCA via SVD, global or per-channel. |
| `cluster.py` | GMM with BIC, merge pass, optional split pass, cluster separation (d'). |
| `matching.py` | Template selection, FFT matched filtering, greedy matching pursuit with refinement. |
| `evaluate.py` | SpikeForest-style accuracy, Hungarian matching, ISI violations, SNR, isolation distance (Harris et al. 2001), collision analysis. |

## Quick start

```bash
pip install -e ".[dev]"
spikesort demo --units 5 --duration 60 --seed 0 --out results/
spikesort demo --no-matching ...                                # stage one only
spikesort sort my_recording.npy --fs 30000 --out results/       # (n_samples, n_channels) array
pytest
```

The demo prints what each stage did, per-unit quality metrics, and the score against ground truth:

```
stage 1: 1730 events in 5 clusters
stage 2: 3 matching rounds, 1748 spikes in 5 units
...
unit  matched  n_gt   n_sorted  precision  recall  accuracy
   0        1    322       320      0.991   0.984     0.975
   1        4    181       180      1.000   0.994     0.994
   2        3    239       240      0.979   0.983     0.963
   3        2    264       262      0.985   0.977     0.963
   4        0    752       746      0.996   0.988     0.984
mean accuracy: 0.976
```

Accuracy follows the SpikeForest definition, matched / (ground truth + sorted - matched), so it penalises misses and false positives together.

## Results

Each point is the mean of 3 random seeds, 60 s recordings, simulated tetrode. Three sorters are compared: stage one alone, the full pipeline, and matching with templates built from the true labels, which bounds what matching can do given perfect stage-one output. Reproduce with `python benchmarks/sweep.py`.

![Benchmark sweep](docs/img/sweep.png)

| Condition | Clustering only | With matching | True templates |
|---|---|---|---|
| 5 units, noise 5 uV | 0.90 | 0.99 | 0.99 |
| 5 units, noise 10 uV | 0.92 | 0.99 | 0.99 |
| 5 units, noise 15 uV | 0.92 | 0.99 | 0.99 |
| 5 units, noise 20 uV | 0.87 | 0.99 | 0.99 |
| 5 units, noise 25 uV | 0.73 | 0.82 | 0.98 |
| 5 units, noise 30 uV | 0.55 | 0.79 | 0.95 |
| 2 units, noise 10 uV | 0.98 | 1.00 | 1.00 |
| 4 units, noise 10 uV | 0.82 | 0.87 | 1.00 |
| 6 units, noise 10 uV | 0.88 | 0.98 | 0.99 |
| 8 units, noise 10 uV | 0.84 | 0.97 | 0.97 |
| 10 units, noise 10 uV | 0.81 | 0.97 | 0.97 |

| Across the unit-count sweep | Clustering only | With matching | True templates |
|---|---|---|---|
| recall, isolated spikes | 0.97 | 0.99 | 1.00 |
| recall, overlapping spikes (another unit within 1 ms) | 0.09 | 0.90 | 0.92 |
| precision | 0.98 | 0.97 | 0.99 |

### Finding 1: clustering alone loses almost every collision

With stage one alone, isolated spikes are recovered 95 to 100% of the time at every unit count, while spikes with another neuron firing within 1 ms are recovered 2 to 26% of the time. Precision stays near 1.0, so the problem is not confusion between neurons. When two spikes overlap, the summed waveform matches neither template and lands in a junk cluster. As the unit count rises from 2 to 10, the collision fraction climbs from 2% to 15% of all spikes, and that accounts for most of the drop in accuracy. Every threshold-and-cluster sorter shares this ceiling.

### Finding 2: template matching removes that ceiling, and what is left is unit discovery

![A resolved collision: filtered trace, fitted templates, residual](docs/img/deconvolution.png)

Stage two takes the stage-one templates back to the filtered trace and runs matching pursuit: find the template and time whose subtraction removes the most energy, subtract it, repeat. Removing the first spike of a collision exposes the second. Recall on overlapping spikes goes from 0.09 to 0.90, against 0.92 with the true templates. The figure shows two units firing 0.6 ms apart, fitted down to noise.

In the 28 of 33 runs where stage one found every neuron, the full pipeline is within 0.012 of the true-template bound in every run and within 0.001 on average (mean accuracy 0.986, against 0.884 for clustering alone). The remaining gap, visible at 25 to 30 uV noise and in the 4-unit case, comes entirely from stage one failing to produce a template: a small unit lost under the detection threshold, or two near-identical neurons merged into one cluster. Matching cannot fit a template it was never given, and the missing unit's spikes get assigned to the closest existing one, which is why precision dips in exactly those runs.

The matched filter is also a much better detector than a single-sample threshold. In the 30 uV run where stage one did find every unit, the 5 sigma threshold had caught 63% of isolated spikes, because the smaller units peak at 3 to 4 sigma. The matched filter integrates over the whole multichannel waveform, and once a template exists it recovered 99% of them, taking that run from 0.65 to 0.96.

Five details that mattered, each found by watching the method fail:

* **Detect with unit amplitude, estimate amplitude afterwards.** Letting the amplitude float during detection let the first spike of a close collision swallow part of the second, which then fell under threshold. Detection now uses fixed-amplitude templates, and amplitudes are fitted at the end with neighbours removed.
* **Rank fits by absolute energy removed.** Ranking by the fraction of a template's own energy let a small template fitted at 1.25x beat the correct fit of a large one.
* **Refine with neighbours subtracted.** A greedy fit made next to an unexplained spike is pulled towards it. After each round, every spike near a new fit is restored and re-fitted against the cleaner residual, which corrected timing errors of 1 to 2 samples and amplitude errors of 20%.
* **Let the re-fit change its mind about the template.** An earlier version of the figure above showed a three-spike pile-up fitted down to noise. Checking it against ground truth showed two spikes: the sorter had explained one large spike as two smaller templates stacked on top of each other, both pinned at the bottom of the amplitude range. The re-fit now considers every template at the restored position, so the right one takes over and the other falls under threshold.
* **Keep the acceptance threshold low and add a noise floor.** A collision partner sitting on the first spike's rebound has a correlation well below its template energy. Requiring only that a fit reduce the residual energy, plus a 5 sigma floor measured from the matched-filter output itself, lifted collision recall from 0.46 to 0.60 on one seed with no change in precision.

### Finding 3: alignment is the biggest lever in stage one

Reproduce with `python benchmarks/alignment.py` (stage one only).

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

### Finding 4: global PCA beats per-channel PCA

The classic tetrode recipe runs PCA separately on each channel (3 components x 4 channels = 12 features). On an 8-channel probe that becomes 24 dimensions, and a full-covariance GMM needs about 300 parameters per component. With a few hundred spikes per neuron, BIC's penalty swamps the likelihood gain and similar neurons get merged. One PCA on the concatenated waveform, keeping 6 components, raised stage-one accuracy on an 8-channel probe from 0.49 to 0.90 and on a tetrode from 0.82 to 0.88 (before the alignment fix).

### Finding 5: score separation on held-out spikes

The merge pass joins two clusters only if their templates correlate above 0.97, their amplitudes are within 15%, and their spike clouds overlap (d' below 2.5 along the axis between their means). Measured naively, d' is inflated by double-dipping. A random split of one neuron scores d' of about 1 when the axis and the score come from the same 100 spikes in 60 dimensions. `separation()` is therefore cross-fitted: the axis comes from half the spikes and d' is scored on the other half.

There is also an optional split pass (`SortConfig(split_dprime=4.0)` or `spikesort demo --split`) that tests each cluster for a hidden second neuron, using the same held-out scoring. It helped with 8 to 10 units when BIC's search was capped at 12 components. Once the cap was raised to 20 it stopped firing, so it is off by default.

## Known limitations

* **Unit discovery at low SNR.** A neuron whose spikes mostly stay under the 5 sigma detection threshold never gets a clean cluster, so stage two has no template for it (noise 25 to 30 uV, two of three seeds). Detecting with the matched filter and then re-clustering would likely fix this.
* **Near-identical neurons** cannot be separated by waveform alone. In one 4-unit seed, two neurons on the same side of the tetrode have templates correlating at 0.989 and ground-truth d' of 2.2. They are sorted as one unit, which explains the dip at 4 units in the sweep.
* **Crowded probes.** With 8 to 10 units on 4 channels, accuracy settles at 0.97 and collision recall at 0.84 to 0.89, the same as with true templates. The residual errors are confusions between similar templates inside collisions, and more channels would be the real fix.
* **Amplitude variation.** The simulator gives every spike of a unit the same amplitude. Real units burst with decaying amplitudes, which the 0.8 to 1.25 amplitude window and the detection step are not tuned for.
* **No drift correction.** Simulated neurons stay put. Real probes move, which is a major source of error in long recordings.
* **Simulated data only so far.** The simulator is simple: Gaussian-shaped spikes, a monopole-like falloff, no bursting.

## Next steps

* Iterate: detect with the matched filter, re-cluster, re-match. Aimed at the low-SNR unit-discovery gap.
* Validation on real hybrid ground-truth data, for example SpikeForest recordings, via SpikeInterface for loading.
* Drift: add slow template drift to the simulator, measure the damage, then correct for it.

## Project layout

```
src/spikesort/      the package
tests/              pytest suite (25 tests: unit tests per stage plus end-to-end)
benchmarks/         sweep.py and alignment.py, results in benchmarks/results/
docs/img/           figures used in this README
```

## References

* Quiroga, Nadasdy and Ben-Shaul (2004). Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. *Neural Computation* 16(8).
* Harris, Hirase, Leinekugel, Henze and Buzsaki (2001). Temporal interaction between single spikes and complex spike bursts in hippocampal pyramidal cells. *Neuron* 32(1). (Isolation distance.)
* Schmitzer-Torbert, Jackson, Henze, Harris and Redish (2005). Quantitative measures of cluster quality for use in extracellular recordings. *Neuroscience* 131(1).
* Pillow, Shlens, Chichilnisky and Simoncelli (2013). A model-based spike sorting algorithm for removing correlation artifacts in multi-neuron recordings. *PLoS ONE* 8(5). (Greedy template subtraction for overlapping spikes.)
* Magland et al. (2020). SpikeForest, reproducible web-facing ground-truth validation of automated neural spike sorters. *eLife* 9.
* Pachitariu et al. (2024). Spike sorting with Kilosort4. *Nature Methods* 21.

## License

MIT
