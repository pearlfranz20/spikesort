"""Feature extraction: PCA on concatenated multichannel waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    """Minimal PCA via SVD, so the math is visible.

    Fit on waveforms flattened to (n_spikes, n_samples * n_channels).
    """

    n_components: int = 3
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Economy SVD. Rows of Vt are the principal axes.
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        # Fix the sign so results are deterministic across runs and platforms.
        signs = np.sign(vt[np.arange(vt.shape[0]), np.abs(vt).argmax(axis=1)])
        vt *= signs[:, None]
        var = s ** 2
        self.components_ = vt[: self.n_components]
        self.explained_variance_ratio_ = var[: self.n_components] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def apply_features(waveforms: np.ndarray, models: list[PCA], per_channel: bool) -> np.ndarray:
    """Project new waveforms onto PCA models fitted by `waveform_features`."""
    n, t, c = waveforms.shape
    if not per_channel:
        return models[0].transform(waveforms.reshape(n, t * c))
    return np.hstack([m.transform(waveforms[:, :, ch]) for ch, m in enumerate(models)])


def waveform_features(waveforms: np.ndarray, n_components: int = 3,
                      per_channel: bool = True) -> tuple[np.ndarray, list[PCA]]:
    """PCA features from multichannel waveforms.

    With `per_channel=True` (the classic tetrode approach) each channel gets
    its own PCA and the components are concatenated, giving
    n_channels * n_components features. Otherwise one PCA runs on the
    flattened waveform.
    """
    n, t, c = waveforms.shape
    if not per_channel:
        pca = PCA(n_components).fit(waveforms.reshape(n, t * c))
        return pca.transform(waveforms.reshape(n, t * c)), [pca]
    feats, models = [], []
    for ch in range(c):
        pca = PCA(n_components).fit(waveforms[:, :, ch])
        feats.append(pca.transform(waveforms[:, :, ch]))
        models.append(pca)
    return np.hstack(feats), models
