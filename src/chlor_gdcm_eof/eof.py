"""EOF/PCA helpers used to initialize missing chlorophyll-a fields."""

from __future__ import annotations

import numpy as np


def func_eofszb(data: np.ndarray):
    """Compute EOF-like principal components and cumulative explained variance.

    The original notebook imported ``func_eofszb`` from an external ``utils`` module that
    was not present in this project folder. This local implementation provides the
    behavior needed by the dataset code: it returns cumulative explained variance so the
    number of EOF modes can be selected, along with common EOF/PCA outputs.
    """
    data = np.asarray(data, dtype=np.float64)
    mean_eof = np.nanmean(data, axis=0, keepdims=True)
    centered = np.nan_to_num(data - mean_eof, nan=0.0)

    if centered.size == 0:
        raise ValueError("EOF input data is empty.")

    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2
    total = eigenvalues.sum()
    if total <= 0:
        cumulative = np.ones_like(eigenvalues)
    else:
        cumulative = np.cumsum(eigenvalues) / total
    pca = u * singular_values
    eigenvectors = vt.T
    return pca, eigenvectors, eigenvalues, cumulative, mean_eof.squeeze()
