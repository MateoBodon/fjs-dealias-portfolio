from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def topk_eigh(
    matrix: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the largest ``k`` eigenpairs of a symmetric matrix.

    Parameters
    ----------
    matrix
        Symmetric matrix shaped ``(n, n)``.
    k
        Number of leading eigenpairs to return.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Eigenvalues and eigenvectors ordered from largest to smallest.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Input matrix must be square.")
    n = array.shape[0]
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n.")

    eigvals, eigvecs = linalg.eigh(array, subset_by_index=(n - k, n - 1))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def project_alignment(
    vector: NDArray[np.float64], subspace: NDArray[np.float64]
) -> float:
    """Compute the projection norm of ``vector`` onto the span of ``subspace``.

    Parameters
    ----------
    vector
        Ambient vector to project.
    subspace
        Matrix whose columns span the target subspace.

    Returns
    -------
    float
        Euclidean norm of the projected vector.
    """

    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    basis = np.asarray(subspace, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("subspace must be a 2-D array with column vectors.")
    if basis.shape[0] != v.shape[0]:
        raise ValueError("Vector and subspace must share the same ambient dimension.")
    if basis.shape[1] == 0:
        return 0.0

    q, _ = np.linalg.qr(basis, mode="reduced")
    projection = q @ (q.T @ v)
    return float(np.linalg.norm(projection))


def _ensure_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_spectrum_with_edges(
    eigenvalues: Sequence[float],
    edges: Iterable[float] | None,
    out_path: str | Path,
    *,
    title: str | None = None,
    xlabel: str = "Eigenvalue index (descending)",
    ylabel: str = "Eigenvalue",
    highlight_threshold: float | None = None,
    highlight_color: str = "C3",
) -> Path:
    """Plot an empirical spectrum together with optional reference edge lines.

    Parameters
    ----------
    eigenvalues
        Sequence of eigenvalues.
    edges
        Iterable of reference edge locations; ``None`` skips overlays.
    out_path
        Output path for the saved figure (PNG/PDF).
    title, xlabel, ylabel
        Plot annotations.

    Returns
    -------
    pathlib.Path
        Path where the figure was saved.
    """

    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional.")
    sorted_vals = np.sort(values)[::-1]
    indices = np.arange(1, len(sorted_vals) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(indices, sorted_vals, marker="o", linestyle="-", label="Spectrum")

    if edges is not None:
        edge_values = list(edges)
        if edge_values:
            upper_edge = max(edge_values)
            ax.axhline(
                upper_edge,
                color="C1",
                linestyle="--",
                linewidth=1.2,
                label="MP upper edge",
            )

    if highlight_threshold is not None:
        thr = float(highlight_threshold)
        mask = sorted_vals > thr
        if mask.any():
            ax.scatter(
                indices[mask],
                sorted_vals[mask],
                color=highlight_color,
                s=30,
                label="> edge",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    output_path = _ensure_path(out_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_spike_timeseries(
    time_index: Sequence[float],
    aliased_series: Sequence[float],
    dealiased_series: Sequence[float],
    out_path: str | Path,
    *,
    title: str | None = None,
    true_value: float | None = None,
    xlabel: str = "Prefix groups",
    ylabel: str = "Spike magnitude",
) -> Path:
    """Plot aliased and de-aliased spike estimates against a time index.

    Parameters
    ----------
    time_index
        Sequence of time points at which estimates are available.
    aliased_series, dealiased_series
        Corresponding spike estimates.
    out_path
        Output path for the saved figure.
    title, true_value, xlabel, ylabel
        Plot annotations.

    Returns
    -------
    pathlib.Path
        Path where the figure was saved.
    """

    times = np.asarray(time_index, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("time_index must be one-dimensional.")
    aliased = np.asarray(aliased_series, dtype=np.float64)
    dealiased = np.asarray(dealiased_series, dtype=np.float64)
    if aliased.shape != times.shape or dealiased.shape != times.shape:
        raise ValueError("Series must align with time_index.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, aliased, color="C0", linewidth=1.6, label="Aliased")
    ax.plot(times, dealiased, color="C1", linewidth=1.6, label="De-aliased")
    if true_value is not None:
        ax.axhline(
            true_value, color="C2", linestyle="--", linewidth=1.2, label="True value"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    output_path = _ensure_path(out_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def estimate_spectrum(
    eigenvalues: Sequence[float],
    *,
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """Return a sorted copy of ``eigenvalues`` (placeholder estimator).

    Parameters
    ----------
    eigenvalues
        Raw eigenvalues to be sorted.
    bandwidth
        Optional smoothing bandwidth (unused placeholder).

    Returns
    -------
    numpy.ndarray
        Sorted eigenvalues.
    """

    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Eigenvalues must be one-dimensional.")
    if bandwidth is not None and bandwidth <= 0:
        raise ValueError("bandwidth must be positive when supplied.")
    return np.sort(values)
