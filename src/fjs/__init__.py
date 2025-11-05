from __future__ import annotations

from .balanced import (
    BalancedConfig,
    compute_balanced_weights,
    group_means,
    mean_squares,
)
from .dealias import DealiasingResult, dealias_covariance, dealias_search
from .mp import MarchenkoPasturModel, marchenko_pastur_edges, marchenko_pastur_pdf

def _missing_matplotlib(*_args, **_kwargs):  # pragma: no cover - helper for soft dep
    raise ImportError("matplotlib is required for fjs.spectra plotting utilities.")


try:
    from .spectra import (
        estimate_spectrum,
        plot_spectrum_with_edges,
        plot_spike_timeseries,
        project_alignment,
        topk_eigh,
    )
except Exception:  # pragma: no cover - soft dependency (matplotlib)
    estimate_spectrum = _missing_matplotlib  # type: ignore[assignment]
    plot_spectrum_with_edges = _missing_matplotlib  # type: ignore[assignment]
    plot_spike_timeseries = _missing_matplotlib  # type: ignore[assignment]
    project_alignment = _missing_matplotlib  # type: ignore[assignment]
    topk_eigh = _missing_matplotlib  # type: ignore[assignment]

__all__ = [
    "BalancedConfig",
    "compute_balanced_weights",
    "group_means",
    "mean_squares",
    "DealiasingResult",
    "dealias_covariance",
    "dealias_search",
    "MarchenkoPasturModel",
    "marchenko_pastur_edges",
    "marchenko_pastur_pdf",
    "estimate_spectrum",
    "topk_eigh",
    "project_alignment",
    "plot_spectrum_with_edges",
    "plot_spike_timeseries",
]
