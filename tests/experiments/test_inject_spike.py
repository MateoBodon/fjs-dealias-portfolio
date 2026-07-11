import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.eval import inject_spike

from fjs.balanced import mean_squares
from fjs.overlay import OverlayConfig, detect_spikes


@pytest.mark.unit
def test_injection_increases_top_eigenvalue() -> None:
    rng = np.random.default_rng(0)
    base = np.zeros((60, 12), dtype=np.float64)
    labels = np.zeros(base.shape[0], dtype=np.intp)
    basis = inject_spike._make_injection_basis(base, labels, rng, inject_mode="total")
    injected_low = inject_spike._apply_injection(base, basis, mu=1.0)
    injected_high = inject_spike._apply_injection(base, basis, mu=4.0)

    def top_eig(matrix: np.ndarray) -> float:
        cov = np.cov(matrix, rowvar=False, ddof=1)
        eigvals = np.linalg.eigvalsh(cov)
        return float(eigvals.max())

    base_top = top_eig(base)
    low_top = top_eig(injected_low)
    high_top = top_eig(injected_high)

    assert low_top > base_top + 1e-9
    assert high_top > low_top + 1e-9


@pytest.mark.unit
def test_curve_csv_writer_columns(tmp_path: Path) -> None:
    rows = [
        {
            "mu": 0.0,
            "inject_mode": "total",
            "detection_rate": 0.1,
            "acceptance_rate": 0.05,
            "n_windows": 10,
            "n_detected": 1,
            "n_accepted": 0,
        },
        {
            "mu": 1.0,
            "inject_mode": "total",
            "detection_rate": 0.6,
            "acceptance_rate": 0.4,
            "n_windows": 10,
            "n_detected": 6,
            "n_accepted": 4,
        },
    ]
    df = inject_spike._build_curve_dataframe(rows)
    out_path = tmp_path / "curve.csv"
    df.to_csv(out_path, index=False)

    loaded = pd.read_csv(out_path)
    assert not loaded.empty
    assert list(loaded.columns) == [
        "mu",
        "inject_mode",
        "detection_rate",
        "acceptance_rate",
        "n_windows",
        "n_detected",
        "n_accepted",
    ]


@pytest.mark.unit
def test_max_windows_sampling_deterministic() -> None:
    indices_a = inject_spike._select_window_indices(20, 5, "random", seed=123)
    indices_b = inject_spike._select_window_indices(20, 5, "random", seed=123)
    assert indices_a == indices_b
    assert len(indices_a) == 5


@pytest.mark.unit
def test_windows_detail_and_gating_outputs(tmp_path: Path) -> None:
    rows = [
        {
            "window_idx": 0,
            "fit_start": "2024-01-02",
            "fit_end": "2024-01-03",
            "horizon_start": "2024-01-04",
            "horizon_end": "2024-01-05",
            "n_obs": 10,
            "n_assets": 4,
            "injected": 0,
            "injected_mu": None,
            "inject_mode": "total",
            "detected_initial": 1,
            "accepted": 0,
            "guard_edge_buffer": 2,
            "gate_reason_edge_margin": 1,
        }
    ]
    detail_df = inject_spike._build_windows_detail_dataframe(rows)
    detail_path = tmp_path / "windows_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    gating_df = inject_spike._build_gating_reasons_dataframe(detail_df)
    gating_path = tmp_path / "gating_reasons.csv"
    gating_df.to_csv(gating_path, index=False)

    loaded_detail = pd.read_csv(detail_path)
    assert set(inject_spike.WINDOW_DETAIL_REQUIRED_COLUMNS).issubset(
        loaded_detail.columns
    )
    loaded_gating = pd.read_csv(gating_path)
    assert {"stage", "reason", "count", "injected_mu", "inject_mode"}.issubset(
        loaded_gating.columns
    )


@pytest.mark.unit
def test_injection_modes_group_means() -> None:
    base = np.zeros((12, 5), dtype=np.float64)
    labels = np.repeat(np.arange(3, dtype=np.intp), 4)

    rng_total = np.random.default_rng(1)
    basis_total = inject_spike._make_injection_basis(
        base, labels, rng_total, inject_mode="total"
    )
    injected_total = inject_spike._apply_injection(base, basis_total, mu=1.0)
    series_total = injected_total @ basis_total.direction
    assert abs(float(np.mean(series_total))) < 1e-8
    assert abs(float(np.std(series_total)) - 1.0) < 1e-8

    rng_between = np.random.default_rng(2)
    basis_between = inject_spike._make_injection_basis(
        base, labels, rng_between, inject_mode="between"
    )
    injected_between = inject_spike._apply_injection(base, basis_between, mu=1.0)
    series_between = injected_between @ basis_between.direction
    group_means_between = [
        float(np.mean(series_between[labels == group])) for group in np.unique(labels)
    ]
    assert float(np.var(group_means_between)) > 1e-6

    rng_within = np.random.default_rng(3)
    basis_within = inject_spike._make_injection_basis(
        base, labels, rng_within, inject_mode="within"
    )
    injected_within = inject_spike._apply_injection(base, basis_within, mu=1.0)
    series_within = injected_within @ basis_within.direction
    group_means_within = [
        float(np.mean(series_within[labels == group])) for group in np.unique(labels)
    ]
    assert max(abs(val) for val in group_means_within) < 1e-8


@pytest.mark.unit
def test_missing_config_path_fails(tmp_path: Path) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("date,A\n2024-01-02,0.01\n", encoding="utf-8")
    args = inject_spike.parse_args(
        [
            "--returns-csv",
            str(returns_path),
            "--config",
            str(tmp_path / "missing.yaml"),
        ]
    )
    with pytest.raises(FileNotFoundError):
        inject_spike._resolve_eval_config_or_fail(args)


@pytest.mark.unit
def test_debug_window_fails_closed_without_tvec_errors() -> None:
    fixture_path = Path("tests/fixtures/debug_window_week_no_root.npz")
    assert fixture_path.exists()
    with np.load(fixture_path, allow_pickle=True) as data:
        obs = data["matrix"]
        groups = data["group_labels"]
        metadata = json.loads(data["metadata"].item())

    cfg_payload = metadata["overlay_config"]
    overlay_cfg = OverlayConfig(**cfg_payload)
    stats = mean_squares(obs, groups)
    detections = detect_spikes(
        obs,
        groups,
        config=overlay_cfg,
        stats=stats,
    )
    diagnostics = stats.get("diagnostics", {})
    assert detections == []
    assert stats["pre_gate"]["raw_outliers_found"] == 0
    assert diagnostics.get("tvec_compute_error", 0) == 0
    assert diagnostics.get("tvec_no_real_root", 0) == 0
