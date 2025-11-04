from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from experiments.synthetic import power_null as power_null_mod
from experiments.synthetic.power_null import (
    DEFAULT_CONFIG,
    calibration_cache_meta,
    calibrate_delta_thresholds,
    load_calibration_cache,
    run_trials,
    summarise_results,
    write_calibration_cache,
)

pytestmark = pytest.mark.slow


def test_power_null_summary_behaves() -> None:
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 8,
        "n_groups": 8,
        "replicates": 2,
        "trials_null": 4,
        "trials_power": 4,
        "spike_grid": [3.5],
        "a_grid": 60,
    })
    rng = np.random.default_rng(99)
    original_gating = dict(power_null_mod.GATING_SETTINGS)
    power_null_mod.GATING_SETTINGS = {"default": original_gating["default"]}
    try:
        results = run_trials(
            config=config,
            edge_modes=["scm"],
            trials_null=int(config["trials_null"]),
            trials_power=int(config["trials_power"]),
            spike_grid=config["spike_grid"],
            two_spike=False,
            rng=rng,
        )
    finally:
        power_null_mod.GATING_SETTINGS = original_gating
    summary = summarise_results(results)
    assert not summary.empty

    null_rows = summary[summary["scenario"] == "null"]
    assert (null_rows["detection_rate"] < 0.5).all()

    power_rows = summary[summary["scenario"] == "power"]
    assert not power_rows.empty
    assert (power_rows["detection_rate"] > 0).any()
    assert power_rows["delta_mse_vs_lw"].notna().any()
    assert power_rows["delta_qlike_vs_lw"].notna().any()


def test_calibrate_delta_thresholds_returns_lookup() -> None:
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 6,
        "n_groups": 6,
        "replicates": 2,
        "trials_null": 6,
        "delta_frac": 0.01,
    })
    rng = np.random.default_rng(321)
    calibration = calibrate_delta_thresholds(
        config=config,
        edge_modes=["scm"],
        trials_null=int(config["trials_null"]),
        alpha=0.5,
        rng=rng,
        delta_grid=[0.0, 0.01, 0.02],
    )
    assert calibration.get("thresholds")
    scm_thresholds = calibration["thresholds"].get("scm")
    assert scm_thresholds, "Expected SCM entry in thresholds"
    key, entry = next(iter(scm_thresholds.items()))
    assert "x" in key  # key format "pxT"
    assert "delta_frac" in entry
    assert 0.0 <= entry["delta_frac"] <= 0.02


def test_calibrate_delta_thresholds_multi_panel_fpr_cap() -> None:
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 6,
        "n_groups": 3,
        "replicates": 3,
        "trials_null": 20,
        "delta_frac": 0.0,
    })
    rng = np.random.default_rng(404)
    calibration = calibrate_delta_thresholds(
        config=config,
        edge_modes=["scm"],
        trials_null=int(config["trials_null"]),
        alpha=0.02,
        rng=rng,
        delta_grid=[0.1],
        panel_specs=[(6, 3, 3)],
    )
    scm_thresholds = calibration["thresholds"].get("scm", {})
    key, entry = next(iter(scm_thresholds.items()))
    assert key == "6x9"
    assert entry["fpr"] <= 0.02 + 1e-9


def test_calibration_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "edge_delta_thresholds.json"
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 4,
        "n_groups": 4,
        "replicates": 2,
        "trials_null": 2,
        "delta_frac": 0.0,
    })
    rng = np.random.default_rng(11)
    payload = calibrate_delta_thresholds(
        config=config,
        edge_modes=["scm"],
        trials_null=int(config["trials_null"]),
        alpha=0.25,
        rng=rng,
        delta_grid=[0.0, 0.01],
    )
    meta = calibration_cache_meta(
        config=config,
        edge_modes=["scm"],
        alpha=0.25,
        trials_null=int(config["trials_null"]),
        delta_grid=[0.0, 0.01],
    )
    write_calibration_cache(cache_path, payload, meta)
    cached = load_calibration_cache(
        cache_path,
        meta,
        dependencies=[Path(__file__).resolve()],
    )
    assert cached is not None
    assert cached["thresholds"] == payload["thresholds"]

    dep_path = tmp_path / "dep.flag"
    dep_path.write_text("stale")
    os.utime(dep_path, (cache_path.stat().st_mtime + 60, cache_path.stat().st_mtime + 60))
    stale = load_calibration_cache(
        cache_path,
        meta,
        dependencies=[dep_path],
    )
    assert stale is None


def test_calibration_cache_force_recomputes(tmp_path: Path) -> None:
    cache_path = tmp_path / "edge_delta_thresholds.json"
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 4,
        "n_groups": 4,
        "replicates": 2,
        "trials_null": 2,
        "delta_frac": 0.0,
    })
    rng1 = np.random.default_rng(101)
    payload1 = calibrate_delta_thresholds(
        config=config,
        edge_modes=["scm"],
        trials_null=int(config["trials_null"]),
        alpha=0.2,
        rng=rng1,
        delta_grid=[0.0, 0.02],
    )
    meta = calibration_cache_meta(
        config=config,
        edge_modes=["scm"],
        alpha=0.2,
        trials_null=int(config["trials_null"]),
        delta_grid=[0.0, 0.02],
    )
    write_calibration_cache(cache_path, payload1, meta)
    cached = load_calibration_cache(
        cache_path,
        meta,
        dependencies=[Path(__file__).resolve()],
    )
    assert cached is not None

    rng2 = np.random.default_rng(202)
    payload2 = calibrate_delta_thresholds(
        config=config,
        edge_modes=["scm"],
        trials_null=int(config["trials_null"]),
        alpha=0.2,
        rng=rng2,
        delta_grid=[0.0, 0.02],
    )
    write_calibration_cache(cache_path, payload2, meta)
    loaded = load_calibration_cache(
        cache_path,
        meta,
        dependencies=[Path(__file__).resolve()],
    )
    assert loaded is not None
    payload2_copy = dict(payload2)
    loaded_copy = dict(loaded)
    loaded_copy.pop("_meta", None)
    assert loaded_copy == payload2_copy
