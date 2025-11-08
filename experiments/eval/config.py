from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(
                merged[key], value  # type: ignore[arg-type]
            )
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalise_layer(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalised = dict(payload)
    if "out" in normalised and "out_dir" not in normalised:
        normalised["out_dir"] = normalised.pop("out")
    return normalised


DEFAULTS: dict[str, Any] = {
    "window": 126,
    "horizon": 21,
    "shrinker": "rie",
    "seed": 0,
    "out_dir": "reports/eval-latest",
    "assets_top": None,
    "calm_quantile": 0.25,
    "crisis_quantile": 0.75,
    "vol_ewma_span": 21,
    "reason_codes": True,
    "echo_config": True,
    "overlay_a_grid": 60,
    "overlay_seed": None,
    "overlay_delta_frac": None,
    "mv_gamma": 1e-4,
    "mv_tau": 0.0,
    "mv_box_lo": 0.0,
    "mv_box_hi": 0.1,
    "mv_turnover_bps": 5.0,
    "mv_condition_cap": 1e6,
    "bootstrap_samples": 0,
    "require_isolated": True,
    "q_max": 1,
    "edge_mode": "tyler",
    "angle_min_cos": None,
    "alignment_top_p": 3,
    "cs_drop_top_frac": None,
    "prewhiten": "ff5mom",
    "use_factor_prewhiten": True,
    "calm_window_sample": None,
    "crisis_window_top_k": None,
    "group_design": "week",
    "group_min_count": 5,
    "group_min_replicates": 3,
    "min_reps_dow": 20,
    "min_reps_vol": 15,
    "max_missing_asset": 0.05,
    "max_missing_group_row": 0.0,
    "ewma_halflife": 30.0,
    "gate_mode": "strict",
    "gate_soft_max": None,
    "gate_delta_calibration": None,
    "gate_delta_frac_min": None,
    "gate_delta_frac_max": None,
    "gate_stability_min": 0.3,
    "gate_alignment_min": None,
    "gate_accept_nonisolated": False,
}


@dataclass(slots=True)
class ResolveResult:
    config: "EvalConfig"
    resolved: dict[str, Any]


def resolve_eval_config(args: Mapping[str, Any]) -> ResolveResult:
    from experiments.eval.run import EvalConfig  # Local import to avoid cycle

    config_path = args.get("config")
    thresholds_path = args.get("thresholds")

    config_path_obj = Path(config_path) if config_path else None
    thresholds_path_obj = Path(thresholds_path) if thresholds_path else None

    defaults = dict(DEFAULTS)
    layers: list[dict[str, Any]] = [defaults]

    thresholds_data = _normalise_layer(
        _load_json(thresholds_path_obj or DEFAULT_THRESHOLDS_PATH)
    )
    if thresholds_data:
        layers.append(thresholds_data)

    yaml_data = _normalise_layer(_load_yaml(config_path_obj or DEFAULT_CONFIG_PATH))
    if yaml_data:
        layers.append(yaml_data)

    cli_data: dict[str, Any] = {}
    for key, value in args.items():
        if key in {"config", "thresholds"}:
            continue
        if value is None:
            continue
        cli_data[key] = value
    cli_data = _normalise_layer(cli_data)
    if cli_data:
        layers.append(cli_data)

    merged: dict[str, Any] = {}
    for layer in layers:
        merged = _deep_merge(merged, layer)

    returns_csv = merged.get("returns_csv")
    if returns_csv is None:
        raise ValueError("returns_csv must be provided via CLI or configuration.")

    factors_csv = merged.get("factors_csv")
    out_dir = merged.get("out_dir")

    require_iso_raw = merged.get("require_isolated", DEFAULTS["require_isolated"])
    if require_iso_raw is None:
        require_isolated = bool(DEFAULTS["require_isolated"])
    else:
        require_isolated = bool(require_iso_raw)

    q_max_raw = merged.get("q_max", DEFAULTS["q_max"])
    q_max = int(q_max_raw) if q_max_raw is not None else DEFAULTS["q_max"]

    edge_mode_val = merged.get("edge_mode", DEFAULTS["edge_mode"]) or DEFAULTS["edge_mode"]

    alignment_top_p_raw = merged.get("alignment_top_p", DEFAULTS["alignment_top_p"])
    alignment_top_p = int(alignment_top_p_raw) if alignment_top_p_raw is not None else DEFAULTS["alignment_top_p"]

    pre_raw = merged.get("prewhiten", DEFAULTS["prewhiten"])
    if isinstance(pre_raw, bool):
        prewhiten_mode = "ff5mom" if pre_raw else "off"
    elif pre_raw is None:
        prewhiten_mode = str(DEFAULTS["prewhiten"]).lower()
    else:
        prewhiten_mode = str(pre_raw).lower()
    valid_pre_modes = {"off", "ff5", "ff5mom"}
    if prewhiten_mode not in valid_pre_modes:
        raise ValueError(
            "prewhiten must be one of {'off', 'ff5', 'ff5mom'}"
        )

    use_factor_raw = merged.get("use_factor_prewhiten", DEFAULTS["use_factor_prewhiten"])
    if isinstance(use_factor_raw, str):
        use_factor_prewhiten = use_factor_raw.strip().lower() not in {"0", "false", "off"}
    elif use_factor_raw is None:
        use_factor_prewhiten = bool(DEFAULTS["use_factor_prewhiten"])
    else:
        use_factor_prewhiten = bool(use_factor_raw)

    group_design_val = str(merged.get("group_design", DEFAULTS["group_design"]) or DEFAULTS["group_design"])
    group_min_count_val = int(
        merged.get("group_min_count", DEFAULTS["group_min_count"])
        if merged.get("group_min_count") is not None
        else DEFAULTS["group_min_count"]
    )
    group_min_replicates_val = int(
        merged.get("group_min_replicates", DEFAULTS["group_min_replicates"])
        if merged.get("group_min_replicates") is not None
        else DEFAULTS["group_min_replicates"]
    )
    min_reps_dow_val = max(
        0,
        int(
            merged.get("min_reps_dow", DEFAULTS["min_reps_dow"])
            if merged.get("min_reps_dow") is not None
            else DEFAULTS["min_reps_dow"]
        ),
    )
    min_reps_vol_val = max(
        0,
        int(
            merged.get("min_reps_vol", DEFAULTS["min_reps_vol"])
            if merged.get("min_reps_vol") is not None
            else DEFAULTS["min_reps_vol"]
        ),
    )
    max_missing_asset_raw = float(merged.get("max_missing_asset", DEFAULTS["max_missing_asset"]))
    max_missing_asset_val = float(min(1.0, max(0.0, max_missing_asset_raw)))
    max_missing_group_row_raw = float(
        merged.get("max_missing_group_row", DEFAULTS["max_missing_group_row"])
    )
    max_missing_group_row_val = float(min(1.0, max(0.0, max_missing_group_row_raw)))
    ewma_halflife_val = float(merged.get("ewma_halflife", DEFAULTS["ewma_halflife"]))
    gate_mode_val = str(merged.get("gate_mode", DEFAULTS["gate_mode"]))
    gate_soft_max_val = (
        int(merged["gate_soft_max"]) if merged.get("gate_soft_max") is not None else None
    )
    gate_delta_calibration_raw = merged.get("gate_delta_calibration")
    gate_delta_calibration_val = (
        Path(gate_delta_calibration_raw) if gate_delta_calibration_raw else None
    )
    gate_delta_frac_min_val = (
        float(merged["gate_delta_frac_min"]) if merged.get("gate_delta_frac_min") is not None else None
    )
    gate_delta_frac_max_val = (
        float(merged["gate_delta_frac_max"]) if merged.get("gate_delta_frac_max") is not None else None
    )
    gate_stability_min_val = (
        float(merged["gate_stability_min"]) if merged.get("gate_stability_min") is not None else float(DEFAULTS["gate_stability_min"])
    )
    gate_alignment_min_val = (
        float(merged["gate_alignment_min"]) if merged.get("gate_alignment_min") is not None else None
    )
    gate_accept_nonisolated_val = bool(merged.get("gate_accept_nonisolated", DEFAULTS["gate_accept_nonisolated"]))

    assets_top_raw = merged.get("assets_top", DEFAULTS["assets_top"])
    assets_top_val = (
        int(assets_top_raw) if assets_top_raw is not None and str(assets_top_raw).strip() != "" else None
    )
    merged["assets_top"] = assets_top_val

    mv_gamma_val = float(merged.get("mv_gamma", DEFAULTS["mv_gamma"]))
    mv_tau_val = float(merged.get("mv_tau", DEFAULTS["mv_tau"]))
    mv_box_lo_val = float(merged.get("mv_box_lo", DEFAULTS["mv_box_lo"]))
    mv_box_hi_val = float(merged.get("mv_box_hi", DEFAULTS["mv_box_hi"]))
    if mv_box_hi_val <= mv_box_lo_val:
        raise ValueError("mv_box_hi must be greater than mv_box_lo.")
    mv_turnover_bps_val = float(merged.get("mv_turnover_bps", DEFAULTS["mv_turnover_bps"]))
    if mv_turnover_bps_val < 0.0:
        raise ValueError("mv_turnover_bps must be non-negative.")
    mv_condition_cap_val = float(merged.get("mv_condition_cap", DEFAULTS["mv_condition_cap"]))
    if mv_condition_cap_val <= 0.0:
        raise ValueError("mv_condition_cap must be positive.")

    overlay_delta_frac_raw = merged.get("overlay_delta_frac")
    overlay_delta_frac_val = (
        float(overlay_delta_frac_raw)
        if overlay_delta_frac_raw is not None and str(overlay_delta_frac_raw).strip() != ""
        else None
    )

    config = EvalConfig(
        returns_csv=Path(returns_csv),
        factors_csv=Path(factors_csv) if factors_csv else None,
        window=int(merged.get("window", DEFAULTS["window"])),
        horizon=int(merged.get("horizon", DEFAULTS["horizon"])),
        out_dir=Path(out_dir) if out_dir else Path(DEFAULTS["out_dir"]),
        assets_top=assets_top_val,
        start=merged.get("start"),
        end=merged.get("end"),
        shrinker=str(merged.get("shrinker", DEFAULTS["shrinker"])),
        seed=int(merged.get("seed", DEFAULTS["seed"])),
        calm_quantile=float(merged.get("calm_quantile", DEFAULTS["calm_quantile"])),
        crisis_quantile=float(merged.get("crisis_quantile", DEFAULTS["crisis_quantile"])),
        vol_ewma_span=int(merged.get("vol_ewma_span", DEFAULTS["vol_ewma_span"])),
        config_path=config_path_obj if (config_path_obj and config_path_obj.exists()) else (
            DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None
        ),
        thresholds_path=thresholds_path_obj if (thresholds_path_obj and thresholds_path_obj.exists()) else (
            DEFAULT_THRESHOLDS_PATH if DEFAULT_THRESHOLDS_PATH.exists() else None
        ),
        echo_config=bool(merged.get("echo_config", DEFAULTS["echo_config"])),
        reason_codes=bool(merged.get("reason_codes", DEFAULTS["reason_codes"])),
        workers=int(merged["workers"]) if merged.get("workers") is not None else None,
        overlay_a_grid=int(merged.get("overlay_a_grid", DEFAULTS["overlay_a_grid"])),
        overlay_seed=int(merged["overlay_seed"]) if merged.get("overlay_seed") is not None else None,
        overlay_delta_frac=overlay_delta_frac_val,
        mv_gamma=mv_gamma_val,
        mv_tau=mv_tau_val,
        mv_box_lo=mv_box_lo_val,
        mv_box_hi=mv_box_hi_val,
        mv_turnover_bps=mv_turnover_bps_val,
        mv_condition_cap=mv_condition_cap_val,
        bootstrap_samples=int(merged.get("bootstrap_samples", DEFAULTS["bootstrap_samples"])),
        require_isolated=require_isolated,
        q_max=q_max,
        edge_mode=str(edge_mode_val),
        angle_min_cos=float(merged["angle_min_cos"]) if merged.get("angle_min_cos") is not None else None,
        alignment_top_p=alignment_top_p,
        cs_drop_top_frac=float(merged["cs_drop_top_frac"]) if merged.get("cs_drop_top_frac") is not None else None,
        prewhiten=prewhiten_mode,
        use_factor_prewhiten=use_factor_prewhiten,
        calm_window_sample=int(merged["calm_window_sample"])
        if merged.get("calm_window_sample") is not None
        else None,
        crisis_window_top_k=int(merged["crisis_window_top_k"])
        if merged.get("crisis_window_top_k") is not None
        else None,
        group_design=group_design_val,
        group_min_count=group_min_count_val,
        group_min_replicates=group_min_replicates_val,
        min_reps_dow=min_reps_dow_val,
        min_reps_vol=min_reps_vol_val,
        max_missing_asset=max_missing_asset_val,
        max_missing_group_row=max_missing_group_row_val,
        ewma_halflife=ewma_halflife_val,
        gate_mode=gate_mode_val,
        gate_soft_max=gate_soft_max_val,
        gate_delta_calibration=gate_delta_calibration_val,
        gate_delta_frac_min=gate_delta_frac_min_val,
        gate_delta_frac_max=gate_delta_frac_max_val,
        gate_stability_min=gate_stability_min_val,
        gate_alignment_min=gate_alignment_min_val,
        gate_accept_nonisolated=gate_accept_nonisolated_val,
    )

    resolved = {
        "returns_csv": str(config.returns_csv),
        "factors_csv": str(config.factors_csv) if config.factors_csv else None,
        "window": config.window,
        "horizon": config.horizon,
        "out_dir": str(config.out_dir),
        "start": config.start,
        "end": config.end,
        "shrinker": config.shrinker,
        "seed": config.seed,
        "calm_quantile": config.calm_quantile,
        "crisis_quantile": config.crisis_quantile,
        "vol_ewma_span": config.vol_ewma_span,
        "workers": config.workers,
        "config_path": str(config.config_path) if config.config_path else None,
        "thresholds_path": str(config.thresholds_path) if config.thresholds_path else None,
        "reason_codes": config.reason_codes,
        "echo_config": config.echo_config,
        "overlay_a_grid": config.overlay_a_grid,
        "overlay_seed": config.overlay_seed,
        "overlay_delta_frac": config.overlay_delta_frac,
        "mv_gamma": config.mv_gamma,
        "mv_tau": config.mv_tau,
        "mv_box_lo": config.mv_box_lo,
        "mv_box_hi": config.mv_box_hi,
        "mv_turnover_bps": config.mv_turnover_bps,
        "mv_condition_cap": config.mv_condition_cap,
        "bootstrap_samples": config.bootstrap_samples,
        "require_isolated": require_isolated,
        "q_max": config.q_max,
        "edge_mode": config.edge_mode,
        "angle_min_cos": config.angle_min_cos,
        "alignment_top_p": config.alignment_top_p,
        "cs_drop_top_frac": config.cs_drop_top_frac,
        "prewhiten": prewhiten_mode,
        "calm_window_sample": config.calm_window_sample,
        "crisis_window_top_k": config.crisis_window_top_k,
        "group_design": config.group_design,
        "group_min_count": config.group_min_count,
        "group_min_replicates": config.group_min_replicates,
        "min_reps_dow": config.min_reps_dow,
        "min_reps_vol": config.min_reps_vol,
        "max_missing_asset": config.max_missing_asset,
        "max_missing_group_row": config.max_missing_group_row,
        "ewma_halflife": config.ewma_halflife,
        "assets_top": assets_top_val,
    }

    return ResolveResult(config=config, resolved=resolved)
