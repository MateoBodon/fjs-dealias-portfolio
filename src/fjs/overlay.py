from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from baselines.covariance import cc_covariance as baseline_cc_covariance, ewma_covariance, quest_covariance, rie_covariance
from finance.ledoit import lw_cov
from finance.shrinkage import oas_covariance
from fjs.dealias import Detection, dealias_search
from fjs.gating import lookup_calibrated_delta, select_top_k

__all__ = [
    "OverlayConfig",
    "detect_spikes",
    "apply_overlay",
]


@dataclass(slots=True)
class OverlayConfig:
    shrinker: str = "rie"
    sample_count: int | None = None
    max_detections: int | None = None
    q_max: int | None = 1
    delta: float = 0.5
    delta_frac: float | None = None
    ewma_halflife: float = 30.0
    eps: float = 0.02
    stability_eta_deg: float = 0.4
    a_grid: int = 120
    require_isolated: bool = True
    off_component_cap: float | None = 0.3
    min_edge_margin: float = 0.0
    edge_mode: str = "tyler"
    seed: int = 0
    cs_drop_top_frac: float | None = None
    gate_mode: str = "strict"
    gate_soft_max: int | None = None
    gate_delta_calibration: str | None = None
    gate_delta_frac_min: float | None = None
    gate_delta_frac_max: float | None = None
    gate_stability_min: float | None = None
    gate_alignment_min: float | None = None
    gate_accept_nonisolated: bool = False
    coarse_candidate: bool = False


def _bracket_status_label(detections: Sequence[Mapping[str, Any]]) -> str:
    tokens = sorted(
        {
            str(det.get("solver_used", "")).strip().lower()
            for det in detections
            if det.get("solver_used")
        }
    )
    if not tokens:
        return "none"
    if tokens == ["grid"]:
        return "grid"
    if tokens == ["rootfind"]:
        return "rootfind"
    if tokens == ["auto"]:
        return "auto"
    if "rootfind" in tokens and "grid" in tokens:
        return "mixed"
    return "+".join(tokens[:3])


def _summarise_pre_gate(detections: Sequence[Mapping[str, Any]], cfg: OverlayConfig) -> dict[str, Any]:
    summary = {
        "raw_outliers_found": int(len(detections)),
        "mp_edge_margin": float("nan"),
        "leakage_offcomp": float("nan"),
        "stability_eta_pass": float("nan"),
        "bracket_status": "none",
    }
    if not detections:
        return summary
    edge_vals = np.asarray([float(det.get("edge_margin", float("nan"))) for det in detections], dtype=np.float64)
    leak_vals = np.asarray(
        [float(det.get("off_component_ratio", float("nan"))) for det in detections], dtype=np.float64
    )
    stab_vals = np.asarray(
        [float(det.get("stability_margin", float("nan"))) for det in detections], dtype=np.float64
    )
    finite_edges = edge_vals[np.isfinite(edge_vals)]
    finite_leak = leak_vals[np.isfinite(leak_vals)]
    finite_stab = stab_vals[np.isfinite(stab_vals)]
    if finite_edges.size:
        summary["mp_edge_margin"] = float(np.mean(finite_edges))
    if finite_leak.size:
        summary["leakage_offcomp"] = float(np.mean(finite_leak))
    threshold = (
        float(cfg.gate_stability_min)
        if cfg.gate_stability_min is not None
        else float(cfg.stability_eta_deg)
    )
    if finite_stab.size:
        summary["stability_eta_pass"] = float(np.mean(finite_stab >= threshold))
    summary["bracket_status"] = _bracket_status_label(detections)
    return summary


def _coarse_candidates(
    observations: NDArray[np.float64],
    cfg: OverlayConfig,
) -> list[Detection]:
    matrix = np.asarray(observations, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= 2 or matrix.shape[1] == 0:
        return []
    finite_mask = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[finite_mask]
    if matrix.shape[0] <= 2 or matrix.shape[1] == 0:
        return []
    try:
        sample_cov = np.cov(matrix, rowvar=False, ddof=1)
    except Exception:
        return []
    if sample_cov.size == 0:
        return []
    try:
        eigvals, eigvecs = np.linalg.eigh(sample_cov)
    except np.linalg.LinAlgError:
        return []
    if eigvals.size == 0:
        return []
    avg_var = float(np.mean(np.clip(eigvals, 1e-12, None)))
    if not np.isfinite(avg_var) or avg_var <= 0.0:
        return []
    n_samples = matrix.shape[0]
    p_dim = matrix.shape[1]
    if n_samples <= 1 or p_dim == 0:
        return []
    aspect = p_dim / float(max(n_samples - 1, 1))
    mp_edge = avg_var * (1.0 + math.sqrt(max(aspect, 0.0))) ** 2
    edge_eps = max(float(cfg.min_edge_margin), 1e-6 * max(abs(mp_edge), 1.0))
    order = np.argsort(eigvals)[::-1]
    candidates: list[Detection] = []
    for idx in order:
        lam_val = float(eigvals[idx])
        edge_margin = lam_val - mp_edge
        if not np.isfinite(edge_margin) or edge_margin <= edge_eps:
            continue
        vec = eigvecs[:, idx].astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm <= 0.0:
            continue
        vec = vec / norm
        det: Detection = Detection(
            mu_hat=lam_val,
            lambda_hat=lam_val,
            a=[1.0],
            components=[lam_val],
            eigvec=vec,
            stability_margin=edge_margin,
            edge_margin=edge_margin,
            buffer_margin=edge_margin,
            t_values=None,
            admissible_root=True,
            solver_used="coarse",
            z_plus=None,
            threshold_main=None,
            z_plus_low=None,
            z_plus_high=None,
            threshold_low=None,
            threshold_high=None,
            stability_margin_low=None,
            stability_margin_high=None,
            sensitivity_low_accept=None,
            sensitivity_high_accept=None,
            target_energy=lam_val,
            target_index=0,
            off_component_ratio=0.0,
            pre_outlier_count=1,
            edge_mode=str(cfg.edge_mode),
            edge_scale=float(mp_edge),
        )
        candidates.append(det)
    if not candidates:
        return []
    cap = len(candidates)
    if cfg.q_max is not None:
        cap = min(cap, int(cfg.q_max))
    if cfg.max_detections is not None:
        cap = min(cap, int(cfg.max_detections))
    return candidates[:cap]


def _baseline_covariance(
    sample_covariance: NDArray[np.float64],
    *,
    observations: NDArray[np.float64] | None,
    config: OverlayConfig,
) -> NDArray[np.float64]:
    shrinker = (config.shrinker or "rie").strip().lower()
    if shrinker == "lw":
        if observations is None:
            raise ValueError("observations required for Ledoit–Wolf shrinkage.")
        return np.asarray(lw_cov(observations), dtype=np.float64)
    if shrinker == "oas":
        if observations is None:
            raise ValueError("observations required for OAS shrinkage.")
        return np.asarray(oas_covariance(observations), dtype=np.float64)
    if shrinker == "cc":
        if observations is None:
            raise ValueError("observations required for constant-correlation shrinkage.")
        return np.asarray(baseline_cc_covariance(observations), dtype=np.float64)
    if shrinker == "sample":
        sigma = np.asarray(sample_covariance, dtype=np.float64)
        return np.asarray(0.5 * (sigma + sigma.T), dtype=np.float64)
    if shrinker == "quest":
        count = config.sample_count
        if count is None:
            if observations is None:
                raise ValueError("observations required to infer sample_count for QuEST.")
            count = int(observations.shape[0])
        return quest_covariance(np.asarray(sample_covariance, dtype=np.float64), sample_count=int(count))
    if shrinker == "ewma":
        if observations is None:
            raise ValueError("observations required for EWMA covariance.")
        return ewma_covariance(np.asarray(observations, dtype=np.float64), halflife=float(config.ewma_halflife))
    # Default RIE-style shrinkage
    count = config.sample_count
    if count is None and observations is not None:
        count = int(observations.shape[0])
    return rie_covariance(
        np.asarray(sample_covariance, dtype=np.float64),
        sample_count=count,
    )


def _resolve_delta_frac(
    cfg: OverlayConfig,
    observations: NDArray[np.float64],
    groups: Sequence[int],
) -> float | None:
    if cfg.delta_frac is not None:
        return float(cfg.delta_frac)
    if cfg.gate_delta_calibration:
        p = int(observations.shape[1]) if observations.size else 0
        q = int(len(np.unique(groups)))
        calibrated = lookup_calibrated_delta(
            cfg.edge_mode,
            p,
            q,
            calibration_path=cfg.gate_delta_calibration,
        )
        return calibrated
    return None


def _gate_detections(
    detections: list[Detection],
    cfg: OverlayConfig,
    soft_cap: int | None,
    delta_frac_used: float | None,
) -> tuple[list[Detection], list[Detection]]:
    if not detections:
        return [], []

    mode = (cfg.gate_mode or "strict").lower()

    base: list[Detection] = []
    accepted: list[Detection] = []
    rejected: list[Detection] = []
    delta_frac_min = cfg.gate_delta_frac_min
    delta_frac_max = cfg.gate_delta_frac_max
    stability_min = cfg.gate_stability_min if cfg.gate_stability_min is not None else cfg.stability_eta_deg
    alignment_min = cfg.gate_alignment_min if cfg.gate_alignment_min is not None else 0.0

    for det in detections:
        if not bool(det.get("admissible_root", True)):
            rejected.append(det)
            continue
        edge_margin = float(det.get("edge_margin", float("-inf")))
        if edge_margin < cfg.min_edge_margin:
            rejected.append(det)
            continue
        pre_count = det.get("pre_outlier_count")
        if (
            pre_count is not None
            and not cfg.gate_accept_nonisolated
            and int(pre_count) != 1
        ):
            rejected.append(det)
            continue
        base.append(det)

    if mode == "soft":
        limit = soft_cap if soft_cap is not None else cfg.q_max
        selected, discarded = select_top_k(base, int(limit) if limit is not None else len(base))
        rejected.extend(discarded)
        return selected, rejected

    for det in base:
        stability = float(det.get("stability_margin", 0.0))
        alignment = float(det.get("alignment_cos", 1.0))
        delta_used = det.get("delta_frac")
        if delta_used is None or (isinstance(delta_used, float) and math.isnan(delta_used)):
            delta_used = delta_frac_used
        delta_used = float(delta_used) if delta_used is not None else float("nan")

        if stability < stability_min:
            rejected.append(det)
            continue
        if alignment < alignment_min:
            rejected.append(det)
            continue
        if delta_frac_min is not None and np.isfinite(delta_used) and delta_used < float(delta_frac_min):
            rejected.append(det)
            continue
        if delta_frac_max is not None and np.isfinite(delta_used) and delta_used > float(delta_frac_max):
            rejected.append(det)
            continue
        accepted.append(det)

    return accepted, rejected


def detect_spikes(
    observations: NDArray[np.float64],
    groups: Sequence[int],
    *,
    config: OverlayConfig | None = None,
    stats: dict | None = None,
) -> list[Detection]:
    cfg = config or OverlayConfig()
    _ = np.random.default_rng(cfg.seed)  # ensure deterministic rng initialisation
    stats_dict = stats if isinstance(stats, dict) else None
    stats_for_search: dict[str, Any] | None
    if stats_dict and {"MS1", "Sigma1_hat"}.issubset(stats_dict.keys()):
        stats_for_search = stats_dict
    else:
        stats_for_search = None
    resolved_delta_frac = _resolve_delta_frac(cfg, observations, groups)
    delta_for_search = (
        float(resolved_delta_frac)
        if resolved_delta_frac is not None
        else float(cfg.delta_frac) if cfg.delta_frac is not None else None
    )

    detections = dealias_search(
        np.asarray(observations, dtype=np.float64),
        np.asarray(groups, dtype=np.intp),
        target_r=0,
        delta=float(cfg.delta),
        delta_frac=delta_for_search,
        eps=float(cfg.eps),
        stability_eta_deg=float(cfg.stability_eta_deg),
        a_grid=int(cfg.a_grid),
        use_tvector=bool(cfg.require_isolated),
        off_component_leak_cap=cfg.off_component_cap,
        edge_mode=str(cfg.edge_mode),
        cs_drop_top_frac=cfg.cs_drop_top_frac,
        stats=stats_for_search,
    )
    if resolved_delta_frac is not None:
        for det in detections:
            det["delta_frac"] = resolved_delta_frac

    coarse_candidates: list[Detection] = []
    if cfg.coarse_candidate and not detections:
        coarse_candidates = _coarse_candidates(observations, cfg)
        if coarse_candidates:
            detections = list(coarse_candidates)
    pre_gate_summary = _summarise_pre_gate(detections, cfg)
    if coarse_candidates:
        pre_gate_summary["coarse_candidates"] = len(coarse_candidates)
    if stats_dict is not None:
        stats_dict.setdefault("pre_gate", {}).update(pre_gate_summary)
    soft_cap = cfg.gate_soft_max if (cfg.gate_mode or "strict").lower() == "soft" else None
    kept, rejected = _gate_detections(detections, cfg, soft_cap, resolved_delta_frac)

    kept.sort(key=lambda det: float(det.get("edge_margin") or det["mu_hat"]), reverse=True)
    limit_q = cfg.q_max if cfg.q_max is not None else len(kept)
    limit_m = cfg.max_detections if cfg.max_detections is not None else len(kept)
    cap = min(limit_q, limit_m)
    if kept and cap < len(kept):
        kept = kept[: int(cap)]

    if stats_dict is not None:
        gating_info = stats_dict.setdefault("gating", {})
        gating_info.update(
            {
                "mode": (cfg.gate_mode or "strict"),
                "initial": len(detections),
                "accepted": len(kept),
                "rejected": len(rejected),
                "soft_cap": int(soft_cap) if soft_cap is not None else None,
                "delta_frac_used": resolved_delta_frac,
            }
        )
    return kept


def apply_overlay(
    sample_covariance: NDArray[np.float64],
    detections: Iterable[Detection],
    *,
    observations: NDArray[np.float64] | None = None,
    config: OverlayConfig | None = None,
    baseline_covariance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    cfg = config or OverlayConfig()
    if baseline_covariance is None:
        base = _baseline_covariance(sample_covariance, observations=observations, config=cfg)
    else:
        base = np.asarray(baseline_covariance, dtype=np.float64)
    overlay = np.asarray(base, dtype=np.float64)
    overlay = 0.5 * (overlay + overlay.T)

    max_use = cfg.max_detections if cfg.max_detections is not None else cfg.q_max
    applied = 0
    for det in detections:
        if max_use is not None and applied >= int(max_use):
            break
        vec = np.asarray(det["eigvec"], dtype=np.float64).reshape(-1, 1)
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            continue
        vec /= norm
        mu = float(det["mu_hat"])
        if not np.isfinite(mu) or mu <= 0.0:
            continue
        current = float((vec.T @ overlay @ vec)[0, 0])
        overlay = overlay + (mu - current) * (vec @ vec.T)
        applied += 1

    overlay = np.asarray(0.5 * (overlay + overlay.T), dtype=np.float64)
    try:
        eigvals = np.linalg.eigvalsh(overlay)
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.eye(overlay.shape[0], dtype=np.float64)
        overlay = np.asarray(0.5 * (overlay + overlay.T), dtype=np.float64) + ridge
        try:
            eigvals = np.linalg.eigvalsh(overlay)
        except np.linalg.LinAlgError:
            return np.asarray(base, dtype=np.float64)
    min_eig = float(eigvals.min(initial=0.0)) if eigvals.size else 0.0
    if min_eig < 0.0:
        overlay = overlay + (-min_eig + 1e-8) * np.eye(overlay.shape[0], dtype=np.float64)
    return overlay
