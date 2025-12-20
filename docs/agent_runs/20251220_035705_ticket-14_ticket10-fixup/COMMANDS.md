git status --porcelain=v1
git rev-parse HEAD
git log -1 --oneline
date -u +"%Y%m%d_%H%M%S"
git checkout -b codex/ticket-14-ticket10-fixup
mkdir -p docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup
cat <<'EOF' > docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/PROMPT.md
touch docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/RESULTS.md docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/TESTS.md docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/META.md
ls calibration
cat calibration/nested_edge_delta_thresholds.json
rg "lookup_calibrated_delta"
sed -n '1,260p' src/fjs/gating.py
sed -n '1,260p' tests/test_calibration_lookup.py
sed -n '1,260p' experiments/synthetic/nested_killtest.py
rg "design: nested" experiments
rg "nested" experiments/equity_panel
cat experiments/equity_panel/config.nested.smoke.yaml
rg "argparse" -n experiments/equity_panel/run.py
sed -n '3765,3905p' experiments/equity_panel/run.py
sed -n '3905,4100p' experiments/equity_panel/run.py
rg "max_windows" -n
rg "max_windows" experiments/equity_panel/run.py
rg "start_indices" -n experiments/equity_panel/run.py
rg "window_idx" -n experiments/equity_panel/run.py
sed -n '1400,1700p' experiments/equity_panel/run.py
sed -n '2500,2900p' experiments/equity_panel/run.py
rg "DEFAULT_CONFIG" -n experiments/equity_panel/run.py
sed -n '1,170p' experiments/equity_panel/run.py
sed -n '740,830p' experiments/equity_panel/run.py
rg "calibration_design" -n experiments/synthetic/nested_killtest.py
sed -n '340,430p' experiments/synthetic/nested_killtest.py
cat experiments/synthetic/config.nested.killtest.yaml
apply_patch src/fjs/gating.py
python - <<'PY'
... (failed: python not found; replaced with python3)
PY
cat <<'PY' > /tmp/update_nested_cal.py
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

cal_path = Path("calibration/nested_edge_delta_thresholds.json")
payload = json.loads(cal_path.read_text())

config_path = Path("experiments/synthetic/config.nested.killtest.yaml")
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

raw_ts = payload.get("generated_at") or payload.get("timestamp_utc")
if raw_ts:
    ts_clean = raw_ts.replace("+00:00", "Z")
else:
    ts_clean = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

selection = payload.get("selection", {})
target_fpr = float(payload.get("alpha", selection.get("target_fpr", 0.0)) or 0.0)
null_rate = float(selection.get("null_rate", 0.0) or 0.0)
null_ci_low = selection.get("null_ci_low")
null_ci_high = selection.get("null_ci_high")
null_trials = int(selection.get("null_trials", selection.get("trials_null", 0) or 0))
power_mod = selection.get("power_moderate")
power_mod_ci_hi = selection.get("power_moderate_ci_high")
power_mod_trials = selection.get("power_moderate_trials")
power_strong = selection.get("power_strong")
power_strong_ci_high = selection.get("power_strong_ci_high")
power_strong_trials = selection.get("power_strong_trials")
config_meta = payload.get("config", {})
years = int(config_meta.get("years", 0) or 0)
replicates = int(config_meta.get("replicates", 0) or 0)
edge_modes_cfg = config_meta.get("edge_modes") or []

if power_strong_trials is None:
    trials_per = config_meta.get("trials_per_scenario") or {}
    try:
        power_strong_trials = int(trials_per.get("strong", 0) or 0)
    except Exception:
        power_strong_trials = None

thresholds_raw = payload.get("thresholds", {}) or {}
aug_thresholds: dict[str, dict[str, dict[str, object]]] = {}
operating_points: list[dict[str, object]] = []
seen_ops: set[tuple[str, int, int]] = set()
for edge_mode, combos in thresholds_raw.items():
    if not isinstance(combos, dict):
        continue
    mode_key = str(edge_mode).lower()
    aug_mode: dict[str, dict[str, object]] = {}
    for key, entry in combos.items():
        if not isinstance(entry, dict):
            entry = {"delta_frac": entry}
        entry = dict(entry)
        try:
            p_str, t_str = key.split("x")
            p_val = int(p_str)
            t_val = int(t_str)
        except Exception:
            continue
        weeks_common = None
        if replicates > 0 and years > 0 and t_val > 0:
            weeks_common = int(round(t_val / float(replicates * years)))
        entry.setdefault("delta_frac", entry.get("delta"))
        entry.setdefault("target_fpr", target_fpr)
        entry["edge_mode"] = mode_key
        entry["p_assets"] = p_val
        entry["n_obs"] = t_val
        if weeks_common:
            entry["weeks_common"] = weeks_common
        if years:
            entry["years"] = years
        if replicates:
            entry["replicates"] = replicates
        op_id = (mode_key, p_val, t_val)
        if op_id not in seen_ops:
            seen_ops.add(op_id)
            operating_points.append(
                {
                    "edge_mode": mode_key,
                    "p": p_val,
                    "t": t_val,
                    "weeks": weeks_common,
                    "years": years or None,
                    "replicates": replicates or None,
                }
            )
        aug_mode[key] = entry
    if aug_mode:
        aug_thresholds[mode_key] = aug_mode

payload["thresholds"] = aug_thresholds
payload["design_thresholds"] = {"nested": {"thresholds": copy.deepcopy(aug_thresholds)}}
payload["schema_version"] = "1.0"
payload["timestamp_utc"] = ts_clean
payload["target_fpr"] = target_fpr
payload["achieved_fpr"] = null_rate

payload["selection"] = dict(selection) | {
    "target_fpr": target_fpr,
    "null_rate": null_rate,
    "null_ci_low": null_ci_low,
    "null_ci_high": null_ci_high,
    "null_trials": null_trials,
    "ci_method": selection.get("ci_method", "wilson(alpha=0.05)"),
    "power_moderate": power_mod,
    "power_moderate_ci_high": power_mod_ci_hi,
    "power_moderate_trials": power_mod_trials,
    "power_strong": power_strong,
    "power_strong_ci_high": power_strong_ci_high,
    "power_strong_trials": power_strong_trials,
}

payload["metadata"] = {
    "run_name": payload.get("run_name") or selection.get("run_name"),
    "timestamp_utc": ts_clean,
    "git_sha": payload.get("git_sha"),
    "config_path": str(config_path),
    "config_hash": config_hash,
    "config_edge_modes": edge_modes_cfg,
    "target_fpr": target_fpr,
    "achieved_fpr": null_rate,
    "achieved_fpr_ci_low": null_ci_low,
    "achieved_fpr_ci_high": null_ci_high,
    "ci_method": selection.get("ci_method", "wilson(alpha=0.05)"),
    "null_trials": null_trials,
    "power_moderate": power_mod,
    "power_moderate_ci_high": power_mod_ci_hi,
    "power_moderate_trials": power_mod_trials,
    "power_strong": power_strong,
    "power_strong_ci_high": power_strong_ci_high,
    "power_strong_trials": power_strong_trials,
    "operating_points": operating_points,
}

cal_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
python3 /tmp/update_nested_cal.py
rm /tmp/update_nested_cal.py
cat <<'PY' > /tmp/update_nested_cal_v2.py
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import yaml

cal_path = Path("calibration/nested_edge_delta_thresholds.json")
payload = json.loads(cal_path.read_text())

config_path = Path("experiments/synthetic/config.nested.killtest.yaml")
config_yaml = yaml.safe_load(config_path.read_text()) or {}
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

trials_cfg = config_yaml.get("trials_per_scenario") or {}
null_trials_cfg = int(trials_cfg.get("null", 0) or 0)
moderate_trials_cfg = int(trials_cfg.get("moderate", 0) or 0)
strong_trials_cfg = int(trials_cfg.get("strong", 0) or 0)

target_fpr = float(config_yaml.get("target_fpr", payload.get("alpha", 0.0)) or 0.0)
selection = payload.get("selection", {})
null_rate = float(selection.get("null_rate", 0.0) or 0.0)
null_ci_low = selection.get("null_ci_low")
null_ci_high = selection.get("null_ci_high")
null_trials = int(selection.get("null_trials", null_trials_cfg) or null_trials_cfg)
power_mod = selection.get("power_moderate")
power_mod_ci_hi = selection.get("power_moderate_ci_high")
power_mod_trials = int(selection.get("power_moderate_trials", moderate_trials_cfg) or moderate_trials_cfg)
power_strong = selection.get("power_strong")
power_strong_ci_high = selection.get("power_strong_ci_high")
power_strong_trials = int(selection.get("power_strong_trials", strong_trials_cfg) or strong_trials_cfg)

raw_ts = payload.get("generated_at") or payload.get("timestamp_utc")
if raw_ts:
    ts_clean = raw_ts.replace("+00:00", "Z")
else:
    ts_clean = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

years = int(config_yaml.get("years", 0) or 0)
replicates = int(config_yaml.get("replicates", 0) or 0)
edge_modes_cfg = config_yaml.get("edge_modes") or []

thresholds_raw = payload.get("thresholds", {}) or {}
aug_thresholds: dict[str, dict[str, dict[str, object]]] = {}
operating_points: list[dict[str, object]] = []
seen_ops: set[tuple[str, int, int]] = set()
for edge_mode, combos in thresholds_raw.items():
    if not isinstance(combos, dict):
        continue
    mode_key = str(edge_mode).lower()
    aug_mode: dict[str, dict[str, object]] = {}
    for key, entry in combos.items():
        if not isinstance(entry, dict):
            entry = {"delta_frac": entry}
        entry = dict(entry)
        try:
            p_str, t_str = key.split("x")
            p_val = int(p_str)
            t_val = int(t_str)
        except Exception:
            continue
        weeks_common = None
        if replicates > 0 and years > 0 and t_val > 0:
            weeks_common = int(round(t_val / float(replicates * years)))
        entry.setdefault("delta_frac", entry.get("delta"))
        entry.setdefault("target_fpr", target_fpr)
        entry["edge_mode"] = mode_key
        entry["p_assets"] = p_val
        entry["n_obs"] = t_val
        if weeks_common:
            entry["weeks_common"] = weeks_common
        if years:
            entry["years"] = years
        if replicates:
            entry["replicates"] = replicates
        op_id = (mode_key, p_val, t_val)
        if op_id not in seen_ops:
            seen_ops.add(op_id)
            operating_points.append(
                {
                    "edge_mode": mode_key,
                    "p": p_val,
                    "t": t_val,
                    "weeks": weeks_common,
                    "years": years or None,
                    "replicates": replicates or None,
                }
            )
        aug_mode[key] = entry
    if aug_mode:
        aug_thresholds[mode_key] = aug_mode

payload["config"] = config_yaml
payload["thresholds"] = aug_thresholds
payload["design_thresholds"] = {"nested": {"thresholds": copy.deepcopy(aug_thresholds)}}
payload["schema_version"] = "1.0"
payload["timestamp_utc"] = ts_clean
payload["target_fpr"] = target_fpr
payload["achieved_fpr"] = null_rate

payload["selection"] = dict(selection) | {
    "target_fpr": target_fpr,
    "null_rate": null_rate,
    "null_ci_low": null_ci_low,
    "null_ci_high": null_ci_high,
    "null_trials": null_trials,
    "ci_method": selection.get("ci_method", "wilson(alpha=0.05)"),
    "power_moderate": power_mod,
    "power_moderate_ci_high": power_mod_ci_hi,
    "power_moderate_trials": power_mod_trials,
    "power_strong": power_strong,
    "power_strong_ci_high": power_strong_ci_high,
    "power_strong_trials": power_strong_trials,
}

payload["metadata"] = {
    "run_name": payload.get("run_name") or selection.get("run_name"),
    "timestamp_utc": ts_clean,
    "git_sha": payload.get("git_sha"),
    "config_path": str(config_path),
    "config_hash": config_hash,
    "config_edge_modes": edge_modes_cfg,
    "trials_per_scenario": trials_cfg,
    "target_fpr": target_fpr,
    "achieved_fpr": null_rate,
    "achieved_fpr_ci_low": null_ci_low,
    "achieved_fpr_ci_high": null_ci_high,
    "ci_method": selection.get("ci_method", "wilson(alpha=0.05)"),
    "null_trials": null_trials,
    "power_moderate": power_mod,
    "power_moderate_ci_high": power_mod_ci_hi,
    "power_moderate_trials": power_mod_trials,
    "power_strong": power_strong,
    "power_strong_ci_high": power_strong_ci_high,
    "power_strong_trials": power_strong_trials,
    "operating_points": operating_points,
}

cal_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
python3 /tmp/update_nested_cal_v2.py
rm /tmp/update_nested_cal_v2.py
apply_patch experiments/synthetic/nested_killtest.py (DEFAULT_CONFIG updates)
apply_patch tests/test_calibration_lookup.py
python3 - <<'PY' ... (print metadata keys)
apply_patch experiments/equity_panel/run.py (add itertools import)
apply_patch experiments/equity_panel/run.py (DEFAULT_CONFIG max_windows)
apply_patch experiments/equity_panel/run.py (add max_windows_override parameter)
apply_patch experiments/equity_panel/run.py (max_windows override set)
apply_patch experiments/equity_panel/run.py (max_windows limiting)
apply_patch experiments/equity_panel/run.py (add --max-windows CLI flag)
apply_patch experiments/equity_panel/run.py (pass max_windows_override)
cat <<'EOF' > experiments/equity_panel/config.nested.smoke.tiny.yaml ...
apply_patch Makefile (add run:equity_nested_smoke_tiny)
git status --short
apply_patch project_state/CONFIG_REFERENCE.md (max_windows + smoke target)
apply_patch Makefile (exec-mode flag for nested tiny smoke)
make test-fast
pip install -e '.[dev]'
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
. .venv/bin/activate && make test-fast
apply_patch experiments/equity_panel/run.py (_run_single_period call adds max_windows)
apply_patch experiments/equity_panel/run.py (use max_windows param)
. .venv/bin/activate && make run:equity_nested_smoke_tiny
cat <<'EOF' > docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/TESTS.md ...
cat <<'EOF' > docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/RESULTS.md ...
date -u +"%Y-%m-%dT%H:%MZ"
apply_patch PROGRESS.md (add ticket-14 entry)
apply_patch project_state/CURRENT_RESULTS.md (ticket-14 note)
apply_patch project_state/KNOWN_ISSUES.md (update nested issue)
git add src/fjs/gating.py tests/test_calibration_lookup.py calibration/nested_edge_delta_thresholds.json experiments/synthetic/nested_killtest.py
git commit -m 'Harden nested calibration metadata and lookup' ...
cat <<'EOF' > docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/META.md ...
git add Makefile experiments/equity_panel/run.py experiments/equity_panel/config.nested.smoke.tiny.yaml project_state/CONFIG_REFERENCE.md project_state/CURRENT_RESULTS.md project_state/KNOWN_ISSUES.md PROGRESS.md docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup
git commit -m 'Add capped nested smoke path and doc updates' ...
git rev-parse HEAD
apply_patch docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/META.md (set end_sha)
git add docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/COMMANDS.md docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/META.md
make gpt-bundle TICKET=ticket-14 RUN_NAME=20251220_035705_ticket-14_ticket10-fixup
unzip -l docs/gpt_bundles/20251220_052600_ticket-14_20251220_035705_ticket-14_ticket10-fixup.zip > docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/bundle_listing.txt
unzip -p docs/gpt_bundles/20251220_052600_ticket-14_20251220_035705_ticket-14_ticket10-fixup.zip LAST_COMMIT.txt
make gpt-bundle TICKET=ticket-14 RUN_NAME=20251220_035705_ticket-14_ticket10-fixup
unzip -l docs/gpt_bundles/20251220_052833_ticket-14_20251220_035705_ticket-14_ticket10-fixup.zip > docs/agent_runs/20251220_035705_ticket-14_ticket10-fixup/bundle_listing.txt
unzip -p docs/gpt_bundles/20251220_052833_ticket-14_20251220_035705_ticket-14_ticket10-fixup.zip LAST_COMMIT.txt
