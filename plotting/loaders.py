"""Load complete, compatible result batches without launching simulations."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from common.paths import project_path

ALIASES = {"tool_agent": "fd_emc", "open_loop_llm": "open_loop",
           "static_mrsu_llm": "static_mrsu", "fd_emc_no_llm": "no_llm"}


def compatibility(record, vehicle_scan=False):
    excluded = {"seed", "output_dir", "mrsu_cache_capacity", "frsu_cache_capacity"}
    if vehicle_scan:
        excluded.update({"vehicle_num", "user_num"})
    config = {k: v for k, v in record["config"].items() if k not in excluded}
    return json.dumps({"simulation": config, "latency": record["summary"].get("latency_model", {})}, sort_keys=True)


def _legacy_records(data, source):
    records = []
    by_capacity = data.get("config_by_capacity", {})
    def walk(node, path):
        if not isinstance(node, dict):
            return
        if isinstance(node.get("summary"), dict) and isinstance(node.get("round_logs"), list):
            summary = node["summary"]
            candidates = [int(x) for x in path if str(x).isdigit() and str(x) in by_capacity]
            cap = summary.get("rsu_cache_capacity", candidates[0] if candidates else None)
            config = node.get("config") or by_capacity.get(str(cap))
            if not config or cap is None:
                raise ValueError("Legacy result needs explicit configuration/seed metadata: "+str(source))
            method = ALIASES.get(summary.get("method"), summary.get("method"))
            records.append({**node, "method": method, "seed": config["seed"], "capacity": int(cap),
                            "config": config, "model": data.get("model_name", "unspecified"),
                            "controller_mode": data.get("controller_mode", "legacy_unverified")})
            return
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, path+[key])
    walk(data, [])
    return records


def load_records(results_root, sources=None):
    root = project_path(results_root)
    candidates = []
    if sources:
        for value in sources:
            p = project_path(value)
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                candidates.extend(p.rglob("results.json"))
            else:
                raise FileNotFoundError(p)
    elif root.exists():
        candidates = list(root.rglob("results.json"))
    latest, provenance, skipped = {}, {}, []
    for source in sorted(set(candidates)):
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        manifest_path = source.parent/"manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if manifest and manifest.get("status") != "complete":
            skipped.append({"source": str(source), "reason": "incomplete batch"})
            continue
        if data.get("schema_version") == 1 and isinstance(data.get("runs"), list):
            rows = data["runs"]
        elif sources:
            rows = _legacy_records(data, source)
        else:
            skipped.append({"source": str(source), "reason": "unrecognised schema"})
            continue
        grouped = defaultdict(list)
        for row in rows:
            if row.get("phase") == "training" or not row.get("round_logs"):
                continue
            if row["method"] == "llm_overhead":
                continue
            row = dict(row, source=str(source))
            row["method"] = ALIASES.get(row["method"], row["method"])
            required = {"seed", "capacity", "config", "summary", "controller_mode"}
            if not required.issubset(row):
                raise ValueError("Missing run metadata in "+str(source))
            key = (row["method"], row.get("model"), row["controller_mode"],
                   compatibility(row, row["method"] == "vehicle_count"))
            grouped[key].append(row)
        stamp = manifest.get("created_at", "")
        for key, group in grouped.items():
            identities = [(r["seed"], r["capacity"], r["config"]["vehicle_num"]) for r in group]
            if len(identities) != len(set(identities)):
                raise ValueError("Duplicate seed/capacity/vehicle-count runs: "+str(source))
            previous = latest.get(key)
            order = (stamp, source.stat().st_mtime_ns)
            if previous is None or order > previous[0]:
                latest[key] = (order, group)
        provenance[str(source)] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                   "status": manifest.get("status", "legacy_unverified")}
    records = []
    for order, rows in latest.values():
        for row in rows:
            row["batch_order"] = order
            records.append(row)
    return records, provenance, skipped
