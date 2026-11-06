"""Timestamped, method-specific outputs; importing this module creates no files."""
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common.paths import ROOT, project_path


def serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(serializable(rows))


def controller_mode(agent):
    if agent is None:
        return "algorithm"
    name = type(agent).__name__.lower()
    if "rule" in name:
        return "rule"
    if "mock" in name or getattr(agent, "mock_mode", False):
        return "mock"
    return "llm"


class RunOutput:
    def __init__(self, method, args):
        self.method, self.args = method, args
        self.records = []
        now = datetime.now(timezone(timedelta(hours=8)))
        parent = project_path(args.output_dir) / method
        parent.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        self.path = parent / stamp
        self.path.mkdir(exist_ok=False)
        self.manifest = {"schema_version": 1, "method": method, "created_at": now.isoformat(),
                         "status": "running", "controller_modes": [], "fallback_count": 0,
                         "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in ("agents", "baselines", "caching", "communication", "mobility", "simulation", "experiments")
                             for p in (ROOT/folder).glob("*.py")},
                         "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"datasets").glob("*.pkl")}}
        write_json(self.path/"config.json", vars(args))
        self.checkpoint()
        print("Results:", self.path)

    def append(self, result, config, agent=None, model=None, **extra):
        summary = result.get("summary", {})
        logs = result.get("round_logs", [])
        fallbacks = sum(any(bool(v) for k, v in log.items() if k.endswith("_error")) for log in logs)
        record = {"method": self.method, "seed": config.seed, "capacity": config.mrsu_cache_capacity,
                  "model": model or (getattr(agent, "model", None) if agent is not None else None),
                  "controller_mode": controller_mode(agent), "fallback_count": fallbacks,
                  "config": asdict(config), **result, **extra}
        record["summary"] = dict(summary)
        if "round_logs" in result and len(logs) != config.decision_rounds:
            raise ValueError("Incomplete simulation: expected decision windows do not match saved logs.")
        self.records.append(record)
        self.checkpoint()

    def checkpoint(self):
        self.manifest["run_count"] = len(self.records)
        self.manifest["controller_modes"] = sorted({r["controller_mode"] for r in self.records})
        self.manifest["fallback_count"] = sum(r["fallback_count"] for r in self.records)
        write_json(self.path/"manifest.json", self.manifest)
        write_json(self.path/"results.json", {"schema_version": 1, "method": self.method, "runs": self.records})
        rows = []
        for r in self.records:
            row = {k: r[k] for k in ("method", "model", "controller_mode", "seed", "capacity", "fallback_count")}
            row["vehicle_num"] = r["config"]["vehicle_num"]
            row.update({k: v for k, v in r["summary"].items() if k not in row and
                        (v is None or isinstance(v, (int, float, str, bool)))})
            rows.append(row)
        write_csv(self.path/"summary.csv", rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.manifest["status"] = "failed" if exc_type else "complete"
        if exc_type:
            self.manifest["error_type"] = exc_type.__name__
        self.checkpoint()
        return False
