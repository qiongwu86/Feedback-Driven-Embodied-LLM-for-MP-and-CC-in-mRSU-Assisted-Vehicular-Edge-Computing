"""Descriptive run-level statistics; windows are not independent replicates."""
import math
import statistics


def summarize(values):
    values = [float(v) for v in values if v is not None]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Finite run-level values are required.")
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = None
    if n > 1:
        from scipy.stats import t
        half = float(t.ppf(.975, n-1))*sd/math.sqrt(n)
    mean = statistics.mean(values)
    return {"n": n, "mean": mean, "sd": sd, "ci95_half": half,
            "ci95_low": None if half is None else mean-half,
            "ci95_high": None if half is None else mean+half}


def metric_value(record, metric):
    summary = record["summary"]
    if metric == "ACTD":
        return float(summary["average_delay_ms"])
    if "local_rsu_achr" in summary:
        return 100*float(summary["local_rsu_achr"])
    logs = record.get("round_logs", [])
    if logs:
        hits = sum(l["metrics"]["hit_count"] for l in logs)
        misses = sum(l["metrics"]["not_cached_count"] for l in logs)
        return 100*hits/(hits+misses) if hits+misses else 0.
    raise ValueError("Local ACHR is unavailable; system ACHR is not an interchangeable fallback.")


def round_values(record, metric):
    values = []
    for log in record.get("round_logs", []):
        if metric == "ACTD":
            values.append(float(log["round_delay_ms"]))
        else:
            m = log["metrics"]
            total = m["hit_count"]+m["not_cached_count"]
            values.append(100*m["hit_count"]/total if total else 0.)
    return values
