"""Rule controller; algorithm logic preserved from the research implementation."""
from __future__ import annotations
import hashlib
import json
import math
import time
from dataclasses import asdict
from agents.mock_tool_agent import MockToolAgent
from communication.latency_model import CV2XLatencyModel
from simulation.runner import run_method


METHOD = "fd_emc_no_llm"


LABEL = "FD-EMC w/o LLM"


HOTSPOT_FIELDS = ("hotspot_id", "potential_cache_gain", "covered_vehicle_count")


FIT_FIELDS = (
    "estimated_gain", "acr_update_mrsu_hit", "acr_update_frsu_hit",
    "keep_mrsu_hit", "keep_frsu_hit",
)


def compact_context(context):
    """Project only pre-action summaries; do not inspect evaluation requests."""
    candidates = context["candidate_hotspots"]
    if not candidates:
        raise ValueError("The rule controller requires at least one candidate hotspot.")
    fit = context["cache_fit_analysis"]
    compact = {
        "candidate_hotspots": [{key: item[key] for key in HOTSPOT_FIELDS} for item in candidates],
        "cache_fit_analysis": {key: fit[key] for key in FIT_FIELDS},
    }
    for row in compact["candidate_hotspots"] + [compact["cache_fit_analysis"]]:
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("Non-finite controller input.")
    return compact


class RuleToolController(MockToolAgent):
    """Fixed existing greedy/threshold rule, not optimized against test outcomes."""

    def __init__(self, update_gain_threshold=0.015):
        super().__init__(update_gain_threshold=update_gain_threshold)
        self.records = []

    def decide_tools(self, context):
        compact = compact_context(context)
        decision = super().decide_tools(compact)
        self.records.append({
            "compact_input": compact,
            "available_context_keys": sorted(context),
            "context_sha256": hashlib.sha256(
                json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "decision": decision.to_dict(),
        })
        return decision


def run_controller(config, latency_config, threshold, verbose):
    controller = RuleToolController(threshold)
    start = time.perf_counter()
    result = run_method(config, "tool_agent", agent=controller, verbose=verbose,
                        latency_model=CV2XLatencyModel(latency_config))
    result["elapsed_seconds"] = time.perf_counter() - start
    result["config"] = asdict(config)
    result["controller_records"] = controller.records
    summary = result["summary"]
    summary.update(
        method=METHOD, method_label=LABEL, seed=config.seed,
        rsu_cache_capacity=config.mrsu_cache_capacity,
        mrsu_cache_capacity=config.mrsu_cache_capacity, frsu_cache_capacity=config.frsu_cache_capacity,
        controller="deterministic_existing_mock_tool_rule", uses_miss_feedback=True,
        llm_api_calls=0, method_note="Shared FD-EMC execution with a compact deterministic high-level rule.",
    )
    # The shared executor reports these as llm_* even when the agent is a rule.
    for log in result["round_logs"]:
        log["uses_miss_feedback"] = True
        log["controller_type"] = "deterministic_rule"
        log["predecision_context_chars"] = log.pop("llm_context_chars", None)
        log["predecision_context_schema"] = log.pop("llm_context_schema", None)
    result["valid_controller_run"] = (
        len(controller.records) == len(result["round_logs"])
        and all(not log.get("tool_decision_error") for log in result["round_logs"])
    )
    return result

