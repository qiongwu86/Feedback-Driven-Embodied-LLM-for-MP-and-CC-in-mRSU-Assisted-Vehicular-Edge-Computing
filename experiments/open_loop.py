"""Open loop; algorithm logic preserved from the research implementation."""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List
from agents.llm_tool_agent import DEFAULT_API_KEY_ENV_VARS, ToolDecision, _extract_json_object
from agents.mock_tool_agent import MockToolAgent
from caching.cache_repair import CacheRepair
from caching.cache_update_evaluator import CacheUpdateEvaluator
from communication.latency_model import CV2XLatencyModel, summarize_round_latencies
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, auto_lambda_smooth
from experiments.ablation_runner import METHOD_OPEN_LOOP_LLM
from simulation.config import MRSUSimulationConfig
from simulation.coverage import CoverageModel
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, summarize_metrics


METHOD_LABEL = "Open-loop FD-EMC"


METHOD_NOTE = (
    "Open-loop FD-EMC keeps the FD-EMC tool chain: the LLM chooses selected_hotspot_id, "
    "update_mrsu_cache, and update_frsu_cache; QPPathPlanner plans the mRSU path; "
    "DemandAwareCooperativeCache rebuilds caches when update flags are enabled. "
    "Previous-round miss feedback and cache-fit residual feedback are hidden. "
    "Runtime caches are preserved when an RSU update flag is false, but cache lists are not exposed to the LLM prompt. "
    "Decision-time demand is regenerated independently from MovieLens training history only; "
    "no test-set requests are used before evaluation."
)


class OpenLoopToolAgent:
    """LLM tool agent with a compact no-feedback prompt for Open-loop FD-EMC."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_context_chars: int = 12000,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model_name
        self.max_context_chars = int(max_context_chars)
        self._client = None

    def decide_tools(self, context: Dict[str, Any]) -> ToolDecision:
        prompt = self._build_prompt(context)
        data = self._call_json(prompt)
        return ToolDecision(
            selected_hotspot_id=int(data.get("selected_hotspot_id", 0)),
            lambda_smooth=0.0,
            update_mrsu_cache=bool(data.get("update_mrsu_cache", True)),
            update_frsu_cache=bool(data.get("update_frsu_cache", True)),
            reason=str(data.get("reason", ""))[:200],
        )

    def _call_json(self, prompt: str) -> dict:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )
        return _extract_json_object(response.choices[0].message.content)

    def _compact_context(self, context: Dict[str, Any]) -> str:
        text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n...TRUNCATED..."

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        return (
            "Open-loop FD-EMC ablation. Return JSON only.\n"
            "You are allowed to choose only selected_hotspot_id, update_mrsu_cache, and update_frsu_cache.\n"
            "Do not output cache content IDs, path points, speed, or lambda_smooth.\n"
            "QPPathPlanner will plan the path after the hotspot choice; DACC will rebuild caches if update flags are true.\n"
            "All demand signals in CONTEXT are current-window predictions from MovieLens training history only.\n"
            "No previous-round hit/miss feedback, cache residuals, vehicle-level miss contents, or test-set requests are available.\n"
            "Each decision should be based only on the current CONTEXT.\n"
            'Schema:{"selected_hotspot_id":int,"update_mrsu_cache":bool,"update_frsu_cache":bool}\n'
            f"CONTEXT:{self._compact_context(context)}"
        )


class MockOpenLoopToolAgent:
    model = "mock-open-loop-daecllm"

    def decide_tools(self, context: Dict[str, Any]) -> ToolDecision:
        candidates = context.get("candidate_hotspots", [])
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("potential_cache_gain", 0.0)),
                int(item.get("covered_vehicle_count", 0)),
            ),
            default={},
        )
        return ToolDecision(
            selected_hotspot_id=int(selected.get("hotspot_id", 0)),
            lambda_smooth=0.0,
            update_mrsu_cache=True,
            update_frsu_cache=True,
            reason="mock open-loop selects the highest training-history hotspot and rebuilds both caches",
        )


def run_open_loop_daec_method(
    config: MRSUSimulationConfig,
    agent,
    latency_model: CV2XLatencyModel = None,
    verbose: bool = True,
) -> dict:
    env = MRSUEnvironment(config)
    latency_model = latency_model or CV2XLatencyModel()
    valid_content_ids = list(range(1, config.movie_num + 1))
    repair = CacheRepair(valid_content_ids, env.global_top_contents)
    cache_evaluator = CacheUpdateEvaluator(
        repair,
        env.global_top_contents,
        config,
        global_popularity=env.global_popularity,
        valid_content_ids=valid_content_ids,
    )
    hotspot_generator = CandidateHotspotGenerator(
        road_length=config.road_length,
        grid_step=config.grid_step,
        mrsu_radius=config.mrsu_radius,
        mrsu_cache_capacity=config.mrsu_cache_capacity,
        candidate_count=config.candidate_count,
    )
    planner = QPPathPlanner(
        road_length=config.road_length,
        dt=config.dt,
        horizon=config.planner_horizon,
        v_min=config.mrsu_v_min,
        v_max=config.mrsu_v_max,
        a_min=config.mrsu_a_min,
        a_max=config.mrsu_a_max,
    )

    round_metrics: List[RoundMetrics] = []
    round_logs: List[dict] = []

    for round_index in range(config.decision_rounds):
        reset_open_loop_prediction_state(env)
        window_ticks = env.decision_window_ticks(round_index)
        decision_requests = env.predict_round_requests(round_index, use_feedback=False)
        vehicle_demands = env.vehicle_request_counters(decision_requests)
        candidate_hotspots = [
            hotspot.to_dict()
            for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
        ]
        selected_for_estimate = find_hotspot(candidate_hotspots, best_hotspot_id(candidate_hotspots))
        decision_request_budget = max(1, sum(len(requests) for requests in decision_requests.values()))
        estimate_lambda_smooth = auto_lambda_smooth(
            current_position=env.mrsu.position,
            target_position=float(selected_for_estimate["position"]),
            potential_cache_gain=float(selected_for_estimate.get("potential_cache_gain", 0.0)),
            road_length=config.road_length,
            request_budget=decision_request_budget,
            default_lambda=config.default_lambda_smooth,
        )
        estimate_plan = planner.plan(
            current_position=env.mrsu.position,
            current_speed=env.mrsu.speed,
            target_position=float(selected_for_estimate["position"]),
            lambda_smooth=estimate_lambda_smooth,
        )
        estimated_coverage = env.project_service_window_coverage(estimate_plan, ticks=window_ticks)
        tool_context = build_open_loop_tool_context(
            env=env,
            config=config,
            round_index=round_index,
            decision_requests=decision_requests,
            candidate_hotspots=candidate_hotspots,
            reference_coverage=estimated_coverage,
        )
        llm_context_chars = len(json.dumps(tool_context, ensure_ascii=False, separators=(",", ":")))
        tool_decision_error = None
        try:
            tool_decision = agent.decide_tools(tool_context)
        except Exception as exc:
            tool_decision_error = f"{type(exc).__name__}: {exc}"
            print(f"[{METHOD_LABEL}] LLM decision failed; using mock fallback. {tool_decision_error}")
            tool_decision = MockOpenLoopToolAgent().decide_tools(tool_context)

        selected_hotspot = find_hotspot(candidate_hotspots, tool_decision.selected_hotspot_id)
        lambda_smooth = auto_lambda_smooth(
            current_position=env.mrsu.position,
            target_position=float(selected_hotspot["position"]),
            potential_cache_gain=float(selected_hotspot.get("potential_cache_gain", 0.0)),
            road_length=config.road_length,
            request_budget=decision_request_budget,
            default_lambda=config.default_lambda_smooth,
        )
        tool_decision.lambda_smooth = lambda_smooth
        path_plan = planner.plan(
            current_position=env.mrsu.position,
            current_speed=env.mrsu.speed,
            target_position=float(selected_hotspot["position"]),
            lambda_smooth=lambda_smooth,
        )
        decision_coverage = env.project_service_window_coverage(path_plan, ticks=window_ticks)
        content_features = env.content_features(
            decision_coverage.mrsu_covered,
            decision_coverage.frsu_covered,
            decision_requests,
            config.global_topk_for_prompt,
        )
        fit_summary = open_loop_fit_summary()
        need_mrsu_update = bool(tool_decision.update_mrsu_cache) or not env.mrsu_cache
        need_frsu_update = bool(tool_decision.update_frsu_cache) or not env.frsu_cache
        mrsu_cache, frsu_cache, cache_tool_details = cache_evaluator.update_with_acr_tool(
            vehicle_requests=decision_requests,
            coverage=decision_coverage,
            selected_hotspot=selected_hotspot,
            content_features=content_features,
            fit_summary=fit_summary,
            update_mrsu=need_mrsu_update,
            update_frsu=need_frsu_update,
            current_mrsu_cache=env.mrsu_cache,
            current_frsu_cache=env.frsu_cache,
        )
        cache_tool_details.update(
            {
                "open_loop_no_feedback": True,
                "previous_runtime_cache_hidden_from_llm": True,
                "runtime_cache_preserved_when_update_false": True,
                "fit_summary_feedback_hidden": True,
                "current_cache_residuals_hidden": True,
                "prediction_source": "training_history_only_independent_per_decision",
            }
        )
        cache_update_used = bool(need_mrsu_update or need_frsu_update)

        env.set_cache(mrsu_cache, frsu_cache)
        coverage = env.execute_service_window(path_plan, ticks=window_ticks)
        vehicle_requests = env.sample_round_requests(round_index)
        metrics = env.evaluate(vehicle_requests, coverage, mrsu_cache, frsu_cache)
        latency = latency_model.evaluate_round(
            vehicle_requests=vehicle_requests,
            vehicle_positions=env.mobility.positions(),
            coverage=coverage,
            mrsu_position=env.mrsu.position,
            frsu_position=env.frsu.position,
            road_length=config.road_length,
            mrsu_radius=config.mrsu_radius,
            frsu_radius=config.frsu_radius,
            mrsu_cache=mrsu_cache,
            frsu_cache=frsu_cache,
        ).to_dict()
        round_metrics.append(metrics)
        round_logs.append(
            {
                "round": int(round_index),
                "physical_tick_start": int(round_index * config.decision_interval),
                "physical_tick_end": int(round_index * config.decision_interval + window_ticks - 1),
                "decision_interval_ticks": int(window_ticks),
                "chr": float(metrics.chr),
                "local_rsu_chr": float(metrics.local_rsu_chr),
                "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
                "latency": latency,
                "request_count": int(metrics.request_count),
                "decision_request_count": int(sum(len(requests) for requests in decision_requests.values())),
                "evaluation_request_count": int(metrics.request_count),
                "decision_request_source": "training_history_only_open_loop_independent",
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "uses_miss_feedback": False,
                "feedback_visible_to_llm": False,
                "feedback_used_by_dacc": False,
                "prediction_state_reset_each_decision": True,
                "test_requests_hidden_until_evaluation": True,
                "mrsu_position": float(env.mrsu.position),
                "mrsu_speed": float(env.mrsu.speed),
                "mrsu_covered": [int(x) for x in coverage.mrsu_covered],
                "frsu_covered": [int(x) for x in coverage.frsu_covered],
                "decision_mrsu_covered": [int(x) for x in decision_coverage.mrsu_covered],
                "decision_frsu_covered": [int(x) for x in decision_coverage.frsu_covered],
                "selected_hotspot": selected_hotspot,
                "lambda_smooth": float(lambda_smooth),
                "lambda_smooth_source": "system_auto_rule",
                "path_plan_status": path_plan.status if path_plan else None,
                "path_plan_solver": path_plan.solver if path_plan else None,
                "tool_decision": tool_decision.to_dict(),
                "tool_decision_error": tool_decision_error,
                "llm_context_chars": int(llm_context_chars),
                "llm_context_schema": "open_loop_daec_tool_decision_v1",
                "cache_fit_analysis": None,
                "cache_tool_details": cache_tool_details,
                "cache_update_used": cache_update_used,
                "mrsu_cache": [int(x) for x in mrsu_cache],
                "frsu_cache": [int(x) for x in frsu_cache],
                "metrics": metrics.to_dict(),
            }
        )
        if verbose:
            print(
                f"[{METHOD_LABEL}] round={round_index:02d} "
                f"LocalCHR={metrics.local_rsu_chr:.4f} "
                f"mRSU_hit={metrics.mrsu_hit_count} "
                f"fRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count} "
                f"Delay={latency.get('average_delay_ms', 0.0):.2f}ms "
                f"cache_update={cache_update_used}"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(summarize_round_latencies(log.get("latency") for log in round_logs))
    summary.update(
        {
            "method": METHOD_OPEN_LOOP_LLM,
            "method_label": METHOD_LABEL,
            "method_note": METHOD_NOTE,
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "round_chr": [float(item.chr) for item in round_metrics],
            "round_local_rsu_chr": [float(item.local_rsu_chr) for item in round_metrics],
            "round_delay_ms": [float(log.get("latency", {}).get("average_delay_ms", 0.0)) for log in round_logs],
            "cache_update_count": sum(1 for item in round_logs if item.get("cache_update_used")),
            "latency_model": latency_model.config.to_dict(),
            "uses_miss_feedback": False,
            "feedback_visible_to_llm": False,
            "feedback_used_by_dacc": False,
            "prediction_state_reset_each_decision": True,
            "prediction_source": "training_history_only",
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def build_open_loop_tool_context(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    round_index: int,
    decision_requests: Dict[int, List[int]],
    candidate_hotspots: List[dict],
    reference_coverage,
) -> dict:
    request_counts = {int(vid): int(len(reqs)) for vid, reqs in decision_requests.items()}
    vehicle_rows = [
        {
            "vehicle_id": int(vehicle.vehicle_id),
            "position": round(float(vehicle.position), 2),
            "speed": round(float(vehicle.speed), 2),
            "predicted_request_count": int(request_counts.get(int(vehicle.vehicle_id), 0)),
        }
        for vehicle in env.mobility.vehicles
    ]
    top_request_vehicles = sorted(
        vehicle_rows,
        key=lambda item: (int(item["predicted_request_count"]), -int(item["vehicle_id"])),
        reverse=True,
    )[:10]
    return {
        "context_schema": "open_loop_daec_tool_decision_v1",
        "round": int(round_index),
        "decision_contract": {
            "llm_outputs": ["selected_hotspot_id", "update_mrsu_cache", "update_frsu_cache"],
            "path_planner": "QPPathPlanner_after_hotspot_selection",
            "cache_tool": "DemandAwareCooperativeCache_after_update_flags",
            "cache_content_ids": "not_visible_to_llm_and_selected_by_DACC",
            "lambda_smooth": "system_auto_rule",
            "real_requests": "hidden_until_evaluation",
            "feedback": "disabled",
        },
        "system_state": {
            "mrsu": {
                "position": round(float(env.mrsu.position), 2),
                "speed": round(float(env.mrsu.speed), 2),
                "coverage_radius": float(config.mrsu_radius),
            },
            "frsu": {
                "position": round(float(env.frsu.position), 2),
                "coverage_radius": float(config.frsu_radius),
                "full_coverage": bool(config.frsu_full_coverage),
            },
            "cache_capacity": {
                "mrsu": int(config.mrsu_cache_capacity),
                "frsu": int(config.frsu_cache_capacity),
            },
            "motion_constraints": {
                "road_topology": "circular_one_way",
                "road_length": float(config.road_length),
                "dt": float(config.dt),
                "decision_interval_ticks": int(config.decision_interval),
                "v_min": float(config.mrsu_v_min),
                "v_max": float(config.mrsu_v_max),
                "a_min": float(config.mrsu_a_min),
                "a_max": float(config.mrsu_a_max),
                "movement_distance_to_hotspot": "forward circular distance only",
                "coverage_distance": "shortest circular distance",
                "service_window": "vehicle serviceable if covered at least once during the decision window",
            },
        },
        "request_signal": {
            "source": "MovieLens training history only",
            "uses_previous_feedback": False,
            "uses_test_requests": False,
            "prediction_state_reset_each_decision": True,
            "total_predicted_requests": int(sum(request_counts.values())),
            "active_vehicle_count": int(sum(1 for count in request_counts.values() if count > 0)),
            "top_request_vehicles": top_request_vehicles,
        },
        "candidate_hotspots": [
            compact_open_loop_hotspot(env, config, hotspot)
            for hotspot in candidate_hotspots
        ],
        "reference_coverage": {
            "basis": "projected_service_window_for_current_best_training_history_hotspot",
            "mrsu_covered_vehicle_count": len(reference_coverage.mrsu_covered),
            "frsu_covered_vehicle_count": len(reference_coverage.frsu_covered),
            "overlap_vehicle_count": len(reference_coverage.overlap),
        },
    }


def compact_open_loop_hotspot(env: MRSUEnvironment, config: MRSUSimulationConfig, hotspot: dict) -> dict:
    position = float(hotspot.get("position", 0.0))
    return {
        "hotspot_id": int(hotspot.get("hotspot_id", 0)),
        "position": round(position, 2),
        "forward_distance_to_mrsu": round(
            CoverageModel.forward_distance(float(env.mrsu.position), position, float(config.road_length)),
            2,
        ),
        "circular_coverage_distance_to_mrsu": round(
            CoverageModel.circular_distance(float(env.mrsu.position), position, float(config.road_length)),
            2,
        ),
        "covered_vehicle_count": int(hotspot.get("covered_vehicle_count", 0)),
        "potential_cache_gain": round(float(hotspot.get("potential_cache_gain", 0.0)), 4),
    }


def reset_open_loop_prediction_state(env: MRSUEnvironment) -> None:
    env.vehicle_prediction_states = {
        int(vehicle_id): Counter(counter)
        for vehicle_id, counter in env.vehicle_prediction_profiles.items()
    }
    env.predicted_request_count_state = env._initial_request_count_state(env.vehicle_prediction_profiles)
    env.last_round_missed_counter = Counter()
    env.last_round_vehicle_missed_counters = defaultdict(Counter)
    env.last_metrics = RoundMetrics(0, 0, 0.0, 0, 0, 0, 0, 0)


def open_loop_fit_summary() -> dict:
    return {
        "mrsu_current_cache_estimated_hits": 0,
        "frsu_current_cache_estimated_hits": 0,
        "mrsu_top_missing_contents": [],
        "frsu_top_missing_contents": [],
        "mrsu_covered_vehicle_feedback_contents": [],
        "frsu_covered_vehicle_feedback_contents": [],
    }


def find_hotspot(candidate_hotspots: List[dict], hotspot_id: int) -> dict:
    if not candidate_hotspots:
        return {
            "hotspot_id": 0,
            "position": 0.0,
            "covered_vehicle_ids": [],
            "covered_vehicle_count": 0,
            "potential_cache_gain": 0.0,
            "dominant_contents": [],
            "demand_summary": {},
        }
    for hotspot in candidate_hotspots:
        if int(hotspot.get("hotspot_id", -1)) == int(hotspot_id):
            return hotspot
    return candidate_hotspots[0]


def best_hotspot_id(candidate_hotspots: List[dict]) -> int:
    best = max(
        candidate_hotspots or [],
        key=lambda item: (
            float(item.get("potential_cache_gain", 0.0)),
            int(item.get("covered_vehicle_count", 0)),
        ),
        default={"hotspot_id": 0},
    )
    return int(best.get("hotspot_id", 0))


def build_open_loop_agent(args: argparse.Namespace):
    api_key = resolve_api_key(args.api_key_env)
    if args.agent == "mock":
        return MockOpenLoopToolAgent()
    if args.agent == "auto" and not api_key:
        print("No API key is set; using MockOpenLoopToolAgent.")
        return MockOpenLoopToolAgent()
    if args.agent == "llm" and not api_key:
        print("No API key is set; using MockOpenLoopToolAgent.")
        return MockOpenLoopToolAgent()
    return OpenLoopToolAgent(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_context_chars=args.max_context_chars,
    )


def describe_agent(agent) -> str:
    if isinstance(agent, OpenLoopToolAgent):
        return f"open_loop_tool_llm ({agent.model})"
    if isinstance(agent, MockOpenLoopToolAgent):
        return "mock_open_loop_tool_agent"
    if isinstance(agent, MockToolAgent):
        return "mock_tool_agent"
    return type(agent).__name__


def set_open_loop_summary_fields(summary: dict) -> None:
    summary["method_label"] = METHOD_LABEL
    summary["method_note"] = METHOD_NOTE
    summary["uses_miss_feedback"] = False
    summary["feedback_visible_to_llm"] = False
    summary["feedback_used_by_dacc"] = False
    summary["prediction_state_reset_each_decision"] = True
    summary["prediction_source"] = "training_history_only"


def resolve_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in DEFAULT_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""

