"""Runner; algorithm logic preserved from the research implementation."""
import argparse
import json
import os
from typing import List
from agents.llm_tool_agent import DEFAULT_API_KEY_ENV_VARS, LLMToolAgent
from agents.gemini_tool_agent import GEMINI_API_KEY_ENV_VARS, GeminiToolAgent
from agents.gemini_rest_tool_agent import GeminiRestToolAgent
from agents.mock_tool_agent import MockToolAgent
from baselines.no_mrsu_topk import no_mrsu_frsu_topk_cache
from baselines.qp_greedy import qp_greedy_decision
from communication.latency_model import CV2XLatencyModel, summarize_round_latencies
from baselines.uniform_mrsu_topk import uniform_topk_cache
from caching.cache_repair import CacheRepair
from caching.cache_update_evaluator import CacheUpdateEvaluator
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, auto_lambda_smooth
from simulation.config import MRSUSimulationConfig
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, local_rsu_chr_from_counts, summarize_metrics


TOOL_AGENT_METHOD = "tool_agent"


TOOL_AGENT_LABEL = "FD-EMC"


METHOD_LABELS = {
    TOOL_AGENT_METHOD: TOOL_AGENT_LABEL,
}


METHOD_NOTES = {
    TOOL_AGENT_METHOD: "LLM tool-agent main method with hotspot selection and cache-update tool calls.",
}


METHODS = [
    TOOL_AGENT_METHOD,
]


def build_agent(args: argparse.Namespace):
    api_key = _resolve_api_key(args.api_key_env)
    gemini_api_key = _resolve_gemini_api_key(args.api_key_env)
    if args.agent == "gemini":
        if not gemini_api_key:
            print("No Gemini API key is set; falling back to MockToolAgent.")
            return MockToolAgent(update_gain_threshold=args.update_threshold)
        return GeminiToolAgent(
            api_key=gemini_api_key,
            model_name=args.model_name,
            api_key_env=args.api_key_env,
        )
    if args.agent == "gemini-rest":
        if not api_key:
            print("No API key is set; falling back to MockToolAgent.")
            return MockToolAgent(update_gain_threshold=args.update_threshold)
        return GeminiRestToolAgent(
            api_key=api_key,
            base_url=args.base_url,
            model_name=args.model_name,
        )
    if args.agent == "auto":
        if api_key:
            return LLMToolAgent(api_key=api_key, base_url=args.base_url, model_name=args.model_name)
        print("No API key is set; using MockToolAgent.")
        return MockToolAgent(update_gain_threshold=args.update_threshold)
    if args.agent == "llm":
        if not api_key:
            print("No API key is set; falling back to MockToolAgent.")
            return MockToolAgent(update_gain_threshold=args.update_threshold)
        return LLMToolAgent(api_key=api_key, base_url=args.base_url, model_name=args.model_name)
    return MockToolAgent(update_gain_threshold=args.update_threshold)


def _resolve_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in DEFAULT_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def _resolve_gemini_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in GEMINI_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def describe_agent(agent) -> str:
    if isinstance(agent, LLMToolAgent):
        return f"llm_tool_agent ({agent.model})"
    if isinstance(agent, GeminiToolAgent):
        return f"gemini_tool_agent ({agent.model})"
    if isinstance(agent, GeminiRestToolAgent):
        return f"gemini_rest_tool_agent ({agent.model})"
    return "mock_tool_agent"


def run_method(
    config: MRSUSimulationConfig,
    method: str,
    agent=None,
    verbose: bool = True,
    output_dir: str = None,
    cache_update_candidate_limit: int = 30,
    latency_model: CV2XLatencyModel = None,
    timing_observer=None,
) -> dict:
    # Timing callbacks observe boundaries without advancing the simulation clock.
    if timing_observer is not None and method != "tool_agent":
        raise ValueError("Timing observation is supported only for tool_agent.")
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
        window_ticks = env.decision_window_ticks(round_index)
        if timing_observer is not None:
            timing_observer("round_start", round_index=round_index, window_ticks=window_ticks)
        decision_requests = env.predict_round_requests(round_index)
        vehicle_demands = env.vehicle_request_counters(decision_requests)
        candidate_hotspots = [
            hotspot.to_dict()
            for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
        ]
        if timing_observer is not None:
            timing_observer("prediction_ready")

        selected_hotspot = None
        path_plan = None
        tool_decision = None
        cache_fit_analysis = None
        cache_tool_details = None
        cache_update_used = False
        tool_decision_error = None

        if method == "no_mrsu_frsu_topk":
            mrsu_cache, frsu_cache = no_mrsu_frsu_topk_cache(
                env.global_top_contents,
                config.frsu_cache_capacity,
            )
            decision_coverage = env.project_service_window_coverage(
                ticks=window_ticks,
                hold_mrsu_position=True,
            )
            decision_coverage.mrsu_covered = []
            decision_coverage.overlap = []

        elif method == "uniform_topk":
            decision_coverage = env.project_service_window_coverage(ticks=window_ticks)
            mrsu_cache, frsu_cache = uniform_topk_cache(
                env.global_top_contents,
                config.mrsu_cache_capacity,
                config.frsu_cache_capacity,
            )

        elif method == "qp_greedy":
            selected_hotspot, path_plan = qp_greedy_decision(
                planner=planner,
                candidate_hotspots=candidate_hotspots,
                current_position=env.mrsu.position,
                current_speed=env.mrsu.speed,
                lambda_smooth=config.default_lambda_smooth,
            )
            decision_coverage = env.project_service_window_coverage(path_plan, ticks=window_ticks)
            mrsu_cache, frsu_cache = cache_evaluator.greedy_repaired_cache(
                decision_requests,
                decision_coverage.mrsu_covered,
                decision_coverage.frsu_covered,
                selected_hotspot,
            )

        elif method == "tool_agent":
            if agent is None:
                agent = MockToolAgent()
            selected_for_estimate = _find_hotspot(candidate_hotspots, _best_hotspot_id(candidate_hotspots))
            decision_request_budget = max(1, sum(len(requests) for requests in decision_requests.values()))
            estimate_lambda_smooth = auto_lambda_smooth(
                current_position=env.mrsu.position,
                target_position=float(selected_for_estimate["position"]),
                potential_cache_gain=float(selected_for_estimate.get("potential_cache_gain", 0.0)),
                road_length=config.road_length,
                request_budget=decision_request_budget,
                default_lambda=config.default_lambda_smooth,
            )
            if timing_observer is not None:
                timing_observer("estimate_path_start")
            estimate_plan = planner.plan(
                current_position=env.mrsu.position,
                current_speed=env.mrsu.speed,
                target_position=float(selected_for_estimate["position"]),
                lambda_smooth=estimate_lambda_smooth,
            )
            if timing_observer is not None:
                timing_observer("estimate_path_end", solver=estimate_plan.solver, status=estimate_plan.status)
            estimated_coverage = env.project_service_window_coverage(
                estimate_plan,
                ticks=window_ticks,
            )
            estimate_content_features = env.content_features(
                estimated_coverage.mrsu_covered,
                estimated_coverage.frsu_covered,
                decision_requests,
                config.global_topk_for_prompt,
            )
            estimate_fit_summary = env.cache_fit_summary(estimated_coverage, decision_requests)
            if timing_observer is not None:
                timing_observer("estimate_cache_start")
            cache_fit_analysis = cache_evaluator.build_acr_cache_fit_analysis(
                current_mrsu_cache=env.mrsu_cache,
                current_frsu_cache=env.frsu_cache,
                vehicle_requests=decision_requests,
                coverage=estimated_coverage,
                selected_hotspot=selected_for_estimate,
                content_features=estimate_content_features,
                fit_summary=estimate_fit_summary,
            )
            if timing_observer is not None:
                timing_observer("estimate_cache_end")
            tool_context = env.build_tool_decision_context(
                round_index=round_index,
                vehicle_requests=decision_requests,
                candidate_hotspots=candidate_hotspots,
                selected_hotspot=selected_for_estimate,
                coverage=estimated_coverage,
                request_source="predicted_history_signal",
                cache_fit_analysis=cache_fit_analysis,
                fit_summary=estimate_fit_summary,
                content_features=estimate_content_features,
            )
            llm_context_chars = len(json.dumps(tool_context, ensure_ascii=False))
            tool_decision_error = None
            if timing_observer is not None:
                timing_observer("agent_start")
            try:
                tool_decision = agent.decide_tools(tool_context)
            except Exception as exc:
                tool_decision_error = f"{type(exc).__name__}: {exc}"
                print(f"[tool_agent] LLM decision failed; using MockToolAgent fallback. {tool_decision_error}")
                tool_decision = MockToolAgent(update_gain_threshold=0.0).decide_tools(tool_context)
            if timing_observer is not None:
                timing_observer("agent_end", fallback=bool(tool_decision_error))
            selected_hotspot = _find_hotspot(candidate_hotspots, tool_decision.selected_hotspot_id)
            lambda_smooth = auto_lambda_smooth(
                current_position=env.mrsu.position,
                target_position=float(selected_hotspot["position"]),
                potential_cache_gain=float(selected_hotspot.get("potential_cache_gain", 0.0)),
                road_length=config.road_length,
                request_budget=decision_request_budget,
                default_lambda=config.default_lambda_smooth,
            )
            tool_decision.lambda_smooth = lambda_smooth
            if timing_observer is not None:
                timing_observer("path_start")
            path_plan = planner.plan(
                current_position=env.mrsu.position,
                current_speed=env.mrsu.speed,
                target_position=float(selected_hotspot["position"]),
                lambda_smooth=lambda_smooth,
            )
            if timing_observer is not None:
                timing_observer("path_end", solver=path_plan.solver, status=path_plan.status)
            decision_coverage = env.project_service_window_coverage(
                path_plan,
                ticks=window_ticks,
            )

            content_features = env.content_features(
                decision_coverage.mrsu_covered,
                decision_coverage.frsu_covered,
                decision_requests,
                config.global_topk_for_prompt,
            )
            fit_summary = env.cache_fit_summary(decision_coverage, decision_requests)
            need_mrsu_update = tool_decision.update_mrsu_cache or not env.mrsu_cache
            need_frsu_update = tool_decision.update_frsu_cache or not env.frsu_cache
            if timing_observer is not None:
                timing_observer("cache_start")
            if need_mrsu_update or need_frsu_update:
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
                cache_update_used = True
            else:
                mrsu_cache, frsu_cache, cache_tool_details = cache_evaluator.update_with_acr_tool(
                    vehicle_requests=decision_requests,
                    coverage=decision_coverage,
                    selected_hotspot=selected_hotspot,
                    content_features=content_features,
                    fit_summary=fit_summary,
                    update_mrsu=False,
                    update_frsu=False,
                    current_mrsu_cache=env.mrsu_cache,
                    current_frsu_cache=env.frsu_cache,
                )
            if timing_observer is not None:
                timing_observer("cache_end")

        else:
            raise ValueError(f"Unknown method: {method}")

        env.set_cache(mrsu_cache, frsu_cache)
        if timing_observer is not None:
            timing_observer("control_ready")
        if method == "no_mrsu_frsu_topk":
            coverage = env.execute_service_window(
                ticks=window_ticks,
                hold_mrsu_position=True,
            )
            coverage.mrsu_covered = []
            coverage.overlap = []
        else:
            coverage = env.execute_service_window(
                path_plan,
                ticks=window_ticks,
                hold_mrsu_position=(method == "static_mrsu"),
            )
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
                "round": round_index,
                "physical_tick_start": int(round_index * config.decision_interval),
                "physical_tick_end": int(round_index * config.decision_interval + window_ticks - 1),
                "decision_interval_ticks": int(window_ticks),
                "chr": metrics.chr,
                "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
                "latency": latency,
                "request_count": metrics.request_count,
                "decision_request_count": sum(len(requests) for requests in decision_requests.values()),
                "evaluation_request_count": metrics.request_count,
                "decision_request_source": "predicted_history_signal",
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "mrsu_position": env.mrsu.position,
                "mrsu_speed": env.mrsu.speed,
                "mrsu_covered": coverage.mrsu_covered,
                "frsu_covered": coverage.frsu_covered,
                "decision_mrsu_covered": getattr(decision_coverage, "mrsu_covered", []),
                "decision_frsu_covered": getattr(decision_coverage, "frsu_covered", []),
                "selected_hotspot": selected_hotspot,
                "lambda_smooth": tool_decision.lambda_smooth if tool_decision else None,
                "lambda_smooth_source": "system_auto_rule" if tool_decision else None,
                "path_plan_status": path_plan.status if path_plan else None,
                "path_plan_solver": path_plan.solver if path_plan else None,
                "tool_decision": tool_decision.to_dict() if tool_decision else None,
                "tool_decision_error": tool_decision_error,
                "llm_context_chars": llm_context_chars if method == "tool_agent" else None,
                "llm_context_schema": "compact_tool_decision_v1" if method == "tool_agent" else None,
                "cache_fit_analysis": cache_fit_analysis,
                "cache_tool_details": cache_tool_details,
                "cache_update_used": cache_update_used,
                "metrics": metrics.to_dict(),
            }
        )
        if timing_observer is not None:
            timing_observer("round_complete", log=round_logs[-1])
        if verbose:
            update_text = f" cache_update={cache_update_used}" if method == "tool_agent" else ""
            print(
                f"[{method}] round={round_index:02d} "
                f"LocalCHR={metrics.local_rsu_chr:.4f} "
                f"mRSU_hit={metrics.mrsu_hit_count} "
                f"sRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count}"
                f" Delay={latency.get('average_delay_ms', 0.0):.2f}ms"
                f"{update_text}"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(summarize_round_latencies(log.get("latency") for log in round_logs))
    summary.update(
        {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "method_note": METHOD_NOTES.get(method, ""),
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "round_chr": [item.chr for item in round_metrics],
            "round_local_rsu_chr": [
                local_rsu_chr_from_counts(item.hit_count, item.not_cached_count)
                for item in round_metrics
            ],
            "cache_update_count": sum(1 for item in round_logs if item.get("cache_update_used")),
            "latency_model": latency_model.config.to_dict(),
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def _best_hotspot_id(candidate_hotspots: List[dict]) -> int:
    if not candidate_hotspots:
        return 0
    best = max(
        candidate_hotspots,
        key=lambda item: (
            float(item.get("potential_cache_gain", 0.0)),
            int(item.get("covered_vehicle_count", 0)),
        ),
    )
    return int(best.get("hotspot_id", 0))


def _find_hotspot(candidate_hotspots: List[dict], hotspot_id: int) -> dict:
    for hotspot in candidate_hotspots:
        if int(hotspot.get("hotspot_id", -1)) == int(hotspot_id):
            return hotspot
    if candidate_hotspots:
        return candidate_hotspots[0]
    return {
        "hotspot_id": 0,
        "position": 0.0,
        "covered_vehicle_ids": [],
        "covered_vehicle_count": 0,
        "potential_cache_gain": 0.0,
        "dominant_contents": [],
        "demand_summary": {},
    }

