"""Ablation runner; algorithm logic preserved from the research implementation."""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np
from agents.llm_tool_agent import DEFAULT_API_KEY_ENV_VARS, LLMToolAgent, ToolDecision, _extract_json_object
from agents.mock_tool_agent import MockToolAgent
from caching.cache_repair import CacheRepair
from caching.cache_update_evaluator import CacheUpdateEvaluator
from communication.latency_model import CV2XLatencyModel, summarize_round_latencies
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, auto_lambda_smooth
from simulation.config import MRSUSimulationConfig
from simulation.coverage import CoverageModel
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, summarize_metrics


METHOD_DIRECT_LLM = "direct_llm"


METHOD_OPEN_LOOP_LLM = "open_loop_llm"


METHOD_STATIC_LLM = "static_mrsu_llm"


METHODS = (
    METHOD_DIRECT_LLM,
    METHOD_OPEN_LOOP_LLM,
    METHOD_STATIC_LLM,
)


METHOD_LABELS = {
    METHOD_DIRECT_LLM: "Direct LLM",
    METHOD_OPEN_LOOP_LLM: "Open-loop FD-EMC",
    METHOD_STATIC_LLM: "Static mRSU",
}


METHOD_NOTES = {
    METHOD_DIRECT_LLM: (
        "The LLM directly outputs target_position and mRSU/fRSU cache priority rankings from a compact prompt; "
        "the simulator computes a feasible speed toward the target and truncates the rankings to the current cache capacity. "
        "No QP path planner or DemandAwareCooperativeCache tool is used, and tool-like cache-fit evidence is hidden. "
        "Decision-time predicted demand still uses previous-round miss feedback."
    ),
    METHOD_OPEN_LOOP_LLM: (
        "Open-loop FD-EMC keeps the FD-EMC tool chain: the LLM chooses selected_hotspot_id, "
        "update_mrsu_cache, and update_frsu_cache; QPPathPlanner plans the mRSU path; "
        "DemandAwareCooperativeCache rebuilds caches when update flags are enabled. "
        "Previous-round miss feedback and cache-fit residual feedback are hidden. "
        "Runtime caches are preserved when an RSU update flag is false. "
        "Decision-time demand is regenerated independently from MovieLens training history only."
    ),
    METHOD_STATIC_LLM: (
        "The mRSU stays at its initial position, so QP path planning and hotspot tracking are disabled. "
        "The LLM still makes tool-level cache-update decisions, and DemandAwareCooperativeCache generates "
        "the mRSU/fRSU cache lists from predicted demand and previous-round feedback."
    ),
}


@dataclass
class DirectLLMDecision:
    target_position: float
    next_speed: float
    mrsu_cache_rank: List[int]
    frsu_cache_rank: List[int]
    reason: str

    @property
    def mrsu_cache_list(self) -> List[int]:
        return self.mrsu_cache_rank

    @property
    def frsu_cache_list(self) -> List[int]:
        return self.frsu_cache_rank

    def to_dict(self) -> dict:
        return {
            "target_position": float(self.target_position),
            "next_speed": float(self.next_speed),
            "mrsu_cache_rank": [int(x) for x in self.mrsu_cache_rank],
            "frsu_cache_rank": [int(x) for x in self.frsu_cache_rank],
            "mrsu_cache_list": [int(x) for x in self.mrsu_cache_rank],
            "frsu_cache_list": [int(x) for x in self.frsu_cache_rank],
            "reason": self.reason,
        }


class MockDirectLLMAgent:
    """Deterministic direct-output agent used for dry runs when no API key is set."""

    model = "mock-direct-llm"

    def __init__(self, cache_rank_limit: int = 400):
        self.cache_rank_limit = int(cache_rank_limit)

    def decide(self, context: Dict[str, Any]) -> DirectLLMDecision:
        hotspots = context.get("hotspots") or context.get("candidate_hotspots") or []
        best = max(
            hotspots,
            key=lambda item: (
                float(item.get("potential_cache_gain", 0.0)),
                int(item.get("covered_vehicle_count", 0)),
            ),
            default={},
        )
        target_position = float(best.get("position", context.get("mrsu", {}).get("position", 0.0)))
        pool = context.get("candidate_pool") or context.get("cache_rank_candidate_pool") or []
        return DirectLLMDecision(
            target_position=target_position,
            next_speed=speed_toward_target(context, target_position),
            mrsu_cache_rank=limit_unique_ints(pool, self.cache_rank_limit),
            frsu_cache_rank=limit_unique_ints(pool, self.cache_rank_limit),
            reason="mock strict direct decision from compact hotspot and training/global pool",
        )


class DirectLLMAgent:
    """Direct LLM controller that bypasses the embodied tool chain."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_context_chars: int = 24000,
        cache_rank_limit: int = 400,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model_name
        self.max_context_chars = int(max_context_chars)
        self.cache_rank_limit = int(cache_rank_limit)
        self._client = None

    def decide(self, context: Dict[str, Any]) -> DirectLLMDecision:
        prompt = self._build_prompt(context)
        data = self._call_json(prompt)
        target_position = float(data.get("target_position", context.get("mrsu", {}).get("position", 0.0)))
        return DirectLLMDecision(
            target_position=target_position,
            next_speed=speed_toward_target(context, target_position),
            mrsu_cache_rank=limit_unique_ints(_as_int_list(data.get("mrsu_cache_rank", [])), self.cache_rank_limit),
            frsu_cache_rank=limit_unique_ints(_as_int_list(data.get("frsu_cache_rank", [])), self.cache_rank_limit),
            reason="strict direct LLM rank output",
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
        text = json.dumps(context, ensure_ascii=False)
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n...TRUNCATED..."

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        cache_rank_limit = int(context.get("cache_rank_limit", self.cache_rank_limit))
        return (
            "Direct LLM ablation. No tools. Return JSON only.\n"
            "Choose a target_position on the circular one-way road; the simulator computes feasible speed.\n"
            "Use candidate_pool content IDs for cache rankings; each list should be unique.\n"
            f"Return about {cache_rank_limit} IDs per ranking when possible; simulator truncates to current capacity.\n"
            'Schema:{"target_position":number,"mrsu_cache_rank":[int],"frsu_cache_rank":[int]}\n'
            f"CONTEXT:{self._compact_context(context)}"
        )


class OpenLoopToolAgent:
    """Tool-level LLM agent for FD-EMC without feedback."""

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


def build_direct_agent(args: argparse.Namespace):
    api_key = resolve_api_key(args.api_key_env)
    cache_rank_limit = int(getattr(args, "direct_cache_rank_limit", 400))
    if args.agent == "mock":
        return MockDirectLLMAgent(cache_rank_limit=cache_rank_limit)
    if args.agent == "auto" and not api_key:
        print("No API key is set; using MockDirectLLMAgent for Direct LLM.")
        return MockDirectLLMAgent(cache_rank_limit=cache_rank_limit)
    if args.agent == "llm" and not api_key:
        print("No API key is set; using MockDirectLLMAgent for Direct LLM.")
        return MockDirectLLMAgent(cache_rank_limit=cache_rank_limit)
    return DirectLLMAgent(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_context_chars=args.max_context_chars,
        cache_rank_limit=cache_rank_limit,
    )


def build_tool_agent(args: argparse.Namespace):
    api_key = resolve_api_key(args.api_key_env)
    if args.agent == "mock":
        return MockToolAgent(update_gain_threshold=0.0)
    if args.agent == "auto" and not api_key:
        print("No API key is set; using MockToolAgent for static tool-level LLM ablation.")
        return MockToolAgent(update_gain_threshold=0.0)
    if args.agent == "llm" and not api_key:
        print("No API key is set; using MockToolAgent for static tool-level LLM ablation.")
        return MockToolAgent(update_gain_threshold=0.0)
    return LLMToolAgent(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_context_chars=args.max_context_chars,
        api_key_env=args.api_key_env,
    )


def llm_context_schema_for_method(method: str) -> str:
    if method == METHOD_DIRECT_LLM:
        return "strict_direct_llm_compact_v1"
    if method == METHOD_OPEN_LOOP_LLM:
        return "open_loop_daec_tool_decision_v1"
    if method == METHOD_STATIC_LLM:
        return "compact_tool_decision_v1_static_no_path_planning"
    return ""


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


def describe_direct_agent(agent) -> str:
    if isinstance(agent, DirectLLMAgent):
        return f"direct_llm ({agent.model}, rank_limit={agent.cache_rank_limit})"
    if isinstance(agent, MockDirectLLMAgent):
        return f"mock_direct_llm (rank_limit={agent.cache_rank_limit})"
    return "mock_direct_llm"


def describe_open_loop_agent(agent) -> str:
    if isinstance(agent, OpenLoopToolAgent):
        return f"open_loop_tool_llm ({agent.model})"
    if isinstance(agent, MockOpenLoopToolAgent):
        return "mock_open_loop_tool_agent"
    return ""


def describe_tool_agent(agent) -> str:
    if isinstance(agent, LLMToolAgent):
        return f"tool_llm ({agent.model})"
    if isinstance(agent, MockToolAgent):
        return "mock_tool_agent"
    return ""


def run_ablation_method(
    config: MRSUSimulationConfig,
    method: str,
    direct_agent=None,
    open_loop_agent=None,
    tool_agent=None,
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
    direct_cache_rank_limit = int(getattr(direct_agent, "cache_rank_limit", 400) or 400)
    hotspot_generator = CandidateHotspotGenerator(
        road_length=config.road_length,
        grid_step=config.grid_step,
        mrsu_radius=config.mrsu_radius,
        mrsu_cache_capacity=int(config.mrsu_cache_capacity),
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

    fixed_position = float(env.mrsu.position)
    round_metrics: List[RoundMetrics] = []
    round_logs: List[dict] = []

    for round_index in range(config.decision_rounds):
        window_ticks = env.decision_window_ticks(round_index)
        selected_hotspot = None
        motion_details = {}
        path_plan_status = None
        path_plan_solver = None
        path_plan = None
        direct_decision = None
        direct_error = None
        direct_fallback = False
        tool_decision = None
        tool_decision_error = None
        llm_context_chars = None
        cache_fit_analysis = None
        cache_update_used = False
        uses_miss_feedback = True
        decision_request_source = "predicted_history_feedback_signal"

        if method == METHOD_DIRECT_LLM:
            decision_requests = env.predict_round_requests(round_index, use_feedback=True)
            vehicle_demands = env.vehicle_request_counters(decision_requests)
            candidate_hotspots = [
                hotspot.to_dict()
                for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
            ]
            context = build_strict_direct_context(
                env=env,
                config=config,
                round_index=round_index,
                decision_requests=decision_requests,
                candidate_hotspots=candidate_hotspots,
                cache_rank_limit=direct_cache_rank_limit,
            )
            llm_context_chars = int(len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))))
            try:
                direct_decision = direct_agent.decide(context)
            except Exception as exc:
                direct_error = f"{type(exc).__name__}: {exc}"
                direct_fallback = True
                direct_decision = fallback_strict_direct_decision(env, config, candidate_hotspots, context)
            motion_details = apply_direct_motion(env, config, direct_decision, advance_position=False)
            decision_coverage = env.project_service_window_coverage(ticks=window_ticks)
            coverage = decision_coverage
            fallback = build_neutral_candidate_pool(
                env,
                cache_rank_limit=max(int(config.movie_num), int(direct_cache_rank_limit)),
            )
            mrsu_cache = fill_single_cache(
                direct_decision.mrsu_cache_rank,
                config.mrsu_cache_capacity,
                fallback,
                valid_content_ids,
            )
            frsu_cache = fill_single_cache(
                direct_decision.frsu_cache_rank,
                config.frsu_cache_capacity,
                fallback,
                valid_content_ids,
            )
            selected_hotspot = _find_best_request_hotspot(candidate_hotspots)
            cache_tool_details = {
                "policy": "strict_direct_llm_rank_truncated",
                "fallback_used": bool(direct_fallback),
                "strict_prompt": True,
                "tool_like_content_evidence_hidden": True,
                "direct_cache_rank_limit": int(direct_cache_rank_limit),
                "mrsu_rank_length": len(direct_decision.mrsu_cache_rank),
                "frsu_rank_length": len(direct_decision.frsu_cache_rank),
                "truncation_rule": "first valid unique ranked contents, then neutral training/global fallback",
            }
            path_plan_status = "direct_llm_no_qp_planner"
            path_plan_solver = "none"
            decision_request_source = "strict_direct_predicted_history_feedback_signal"

        elif method == METHOD_OPEN_LOOP_LLM:
            if open_loop_agent is None:
                open_loop_agent = MockOpenLoopToolAgent()
            reset_open_loop_prediction_state(env)
            uses_miss_feedback = False
            decision_request_source = "training_history_only_open_loop_independent"
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
            llm_context_chars = int(len(json.dumps(tool_context, ensure_ascii=False, separators=(",", ":"))))
            try:
                tool_decision = open_loop_agent.decide_tools(tool_context)
            except Exception as exc:
                tool_decision_error = f"{type(exc).__name__}: {exc}"
                print(f"[{METHOD_LABELS[method]}] LLM decision failed; using mock fallback. {tool_decision_error}")
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
            coverage = decision_coverage
            motion_details = {
                "policy": "open_loop_daec_qp_path_planning",
                "selected_hotspot_id": int(selected_hotspot.get("hotspot_id", 0)),
                "selected_hotspot_position": float(selected_hotspot.get("position", 0.0)),
                "lambda_smooth": float(lambda_smooth),
            }
            path_plan_status = path_plan.status if path_plan else None
            path_plan_solver = path_plan.solver if path_plan else None

        elif method == METHOD_STATIC_LLM:
            uses_miss_feedback = True
            decision_request_source = "predicted_history_feedback_signal"
            decision_requests = env.predict_round_requests(round_index, use_feedback=True)
            vehicle_demands = env.vehicle_request_counters(decision_requests)
            candidate_hotspots = [
                hotspot.to_dict()
                for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
            ]
            if tool_agent is None:
                tool_agent = MockToolAgent(update_gain_threshold=0.0)
            env.mrsu.position = fixed_position % config.road_length
            env.mrsu.speed = 0.0
            selected_hotspot = build_position_hotspot(
                env,
                env.mrsu.position,
                decision_requests,
                hotspot_id=0,
                selection_basis="static_initial_position_no_path_planning",
            )
            decision_coverage = env.project_service_window_coverage(
                ticks=window_ticks,
                hold_mrsu_position=True,
            )
            content_features = env.content_features(
                decision_coverage.mrsu_covered,
                decision_coverage.frsu_covered,
                decision_requests,
                config.global_topk_for_prompt,
            )
            fit_summary = env.cache_fit_summary(decision_coverage, decision_requests)
            cache_fit_analysis = cache_evaluator.build_acr_cache_fit_analysis(
                current_mrsu_cache=env.mrsu_cache,
                current_frsu_cache=env.frsu_cache,
                vehicle_requests=decision_requests,
                coverage=decision_coverage,
                selected_hotspot=selected_hotspot,
                content_features=content_features,
                fit_summary=fit_summary,
            )
            tool_context = env.build_tool_decision_context(
                round_index=round_index,
                vehicle_requests=decision_requests,
                candidate_hotspots=candidate_hotspots,
                selected_hotspot=selected_hotspot,
                coverage=decision_coverage,
                request_source=decision_request_source,
                cache_fit_analysis=cache_fit_analysis,
                fit_summary=fit_summary,
                content_features=content_features,
            )
            tool_context["ablation_method"] = method
            tool_context["ablation_label"] = METHOD_LABELS[method]
            tool_context["path_planning_disabled"] = True
            tool_context["uses_miss_feedback"] = bool(uses_miss_feedback)
            tool_context["ablation_contract"] = (
                "The mRSU is fixed at its initial position. Do not plan movement and do not output cache lists. "
                "Only decide whether the mRSU/fRSU caches should be updated by DemandAwareCooperativeCacheTool."
            )
            llm_context_chars = len(json.dumps(tool_context, ensure_ascii=False))
            try:
                tool_decision = tool_agent.decide_tools(tool_context)
            except Exception as exc:
                tool_decision_error = f"{type(exc).__name__}: {exc}"
                print(f"[Static mRSU] LLM tool decision failed; using MockToolAgent fallback. {tool_decision_error}")
                tool_decision = MockToolAgent(update_gain_threshold=0.0).decide_tools(tool_context)
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
            cache_update_used = bool(need_mrsu_update or need_frsu_update)
            coverage = decision_coverage
            motion_details = {
                "policy": "static_mrsu_no_path_planning_tool_cache",
                "position": float(env.mrsu.position),
                "speed": float(env.mrsu.speed),
                "tool_selected_hotspot_id": int(tool_decision.selected_hotspot_id),
                "tool_selected_hotspot_ignored_for_motion": True,
            }
            path_plan_status = "static_no_path_planning"
            path_plan_solver = "none"

        else:
            raise ValueError(f"Unknown ablation method: {method}")

        env.set_cache(mrsu_cache, frsu_cache)
        if method == METHOD_STATIC_LLM:
            coverage = env.execute_service_window(
                ticks=window_ticks,
                hold_mrsu_position=True,
            )
        else:
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
                "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
                "latency": latency,
                "request_count": int(metrics.request_count),
                "decision_request_count": int(sum(len(requests) for requests in decision_requests.values())),
                "evaluation_request_count": int(metrics.request_count),
                "decision_request_source": decision_request_source,
                "uses_miss_feedback": bool(uses_miss_feedback),
                "feedback_visible_to_llm": bool(uses_miss_feedback),
                "feedback_used_by_dacc": bool(uses_miss_feedback),
                "prediction_state_reset_each_decision": method == METHOD_OPEN_LOOP_LLM,
                "test_requests_hidden_until_evaluation": True,
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "selected_hotspot": selected_hotspot,
                "motion_details": motion_details,
                "path_plan_status": path_plan_status,
                "path_plan_solver": path_plan_solver,
                "tool_decision": tool_decision.to_dict() if tool_decision else None,
                "tool_decision_error": tool_decision_error,
                "llm_context_chars": llm_context_chars,
                "llm_context_schema": llm_context_schema_for_method(method),
                "cache_fit_analysis": cache_fit_analysis,
                "cache_update_used": cache_update_used,
                "direct_llm_decision": direct_decision.to_dict() if direct_decision else None,
                "direct_llm_error": direct_error,
                "direct_llm_fallback": direct_fallback,
                "direct_cache_rank_limit": int(direct_cache_rank_limit)
                if method in (METHOD_DIRECT_LLM, METHOD_OPEN_LOOP_LLM)
                else None,
                "direct_llm_mrsu_rank_length": len(direct_decision.mrsu_cache_rank)
                if direct_decision
                else 0,
                "direct_llm_frsu_rank_length": len(direct_decision.frsu_cache_rank)
                if direct_decision
                else 0,
                "mrsu_position": float(env.mrsu.position),
                "mrsu_speed": float(env.mrsu.speed),
                "mrsu_covered": [int(x) for x in coverage.mrsu_covered],
                "frsu_covered": [int(x) for x in coverage.frsu_covered],
                "decision_mrsu_covered": [int(x) for x in decision_coverage.mrsu_covered],
                "decision_frsu_covered": [int(x) for x in decision_coverage.frsu_covered],
                "overlap": [int(x) for x in coverage.overlap],
                "mrsu_cache": [int(x) for x in mrsu_cache],
                "frsu_cache": [int(x) for x in frsu_cache],
                "cache_tool_details": cache_tool_details,
                "metrics": metrics.to_dict(),
            }
        )
        if verbose:
            fallback_text = " direct_fallback=True" if direct_fallback else ""
            print(
                f"[{METHOD_LABELS[method]}] round={round_index:02d} "
                f"LocalCHR={metrics.local_rsu_chr:.4f} "
                f"mRSU_hit={metrics.mrsu_hit_count} "
                f"fRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count}"
                f" Delay={latency.get('average_delay_ms', 0.0):.2f}ms"
                f"{fallback_text}"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(summarize_round_latencies(log.get("latency") for log in round_logs))
    summary.update(
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "method_note": METHOD_NOTES[method],
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "latency_model": latency_model.config.to_dict(),
            "round_chr": [float(item.chr) for item in round_metrics],
            "round_local_rsu_chr": [float(item.local_rsu_chr) for item in round_metrics],
            "round_delay_ms": [float(log.get("latency", {}).get("average_delay_ms", 0.0)) for log in round_logs],
            "cache_update_count": (
                sum(1 for log in round_logs if log.get("cache_update_used"))
                if method in (METHOD_OPEN_LOOP_LLM, METHOD_STATIC_LLM)
                else 0
            ),
            "direct_llm_fallback_count": sum(1 for log in round_logs if log.get("direct_llm_fallback")),
            "uses_miss_feedback": method != METHOD_OPEN_LOOP_LLM,
            "feedback_visible_to_llm": method != METHOD_OPEN_LOOP_LLM,
            "feedback_used_by_dacc": method != METHOD_OPEN_LOOP_LLM,
            "prediction_state_reset_each_decision": method == METHOD_OPEN_LOOP_LLM,
            "prediction_source": (
                "training_history_only" if method == METHOD_OPEN_LOOP_LLM else "predicted_history_feedback_signal"
            ),
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def update_cache_with_dacc(
    env: MRSUEnvironment,
    cache_evaluator: CacheUpdateEvaluator,
    vehicle_requests: Dict[int, List[int]],
    coverage,
    selected_hotspot: dict,
) -> Tuple[List[int], List[int], dict]:
    content_features = env.content_features(
        coverage.mrsu_covered,
        coverage.frsu_covered,
        vehicle_requests,
        env.config.global_topk_for_prompt,
    )
    fit_summary = env.cache_fit_summary(coverage, vehicle_requests)
    return cache_evaluator.update_with_acr_tool(
        vehicle_requests=vehicle_requests,
        coverage=coverage,
        selected_hotspot=selected_hotspot,
        content_features=content_features,
        fit_summary=fit_summary,
        update_mrsu=True,
        update_frsu=True,
        current_mrsu_cache=env.mrsu_cache,
        current_frsu_cache=env.frsu_cache,
    )


def build_strict_direct_context(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    round_index: int,
    decision_requests: Dict[int, List[int]],
    candidate_hotspots: List[dict],
    cache_rank_limit: int,
) -> dict:
    request_counts = {int(vid): int(len(reqs)) for vid, reqs in decision_requests.items()}
    vehicles = [
        {
            "id": int(vehicle.vehicle_id),
            "x": round(float(vehicle.position), 1),
            "v": round(float(vehicle.speed), 1),
            "req_n": request_counts.get(int(vehicle.vehicle_id), 0),
        }
        for vehicle in env.mobility.vehicles
    ]
    hotspots = [
        {
            "id": int(hotspot.get("hotspot_id", idx)),
            "position": round(float(hotspot.get("position", 0.0)), 1),
            "covered_vehicle_count": int(hotspot.get("covered_vehicle_count", 0)),
            "potential_cache_gain": round(float(hotspot.get("potential_cache_gain", 0.0)), 2),
            "forward_distance": round(
                CoverageModel.forward_distance(
                    float(env.mrsu.position),
                    float(hotspot.get("position", 0.0)),
                    float(config.road_length),
                ),
                1,
            ),
        }
        for idx, hotspot in enumerate(candidate_hotspots[: config.candidate_count])
    ]
    return {
        "round": int(round_index),
        "road": {
            "topology": "circular_one_way",
            "length": float(config.road_length),
            "dt": float(config.dt),
            "decision_interval_ticks": int(config.decision_interval),
            "service_window": "decision is held; vehicles covered at least once can be served",
        },
        "mrsu": {
            "position": round(float(env.mrsu.position), 1),
            "speed": round(float(env.mrsu.speed), 1),
            "radius": float(config.mrsu_radius),
            "cache_capacity": int(config.mrsu_cache_capacity),
            "v_min": float(config.mrsu_v_min),
            "v_max": float(config.mrsu_v_max),
            "a_min": float(config.mrsu_a_min),
            "a_max": float(config.mrsu_a_max),
        },
        "frsu": {
            "position": round(float(env.frsu.position), 1),
            "radius": float(config.frsu_radius),
            "cache_capacity": int(config.frsu_cache_capacity),
        },
        "feedback": {
            "last_chr": round(float(env.last_metrics.chr), 4),
            "last_not_covered_count": int(env.last_metrics.not_covered_count),
            "last_not_cached_count": int(env.last_metrics.not_cached_count),
            "last_mbs_miss_count": int(env.last_metrics.mbs_miss_count),
        },
        "vehicles": vehicles,
        "hotspots": hotspots,
        "candidate_pool": build_neutral_candidate_pool(env, cache_rank_limit),
        "cache_rank_limit": int(cache_rank_limit),
        "valid_content_id_range": [1, int(config.movie_num)],
        "hidden_from_direct_llm": [
            "QPPathPlanner output",
            "DACC cache-fit analysis",
            "regional content demand summaries",
            "current-round true requests",
        ],
    }


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


def build_neutral_candidate_pool(env: MRSUEnvironment, cache_rank_limit: int) -> List[int]:
    items: List[int] = []
    items.extend(int(x) for x in env.global_top_contents)
    for user_id in env.user_ids:
        history = env.user_history.get(int(user_id), [])
        for row in history:
            if isinstance(row, (list, tuple)) and row:
                items.append(int(row[0]))
            else:
                try:
                    items.append(int(row))
                except (TypeError, ValueError):
                    continue
    return _unique_valid(items, range(1, env.config.movie_num + 1), int(cache_rank_limit))


def speed_toward_target(context: Dict[str, Any], target_position: float) -> float:
    road = context.get("road") or {}
    mrsu = context.get("mrsu") or {}
    road_length = max(float(road.get("length", 1000.0)), 1e-9)
    dt = max(float(road.get("dt", 1.0)), 1e-9)
    interval = max(int(road.get("decision_interval_ticks", 1)), 1)
    current_position = float(mrsu.get("position", 0.0))
    distance = CoverageModel.forward_distance(current_position, float(target_position), road_length)
    requested_speed = float(distance) / max(float(interval) * dt, 1e-9)
    v_min = float(mrsu.get("v_min", 0.0))
    v_max = float(mrsu.get("v_max", max(requested_speed, 0.0)))
    return float(np.clip(requested_speed, v_min, v_max))


def fallback_strict_direct_decision(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    candidate_hotspots: List[dict],
    context: Dict[str, Any],
) -> DirectLLMDecision:
    best = _find_best_request_hotspot(candidate_hotspots)
    target_position = float(best.get("position", env.mrsu.position)) % float(config.road_length)
    distance = CoverageModel.forward_distance(env.mrsu.position, target_position, config.road_length)
    requested_speed = float(distance) / max(float(config.decision_interval) * float(config.dt), 1e-9)
    pool = context.get("candidate_pool") or build_neutral_candidate_pool(env, 400)
    limit = int(context.get("cache_rank_limit", 400))
    return DirectLLMDecision(
        target_position=target_position,
        next_speed=float(np.clip(requested_speed, config.mrsu_v_min, config.mrsu_v_max)),
        mrsu_cache_rank=limit_unique_ints(pool, limit),
        frsu_cache_rank=limit_unique_ints(pool, limit),
        reason="fallback strict direct decision from neutral training/global pool",
    )


def strip_feedback_from_context(context: Dict[str, Any]) -> None:
    context["request_signal_note"] = (
        "Demand fields are decision-time open-loop prediction signals from MovieLens training history "
        "and global popularity. Previous-round hit/miss feedback is hidden from this ablation."
    )
    context["last_round_hit_ratio"] = None
    context["last_round_missed_contents"] = []
    context["last_round_vehicle_missed_contents"] = {}
    context["miss_reason_summary"] = {}
    fit = context.get("cache_fit_summary")
    if isinstance(fit, dict):
        fit["mrsu_covered_vehicle_feedback_contents"] = []
        fit["frsu_covered_vehicle_feedback_contents"] = []


def apply_direct_motion(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    decision: DirectLLMDecision,
    advance_position: bool = True,
) -> dict:
    current_position = float(env.mrsu.position)
    current_speed = float(env.mrsu.speed)
    dt = max(float(config.dt), 1e-9)
    road_length = max(float(config.road_length), 1e-9)
    target_position = float(decision.target_position) % road_length
    requested_speed = float(decision.next_speed)
    if not np.isfinite(requested_speed):
        requested_speed = CoverageModel.forward_distance(current_position, target_position, road_length) / dt
    accel_limited = float(
        np.clip(
            requested_speed,
            current_speed + config.mrsu_a_min * dt,
            current_speed + config.mrsu_a_max * dt,
        )
    )
    next_speed = float(np.clip(accel_limited, config.mrsu_v_min, config.mrsu_v_max))
    next_position = (current_position + next_speed * dt) % road_length
    env.mrsu.speed = next_speed
    if advance_position:
        env.mrsu.position = next_position
    return {
        "policy": "direct_llm_motion",
        "target_position": target_position,
        "requested_speed": requested_speed,
        "executed_speed": next_speed,
        "previous_position": current_position,
        "next_position": next_position,
        "position_advanced_immediately": bool(advance_position),
        "acceleration": (next_speed - current_speed) / dt,
    }


def fallback_direct_decision(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    candidate_hotspots: List[dict],
    vehicle_requests: Dict[int, List[int]],
) -> DirectLLMDecision:
    best = _find_best_request_hotspot(candidate_hotspots) or {}
    target_position = float(best.get("position", env.mrsu.position)) % float(config.road_length)
    local = build_cache_fallback(env, env.coverage_snapshot(), vehicle_requests, candidate_hotspots)
    return DirectLLMDecision(
        target_position=target_position,
        next_speed=CoverageModel.forward_distance(env.mrsu.position, target_position, config.road_length)
        / max(config.dt, 1e-9),
        mrsu_cache_rank=local,
        frsu_cache_rank=local,
        reason="fallback direct decision from predicted demand after LLM failure",
    )


def build_position_hotspot(
    env: MRSUEnvironment,
    position: float,
    vehicle_requests: Dict[int, List[int]],
    hotspot_id: int,
    selection_basis: str,
    potential_cache_gain: float = None,
) -> dict:
    coverage = env.coverage_snapshot(mrsu_position=float(position))
    demand = env.demand_for_vehicles(coverage.mrsu_covered, vehicle_requests)
    demand_summary = {int(content_id): int(count) for content_id, count in demand.most_common(20)}
    return {
        "hotspot_id": int(hotspot_id),
        "position": float(position) % float(env.config.road_length),
        "covered_vehicle_ids": [int(x) for x in coverage.mrsu_covered],
        "covered_vehicle_count": len(coverage.mrsu_covered),
        "potential_cache_gain": float(
            potential_cache_gain if potential_cache_gain is not None else sum(demand.values())
        ),
        "dominant_contents": [int(content_id) for content_id, _ in demand.most_common(20)],
        "demand_summary": demand_summary,
        "selection_basis": selection_basis,
    }


def _find_best_request_hotspot(candidate_hotspots: List[dict]) -> dict:
    if not candidate_hotspots:
        return {}
    return max(
        candidate_hotspots,
        key=lambda item: (
            float(item.get("potential_cache_gain", 0.0)),
            int(item.get("covered_vehicle_count", 0)),
        ),
    )


def build_cache_fallback(
    env: MRSUEnvironment,
    coverage,
    vehicle_requests: Dict[int, List[int]],
    candidate_hotspots: List[dict],
) -> List[int]:
    items: List[int] = []
    mrsu_demand = env.demand_for_vehicles(coverage.mrsu_covered, vehicle_requests)
    frsu_demand = env.demand_for_vehicles(coverage.frsu_covered, vehicle_requests)
    items.extend(content for content, _ in mrsu_demand.most_common())
    items.extend(content for content, _ in frsu_demand.most_common())
    for hotspot in candidate_hotspots:
        items.extend(_sorted_keys_by_value(hotspot.get("demand_summary") or {}))
        items.extend(int(x) for x in hotspot.get("dominant_contents", []))
    items.extend(env.global_top_contents)
    return _unique_valid(items, range(1, env.config.movie_num + 1))


def rank_with_global_fallback(items: Sequence[int], context: Dict[str, Any], limit: int) -> List[int]:
    fallback: List[int] = []
    fallback.extend(int(x) for x in context.get("cache_rank_candidate_pool") or [])
    for feature in sorted(
        context.get("candidate_contents") or [],
        key=lambda row: int(row.get("global_popularity", 0)),
        reverse=True,
    ):
        fallback.append(int(feature.get("content_id", 0)))
    for hotspot in context.get("candidate_hotspots") or []:
        fallback.extend(_sorted_keys_by_value(hotspot.get("demand_summary") or {}))
        fallback.extend(int(x) for x in hotspot.get("dominant_contents", []))
    for history in (context.get("user_history_sample") or {}).values():
        for item in history or []:
            if isinstance(item, (list, tuple)) and item:
                fallback.append(int(item[0]))
            else:
                try:
                    fallback.append(int(item))
                except (TypeError, ValueError):
                    continue
    return limit_unique_ints(list(items or []) + fallback, limit)


def fill_single_cache(
    items: Sequence[int],
    capacity: int,
    fallback: Sequence[int],
    valid_content_ids: Iterable[int],
) -> List[int]:
    valid = set(int(x) for x in valid_content_ids)
    result: List[int] = []
    seen = set()
    for item in list(items or []) + list(fallback or []):
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id not in valid or content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
        if len(result) >= int(capacity):
            break
    return result[: int(capacity)]


def limit_unique_ints(items: Sequence[int], limit: int) -> List[int]:
    result: List[int] = []
    seen = set()
    for item in items or []:
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
        if len(result) >= int(limit):
            break
    return result


def resolve_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in DEFAULT_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def _as_int_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    result: List[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _sorted_keys_by_value(mapping: Dict) -> List[int]:
    return [
        int(key)
        for key, _ in sorted(
            mapping.items(),
            key=lambda item: int(item[1]),
            reverse=True,
        )
    ]


def _unique_valid(items: Iterable[int], valid_content_ids: Iterable[int], limit: int = None) -> List[int]:
    valid = set(int(x) for x in valid_content_ids)
    seen = set()
    result = []
    for item in items:
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id in seen or content_id not in valid:
            continue
        seen.add(content_id)
        result.append(content_id)
        if limit is not None and len(result) >= int(limit):
            break
    return result

