"""Parameter parsing shared by the independent executable scripts."""
import argparse
import json
from dataclasses import fields

from common.paths import ROOT, RESULTS, project_path
from communication.latency_model import CV2XLatencyModel, LatencyModelConfig
from simulation.config import MRSUSimulationConfig


def int_list(value):
    values = value if isinstance(value, (list, tuple)) else str(value).split(",")
    result = list(dict.fromkeys(int(x) for x in values if str(x).strip()))
    if not result:
        raise ValueError("At least one integer is required.")
    return result


def parse_args(description, hyperparameters, argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(ROOT / "configs/default.json"))
    preliminary, _ = pre.parse_known_args(argv)
    defaults = json.loads(project_path(preliminary.config).read_text(encoding="utf-8"))
    defaults.update(hyperparameters)
    parser = argparse.ArgumentParser(description=description, parents=[pre])
    excluded = {"seed", "mrsu_cache_capacity", "frsu_cache_capacity", "output_dir"}
    for field in fields(MRSUSimulationConfig):
        if field.name in excluded:
            continue
        default = defaults.get(field.name, field.default)
        option = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(option, type=float if field.default is None else type(field.default), default=default)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rsu-cache", type=int, default=defaults.get("rsu_cache", 200))
    parser.add_argument("--capacities", default=None)
    parser.add_argument("--mrsu-cache", type=int, default=None)
    parser.add_argument("--frsu-cache", type=int, default=None)
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--agent", choices=["auto", "mock", "llm", "gemini", "gemini-rest"], default=defaults.get("agent", "auto"))
    parser.add_argument("--model-name", default=defaults.get("model_name", "qwen3.7-flash"))
    parser.add_argument("--base-url", default=defaults.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key-env", default=defaults.get("api_key_env", ""))
    parser.add_argument("--max-context-chars", type=int, default=defaults.get("max_context_chars", 24000))
    parser.add_argument("--strict-cache-rank-limit", type=int, default=defaults.get("strict_cache_rank_limit", 400))
    parser.add_argument("--update-threshold", type=float, default=defaults.get("update_threshold", .015))
    parser.add_argument("--cache-update-candidate-limit", type=int, default=defaults.get("cache_update_candidate_limit", 30))
    parser.add_argument("--cp-sat-time-limit", type=float, default=defaults.get("cp_sat_time_limit", 10.))
    for field in fields(LatencyModelConfig):
        parser.add_argument("--latency-" + field.name.replace("_", "-"), type=float,
                            default=defaults.get("latency_"+field.name, field.default))
    typed = {"episodes": (int, 800), "hidden_dim": (int, 128), "lr": (float, .001),
             "gamma": (float, .98), "batch_size": (int, 64), "buffer_size": (int, 20000),
             "epsilon_start": (float, 1.), "epsilon_end": (float, .05),
             "epsilon_decay": (float, .99), "target_update_steps": (int, 200),
             "reward_not_covered_penalty": (float, 0.), "reward_not_cached_penalty": (float, 0.),
             "temperature": (float, 0.), "top_p": (float, 1.), "api_timeout": (float, 120.),
             "max_retries": (int, 2), "retry_base_delay": (float, .5)}
    for name, (kind, default) in typed.items():
        parser.add_argument("--"+name.replace("_", "-"), type=kind, default=defaults.get(name, default))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=defaults.get("device", "auto"))
    parser.add_argument("--model-path", default=defaults.get("model_path", ""))
    parser.add_argument("--models", default=None)
    parser.add_argument("--vehicle-counts", default=None)
    parser.add_argument("--enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--on-api-error", choices=["fallback", "abort"], default="fallback")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.seeds = [args.seed] if args.seed is not None else int_list(defaults.get("seeds", [42]))
    args.capacities = int_list(args.capacities or ([args.rsu_cache] if "--rsu-cache" in (argv or __import__("sys").argv[1:]) else defaults.get("capacities", [args.rsu_cache])))
    args.models = [s.strip() for s in (args.models or ",".join(defaults.get("models", [args.model_name]))).split(",") if s.strip()]
    args.vehicle_counts = int_list(args.vehicle_counts or defaults.get("vehicle_counts", [args.vehicle_num]))
    if any(c <= 0 for c in args.capacities) or args.rounds <= 0 or args.decision_interval <= 0:
        parser.error("Capacities, rounds and decision interval must be positive.")
    return args


def build_config(args, seed, capacity, output_dir=None, vehicle_num=None):
    excluded = {"seed", "mrsu_cache_capacity", "frsu_cache_capacity", "output_dir"}
    values = {f.name: getattr(args, f.name) for f in fields(MRSUSimulationConfig) if f.name not in excluded}
    if vehicle_num is not None:
        values["vehicle_num"] = vehicle_num
        if args.user_num <= 0:
            values["user_num"] = vehicle_num
    return MRSUSimulationConfig(**values, seed=seed,
        mrsu_cache_capacity=args.mrsu_cache if args.mrsu_cache is not None else capacity,
        frsu_cache_capacity=args.frsu_cache if args.frsu_cache is not None else capacity,
        output_dir=str(output_dir or project_path(args.output_dir)))


def build_latency_model(args):
    return CV2XLatencyModel(LatencyModelConfig(**{
        f.name: getattr(args, "latency_"+f.name) for f in fields(LatencyModelConfig)}))


def show_plan(method, args):
    if args.dry_run:
        print(json.dumps({"method": method, "seeds": args.seeds, "capacities": args.capacities,
                          "model": args.model_name, "output_root": str(project_path(args.output_dir))}, indent=2))
    return args.dry_run
