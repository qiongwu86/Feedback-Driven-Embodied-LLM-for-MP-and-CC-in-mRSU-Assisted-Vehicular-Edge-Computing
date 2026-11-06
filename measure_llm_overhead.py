"""Measure end-to-end LLM API response latency without injecting it into ACTD."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from simulation.runner import run_method, _resolve_api_key
from experiments.llm_overhead import MeasuredLLMAgent, WindowObserver, TelemetryStore


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
    "agent": "llm",
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("llm_overhead", args):
        return
    if args.agent not in ("llm", "mock"):
        raise ValueError("Overhead measurement requires --agent llm or --agent mock.")
    api_key = _resolve_api_key(args.api_key_env) if args.agent == "llm" else ""
    if args.agent == "llm" and not api_key:
        raise RuntimeError("No API key is configured; overhead measurement will not silently switch to Mock.")
    latency = build_latency_model(args)
    with RunOutput("llm_overhead", args) as output:
        telemetry = TelemetryStore(output.path)
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                identity = {"model": args.model_name, "seed": seed, "capacity": capacity}
                agent = MeasuredLLMAgent(args, args.model_name, api_key, telemetry, identity)
                observer = WindowObserver(config, agent, telemetry, identity)
                status = "failed"
                try:
                    result = run_method(config, "tool_agent", agent=agent, verbose=False,
                                        latency_model=latency, timing_observer=observer,
                                        cache_update_candidate_limit=args.cache_update_candidate_limit)
                    output.append(result, config, agent=agent, model=args.model_name,
                                  controller_mode=args.agent, api_wait_injected_into_simulation=False)
                    status = "complete"
                finally:
                    agent.close()
                    telemetry.checkpoint(status)


if __name__ == "__main__":
    main()
