"""Compare LLM backbones using the same FD-EMC execution chain."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from simulation.runner import run_method, build_agent


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
    "models": ["qwen3.7-flash", "deepseek-v4-flash"],
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("llm_models", args):
        return
    latency = build_latency_model(args)
    with RunOutput("llm_models", args) as output:
        for model in args.models:
            args.model_name = model
            agent = build_agent(args)
            for seed in args.seeds:
                for capacity in args.capacities:
                    config = build_config(args, seed, capacity, output.path)
                    result = run_method(config, "tool_agent", agent=agent, verbose=not args.quiet,
                                        latency_model=latency,
                                        cache_update_candidate_limit=args.cache_update_candidate_limit)
                    output.append(result, config, agent=agent, model=model)


if __name__ == "__main__":
    main()
