"""Evaluate FD-EMC with configurable vehicle counts."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from simulation.runner import run_method, build_agent


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
    "vehicle_counts": [50],
    "user_num": 0,
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("vehicle_count", args):
        return
    agent, latency = build_agent(args), build_latency_model(args)
    with RunOutput("vehicle_count", args) as output:
        for count in args.vehicle_counts:
            for seed in args.seeds:
                for capacity in args.capacities:
                    config = build_config(args, seed, capacity, output.path, vehicle_num=count)
                    result = run_method(config, "tool_agent", agent=agent, verbose=not args.quiet,
                                        latency_model=latency,
                                        cache_update_candidate_limit=args.cache_update_candidate_limit)
                    output.append(result, config, agent=agent, model=args.model_name)


if __name__ == "__main__":
    main()
