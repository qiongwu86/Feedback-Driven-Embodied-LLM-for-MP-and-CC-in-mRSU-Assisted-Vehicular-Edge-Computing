"""Run the cp_sat baseline."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from baselines.traditional_baselines import run_traditional_baseline


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("cp_sat", args):
        return
    latency = build_latency_model(args)
    with RunOutput("cp_sat", args) as output:
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                result = run_traditional_baseline(config, "cp_sat", verbose=not args.quiet,
                                                 cp_sat_time_limit=args.cp_sat_time_limit, latency_model=latency)
                output.append(result, config)


if __name__ == "__main__":
    main()
