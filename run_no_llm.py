"""Run FD-EMC w/o LLM with the existing deterministic numerical rule."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from experiments.rule_controller import run_controller


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("no_llm", args):
        return
    latency = build_latency_model(args)
    with RunOutput("no_llm", args) as output:
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                result = run_controller(config, latency.config, args.update_threshold, not args.quiet)
                output.append(result, config, controller_mode="rule")


if __name__ == "__main__":
    main()
