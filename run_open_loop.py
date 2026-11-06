"""Run the open_loop ablation."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from experiments.open_loop import build_open_loop_agent, run_open_loop_daec_method


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("open_loop", args):
        return
    agent, latency = build_open_loop_agent(args), build_latency_model(args)
    with RunOutput("open_loop", args) as output:
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                result = run_open_loop_daec_method(config, agent=agent, latency_model=latency, verbose=not args.quiet)
                output.append(result, config, agent=agent, model=args.model_name)


if __name__ == "__main__":
    main()
