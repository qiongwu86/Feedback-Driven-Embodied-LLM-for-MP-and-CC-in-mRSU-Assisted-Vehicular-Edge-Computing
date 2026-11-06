"""Run the direct_llm ablation."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from experiments.direct_llm import build_strict_direct_agent, run_strict_direct_llm_method


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("direct_llm", args):
        return
    agent, latency = build_strict_direct_agent(args), build_latency_model(args)
    with RunOutput("direct_llm", args) as output:
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                result = run_strict_direct_llm_method(config, direct_agent=agent, latency_model=latency, verbose=not args.quiet)
                output.append(result, config, agent=agent, model=args.model_name)


if __name__ == "__main__":
    main()
