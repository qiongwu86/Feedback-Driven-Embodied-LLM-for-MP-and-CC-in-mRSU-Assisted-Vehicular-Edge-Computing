"""Run the Static mRSU ablation."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from experiments.ablation_runner import run_ablation_method, build_tool_agent


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("static_mrsu", args):
        return
    agent, latency = build_tool_agent(args), build_latency_model(args)
    with RunOutput("static_mrsu", args) as output:
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                result = run_ablation_method(config, "static_mrsu_llm", tool_agent=agent,
                                             latency_model=latency, verbose=not args.quiet)
                output.append(result, config, agent=agent, model=args.model_name)


if __name__ == "__main__":
    main()
