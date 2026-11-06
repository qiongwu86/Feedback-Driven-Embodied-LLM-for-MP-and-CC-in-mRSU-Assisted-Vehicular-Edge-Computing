"""Evaluate a trained DQN baseline."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from baselines.dqn_baseline import evaluate_dqn
from common.paths import ROOT, project_path


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
}


def checkpoint_for(args, capacity):
    if args.model_path:
        path = project_path(args.model_path.format(capacity=capacity))
        if not path.is_file():
            raise FileNotFoundError(f"DQN checkpoint not found: {path}")
        return path
    name = f"dqn_model_capacity_{capacity}.pt"
    fixed = ROOT/"checkpoints"/name
    if fixed.is_file():
        return fixed
    matches = list((project_path(args.output_dir)/"dqn").glob("*/checkpoints/seed_*/capacity_*/"+name))
    if not matches:
        raise FileNotFoundError(f"No trained DQN checkpoint for capacity {capacity}.")
    return max(matches, key=lambda p: p.stat().st_mtime_ns)


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("dqn", args):
        return
    models = {capacity: checkpoint_for(args, capacity) for capacity in args.capacities}
    latency = build_latency_model(args)
    with RunOutput("dqn", args) as output:
        output.manifest["phase"] = "evaluation"
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                result = evaluate_dqn(config, models[capacity], seed=seed, device=args.device,
                                      latency_model=latency, verbose=not args.quiet)
                output.append(result, config, checkpoint=str(models[capacity]), phase="evaluation")


if __name__ == "__main__":
    main()
