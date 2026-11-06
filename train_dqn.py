"""Train the conventional DQN baseline."""
from common.cli import parse_args, build_config, build_latency_model, show_plan
from common.io import RunOutput
from baselines.dqn_baseline import train_dqn


HYPERPARAMETERS = {
    "seeds": [42],
    "capacities": [200],
    "episodes": 800,
    "lr": 0.001,
    "gamma": 0.98,
    "batch_size": 64,
    "epsilon_decay": 0.99,
}


def main(argv=None):
    args = parse_args(__doc__, HYPERPARAMETERS, argv)
    if show_plan("dqn_train", args):
        return
    names = ("episodes", "hidden_dim", "lr", "gamma", "batch_size", "buffer_size",
             "epsilon_start", "epsilon_end", "epsilon_decay", "target_update_steps",
             "device", "reward_not_covered_penalty", "reward_not_cached_penalty")
    with RunOutput("dqn", args) as output:
        output.manifest["phase"] = "training"
        for seed in args.seeds:
            for capacity in args.capacities:
                config = build_config(args, seed, capacity, output.path)
                directory = output.path/"checkpoints"/f"seed_{seed}"/f"capacity_{capacity}"
                result = train_dqn(config, str(directory), seed=seed, rounds=config.rounds,
                                   verbose=not args.quiet, **{n: getattr(args, n) for n in names})
                result["summary"] = {"episodes": args.episodes, "model_path": result["model_path"]}
                output.append(result, config, phase="training")


if __name__ == "__main__":
    main()
