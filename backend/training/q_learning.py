from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from random import Random
from typing import Sequence

from backend.env.environment import RacingEnv
from backend.rl.agent import QLearningAgent, QLearningConfig

from .evaluator import EvaluationRecord
from .runner import EpisodeRecord, summarize_run
from .trainer import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the tabular Q-learning racer")
    parser.add_argument("--episodes", type=_positive_int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-every", type=_positive_int, default=50)
    parser.add_argument("--evaluation-episodes", type=_positive_int, default=10)
    parser.add_argument("--report-every", type=_positive_int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--buckets", type=_bucket_count, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = QLearningConfig(
            learning_rate=args.learning_rate,
            discount=args.discount,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            bucket_count=args.buckets,
        )
    except ValueError as error:
        raise SystemExit(f"invalid Q-learning configuration: {error}") from error

    agent = QLearningAgent(Random(args.seed), config)

    def report_episode(record: EpisodeRecord, total_steps: int, elapsed: float, epsilon: float) -> None:
        if record.episode % args.report_every == 0 or record.episode == args.episodes:
            throughput = total_steps / elapsed if elapsed else 0.0
            print(
                f"episode {record.episode}/{args.episodes} | steps {total_steps} | "
                f"{throughput:.1f} steps/s | epsilon {epsilon:.4f}"
            )

    def report_evaluation(record: EvaluationRecord) -> None:
        print(
            f"evaluation @{record.training_episode} | mean progress "
            f"{record.mean_progress:.3%} | best {record.best_progress:.3%}"
        )

    result = run_training(
        RacingEnv(),
        agent,
        args.episodes,
        evaluate_every=args.evaluate_every,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_seed=args.seed + 1_000_000,
        on_episode=report_episode,
        on_evaluation=report_evaluation,
    )
    summary = summarize_run(list(result.records), seed=args.seed, wall_time=result.training_wall_time)
    output = {
        "config": asdict(config),
        "summary": asdict(summary),
        "final_epsilon": agent.epsilon,
        "visited_states": agent.visited_states,
        "evaluations": [asdict(evaluation) for evaluation in result.evaluations],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _bucket_count(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
