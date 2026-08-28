from __future__ import annotations

import argparse
import json
import signal
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Sequence, TypeVar

from backend.env.environment import RacingEnv
from backend.rl.agent import QLearningAgent, QLearningConfig

from .checkpoint import (
    DEFAULT_CHECKPOINT_PATH,
    CheckpointError,
    TrainingCheckpoint,
    checkpoint_from_agent,
    load_checkpoint,
    save_checkpoint,
)
from .curriculum import AdaptiveCurriculum
from .evaluator import EvaluationRecord
from .replay import EvaluationReplay
from .runner import EpisodeRecord, summarize_run
from .trainer import run_training

T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the tabular Q-learning racer")
    parser.add_argument("--episodes", type=_positive_int, default=3_000, help="total target episode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-every", type=_positive_int)
    parser.add_argument("--evaluation-episodes", type=_positive_int)
    parser.add_argument("--report-every", type=_positive_int, default=50)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--discount", type=float)
    parser.add_argument("--epsilon-start", type=float)
    parser.add_argument("--epsilon-min", type=float)
    parser.add_argument("--epsilon-decay", type=float)
    parser.add_argument("--buckets", type=_bucket_count)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=f"checkpoint output path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument("--resume", type=Path, help="resume a saved checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        setup = _training_setup(args)
    except (CheckpointError, ValueError) as error:
        raise SystemExit(str(error)) from error

    agent = setup.agent
    records = list(setup.records)
    evaluations = list(setup.evaluations)
    replays = list(setup.replays)
    curriculum = (
        AdaptiveCurriculum.from_snapshot(setup.curriculum)
        if setup.curriculum is not None
        else AdaptiveCurriculum(Random(setup.seed + 2_000_000))
    )
    base_wall_time = setup.training_wall_time
    checkpoint_path = args.checkpoint or args.resume or DEFAULT_CHECKPOINT_PATH
    stop = _GracefulStop()
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, stop.handle_signal)
    latest_elapsed = 0.0

    def persist(current_wall_time: float, destination: Path | None = None) -> None:
        if not records:
            return
        checkpoint = checkpoint_from_agent(
            agent,
            seed=setup.seed,
            completed_episode=records[-1].episode,
            evaluate_every=setup.evaluate_every,
            evaluation_episodes=setup.evaluation_episodes,
            evaluation_seed=setup.evaluation_seed,
            training_wall_time=base_wall_time + current_wall_time,
            records=tuple(records),
            evaluations=tuple(evaluations),
            replays=tuple(replays),
            curriculum=curriculum.snapshot(),
        )
        save_checkpoint(checkpoint, destination or checkpoint_path)

    def report_episode(record: EpisodeRecord, total_steps: int, elapsed: float, epsilon: float) -> None:
        nonlocal latest_elapsed
        latest_elapsed = elapsed
        records.append(record)
        if record.episode % args.report_every == 0 or record.episode == args.episodes:
            previous_steps = sum(item.steps for item in setup.records)
            total_elapsed = base_wall_time + elapsed
            throughput = (previous_steps + total_steps) / total_elapsed if total_elapsed else 0.0
            print(
                f"episode {record.episode}/{args.episodes} | steps {previous_steps + total_steps} | "
                f"{throughput:.1f} steps/s | epsilon {epsilon:.4f}"
            )

    def retain_replay(replay: EvaluationReplay) -> None:
        replays.append(replay)

    def report_evaluation(record: EvaluationRecord) -> None:
        evaluations.append(record)
        print(
            f"evaluation @{record.training_episode} | mean progress "
            f"{record.mean_progress:.3%} | best {record.best_progress:.3%}"
        )
        persist(latest_elapsed)
        best = max(
            evaluations,
            key=lambda value: (value.lap_completions, value.best_progress, value.mean_return),
        )
        if record is best:
            best_path = checkpoint_path.with_name(f"{checkpoint_path.stem}-best{checkpoint_path.suffix}")
            persist(latest_elapsed, best_path)

    try:
        result = run_training(
            RacingEnv(),
            agent,
            args.episodes,
            start_episode=setup.completed_episode + 1,
            evaluate_every=setup.evaluate_every,
            evaluation_episodes=setup.evaluation_episodes,
            evaluation_seed=setup.evaluation_seed,
            record_replays=True,
            on_episode=report_episode,
            on_evaluation=report_evaluation,
            on_replay=retain_replay,
            stop_requested=stop.requested,
            curriculum=curriculum,
        )
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    persist(result.training_wall_time)
    if result.stopped_early:
        print(f"stopped after episode {records[-1].episode}; checkpoint saved to {checkpoint_path}")
        return 130

    total_wall_time = base_wall_time + result.training_wall_time
    summary = summarize_run(records, seed=setup.seed, wall_time=total_wall_time)
    output = {
        "checkpoint": str(checkpoint_path),
        "config": asdict(setup.config),
        "summary": asdict(summary),
        "final_epsilon": agent.epsilon,
        "visited_states": agent.visited_states,
        "evaluations": [asdict(evaluation) for evaluation in evaluations],
        "replays": [
            {
                "training_episode": replay.training_episode,
                "evaluation_episode": replay.evaluation_episode,
                "progress": replay.furthest_progress,
                "return": replay.total_return,
                "termination_reason": replay.termination_reason,
            }
            for replay in replays
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


class _GracefulStop:
    def __init__(self) -> None:
        self._requested = False

    def requested(self) -> bool:
        return self._requested

    def handle_signal(self, signum: int, frame: object) -> None:
        del signum, frame
        if self._requested:
            signal.default_int_handler(signal.SIGINT, None)
        self._requested = True
        print("interrupt requested; finishing the current episode before checkpointing")


class _TrainingSetup:
    def __init__(
        self,
        *,
        seed: int,
        completed_episode: int,
        config: QLearningConfig,
        evaluate_every: int,
        evaluation_episodes: int,
        evaluation_seed: int,
        training_wall_time: float,
        agent: QLearningAgent,
        records: tuple[EpisodeRecord, ...] = (),
        evaluations: tuple[EvaluationRecord, ...] = (),
        replays: tuple[EvaluationReplay, ...] = (),
        curriculum: dict[str, object] | None = None,
    ) -> None:
        self.seed = seed
        self.completed_episode = completed_episode
        self.config = config
        self.evaluate_every = evaluate_every
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_seed = evaluation_seed
        self.training_wall_time = training_wall_time
        self.agent = agent
        self.records = records
        self.evaluations = evaluations
        self.replays = replays
        self.curriculum = curriculum


def _training_setup(args: argparse.Namespace) -> _TrainingSetup:
    if args.resume is None:
        seed = args.seed if args.seed is not None else 0
        config = QLearningConfig(
            learning_rate=_value_or(args.learning_rate, 0.1),
            discount=_value_or(args.discount, 0.9995),
            epsilon_start=_value_or(args.epsilon_start, 1.0),
            epsilon_min=_value_or(args.epsilon_min, 0.05),
            epsilon_decay=_value_or(args.epsilon_decay, 0.997),
            bucket_count=_value_or(args.buckets, 5),
        )
        return _TrainingSetup(
            seed=seed,
            completed_episode=0,
            config=config,
            evaluate_every=_value_or(args.evaluate_every, 50),
            evaluation_episodes=_value_or(args.evaluation_episodes, 10),
            evaluation_seed=seed + 1_000_000,
            training_wall_time=0.0,
            agent=QLearningAgent(Random(seed), config),
        )

    checkpoint = load_checkpoint(args.resume)
    if args.episodes <= checkpoint.completed_episode:
        raise ValueError(
            f"target episodes ({args.episodes}) must exceed checkpoint episode {checkpoint.completed_episode}"
        )
    _match("seed", args.seed, checkpoint.seed)
    _match("evaluate_every", args.evaluate_every, checkpoint.evaluate_every)
    _match("evaluation_episodes", args.evaluation_episodes, checkpoint.evaluation_episodes)
    _match("learning_rate", args.learning_rate, checkpoint.config.learning_rate)
    _match("discount", args.discount, checkpoint.config.discount)
    _match("epsilon_start", args.epsilon_start, checkpoint.config.epsilon_start)
    _match("epsilon_min", args.epsilon_min, checkpoint.config.epsilon_min)
    _match("epsilon_decay", args.epsilon_decay, checkpoint.config.epsilon_decay)
    _match("buckets", args.buckets, checkpoint.config.bucket_count)
    return _setup_from_checkpoint(checkpoint)


def _setup_from_checkpoint(checkpoint: TrainingCheckpoint) -> _TrainingSetup:
    if any(len(state) == 6 for state in checkpoint.q_table):
        raise ValueError("legacy six-value Q-tables can be replayed but cannot resume ten-value training")
    return _TrainingSetup(
        seed=checkpoint.seed,
        completed_episode=checkpoint.completed_episode,
        config=checkpoint.config,
        evaluate_every=checkpoint.evaluate_every,
        evaluation_episodes=checkpoint.evaluation_episodes,
        evaluation_seed=checkpoint.evaluation_seed,
        training_wall_time=checkpoint.training_wall_time,
        agent=checkpoint.restore_agent(),
        records=checkpoint.records,
        evaluations=checkpoint.evaluations,
        replays=checkpoint.replays,
        curriculum=checkpoint.curriculum,
    )


def _match(name: str, supplied: T | None, saved: T) -> None:
    if supplied is not None and supplied != saved:
        raise ValueError(f"--{name.replace('_', '-')}={supplied} conflicts with saved value {saved}")


def _value_or(value: T | None, default: T) -> T:
    return default if value is None else value


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
