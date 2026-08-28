from __future__ import annotations

import argparse
import json
import signal
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Sequence, TypeVar

from backend.env.environment import RacingEnv
from backend.env.simulation import DT
from backend.rl.agent import TABULAR_ARCHITECTURE, QLearningAgent, QLearningConfig

from .checkpoint import (
    CheckpointError,
    TrainingCheckpoint,
    checkpoint_from_agent,
    load_checkpoint,
    save_checkpoint,
)
from .curriculum import AdaptiveCurriculum, CurriculumConfig
from .evaluator import EvaluationRecord
from .replay import EvaluationReplay, steering_metrics
from .runner import EpisodeRecord, summarize_run
from .run_storage import create_run, new_run_metadata, resume_run_metadata
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
    parser.add_argument("--epsilon-decay-steps", type=_positive_int)
    parser.add_argument("--epsilon-reheat", type=float)
    parser.add_argument("--buckets", type=_bucket_count)
    parser.add_argument("--action-repeat", type=_positive_int)
    parser.add_argument("--sticky-tolerance", type=_non_negative_float)
    parser.add_argument(
        "--canonical-start-probability",
        "--canonical-probability",
        dest="canonical_start_probability",
        type=_probability,
    )
    parser.add_argument(
        "--tabular-stall-seconds",
        "--stall-seconds",
        dest="tabular_stall_seconds",
        type=_positive_float,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="custom checkpoint output path (default: a timestamped artifacts/runs directory)",
    )
    parser.add_argument("--resume", type=Path, help="resume a saved checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is not None and args.checkpoint is not None:
        raise SystemExit("--checkpoint cannot be combined with --resume")
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
        else _new_curriculum(setup.seed, setup.config)
    )
    base_wall_time = setup.training_wall_time
    if args.resume is not None:
        checkpoint_path = args.resume
        run_metadata = resume_run_metadata(
            checkpoint_path,
            run_id=setup.run_id,
            created_at=setup.created_at,
        )
    elif args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        run_metadata = new_run_metadata("tabular-smooth", setup.seed)
    else:
        run_metadata, checkpoint_path = create_run("tabular-smooth", setup.seed)
    print(f"run {run_metadata.run_id} | checkpoint {checkpoint_path}")
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
            run_metadata=run_metadata.touch(),
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
                f"{throughput:.1f} physical steps/s | decisions {agent.training_steps} | "
                f"epsilon {epsilon:.4f}"
            )

    def retain_replay(replay: EvaluationReplay) -> None:
        replays.append(replay)

    def report_evaluation(record: EvaluationRecord) -> None:
        evaluations.append(record)
        print(
            f"evaluation @{record.training_episode} | mean progress "
            f"{record.mean_progress:.3%} | best {record.best_progress:.3%} | "
            f"laps {record.lap_completions} | crashes {record.crash_count / record.episodes:.0%} | "
            f"stalls {record.stalled_count / record.episodes:.0%} | "
            f"timeouts {record.timeout_count / record.episodes:.0%}"
        )
        persist(latest_elapsed)
        best = max(
            evaluations,
            key=lambda value: (value.lap_completions, value.best_progress, value.mean_return),
        )
        if record is best:
            best_path = (
                checkpoint_path.with_name("best.json")
                if checkpoint_path.name == "checkpoint.json"
                else checkpoint_path.with_name(f"{checkpoint_path.stem}-best{checkpoint_path.suffix}")
            )
            persist(latest_elapsed, best_path)

    try:
        result = run_training(
            RacingEnv(stall_steps=_stall_steps(setup.config)),
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
        "run_id": run_metadata.run_id,
        "checkpoint": str(checkpoint_path),
        "config": asdict(setup.config),
        "summary": asdict(summary),
        "final_epsilon": agent.epsilon,
        "tabular_decisions": agent.training_steps,
        "visited_states": agent.visited_states,
        "evaluations": [asdict(evaluation) for evaluation in evaluations],
        "replays": [
            {
                "training_episode": replay.training_episode,
                "evaluation_episode": replay.evaluation_episode,
                "progress": replay.furthest_progress,
                "return": replay.total_return,
                "termination_reason": replay.termination_reason,
                "steering_changes_per_second": steering_metrics(replay).changes_per_second,
                "direct_steering_reversals_per_second": (
                    steering_metrics(replay).direct_reversals_per_second
                ),
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
        run_id: str | None = None,
        created_at: str | None = None,
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
        self.run_id = run_id
        self.created_at = created_at


def _training_setup(args: argparse.Namespace) -> _TrainingSetup:
    if args.resume is None:
        seed = args.seed if args.seed is not None else 0
        config = QLearningConfig(
            learning_rate=_value_or(args.learning_rate, 0.1),
            discount=_value_or(args.discount, 0.9995),
            epsilon_start=_value_or(args.epsilon_start, 1.0),
            epsilon_min=_value_or(args.epsilon_min, 0.10),
            epsilon_decay=_value_or(args.epsilon_decay, 0.997),
            epsilon_decay_steps=_value_or(args.epsilon_decay_steps, 200_000),
            epsilon_reheat=_value_or(args.epsilon_reheat, 0.30),
            bucket_count=_value_or(args.buckets, 5),
            action_repeat=_value_or(args.action_repeat, 2),
            sticky_tolerance=_value_or(args.sticky_tolerance, 0.03),
            canonical_start_probability=_value_or(args.canonical_start_probability, 0.50),
            tabular_stall_seconds=_value_or(args.tabular_stall_seconds, 10.0),
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
    _match("epsilon_decay_steps", args.epsilon_decay_steps, checkpoint.config.epsilon_decay_steps)
    _match("epsilon_reheat", args.epsilon_reheat, checkpoint.config.epsilon_reheat)
    _match("buckets", args.buckets, checkpoint.config.bucket_count)
    _match("action_repeat", args.action_repeat, checkpoint.config.action_repeat)
    _match("sticky_tolerance", args.sticky_tolerance, checkpoint.config.sticky_tolerance)
    _match(
        "canonical_start_probability",
        args.canonical_start_probability,
        checkpoint.config.canonical_start_probability,
    )
    _match(
        "tabular_stall_seconds",
        args.tabular_stall_seconds,
        checkpoint.config.tabular_stall_seconds,
    )
    return _setup_from_checkpoint(checkpoint)


def _setup_from_checkpoint(checkpoint: TrainingCheckpoint) -> _TrainingSetup:
    if checkpoint.config.architecture != TABULAR_ARCHITECTURE or any(
        len(state) != 7 for state in checkpoint.q_table
    ):
        raise ValueError(
            "this checkpoint uses a previous tabular architecture; it remains replayable, "
            "but smooth tabular training must start as a fresh run without --resume"
        )
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
        run_id=checkpoint.run_id,
        created_at=checkpoint.created_at,
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
    if parsed != 5:
        raise argparse.ArgumentTypeError("the smooth tabular architecture requires 5")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _new_curriculum(seed: int, config: QLearningConfig) -> AdaptiveCurriculum:
    return AdaptiveCurriculum(
        Random(seed + 2_000_000),
        CurriculumConfig(
            canonical_probability=config.canonical_start_probability,
            bounded_stages=True,
        ),
    )


def _stall_steps(config: QLearningConfig) -> int:
    return max(1, round(config.tabular_stall_seconds / DT))


if __name__ == "__main__":
    raise SystemExit(main())
