from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from backend.rl.agent import QLearningAgent, QLearningConfig
from backend.rl.discretisation import DiscreteState
from backend.rl.q_table import QTable

from .evaluator import EvaluationRecord
from .replay import EvaluationReplay, ReplayState, ReplayTransition
from .runner import EpisodeRecord

SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_PATH = Path("artifacts/latest.json")


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    seed: int
    completed_episode: int
    config: QLearningConfig
    evaluate_every: int
    evaluation_episodes: int
    evaluation_seed: int
    training_wall_time: float
    epsilon: float
    rng_state: tuple[object, ...]
    q_table: dict[DiscreteState, tuple[float, ...]]
    records: tuple[EpisodeRecord, ...]
    evaluations: tuple[EvaluationRecord, ...]
    replays: tuple[EvaluationReplay, ...]
    curriculum: dict[str, object] | None = None
    training_steps: int = 0
    epsilon_schedule_start_step: int = 0
    epsilon_schedule_start_value: float | None = None

    def __post_init__(self) -> None:
        if self.completed_episode < 1:
            raise CheckpointError("completed_episode must be positive")
        if self.evaluate_every < 1 or self.evaluation_episodes < 1:
            raise CheckpointError("evaluation settings must be positive")
        if not isfinite(self.training_wall_time) or self.training_wall_time < 0:
            raise CheckpointError("training_wall_time must be finite and non-negative")
        if not isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise CheckpointError("epsilon must be finite and in [0, 1]")
        if self.training_steps < 0 or self.epsilon_schedule_start_step < 0:
            raise CheckpointError("exploration step counters must be non-negative")
        expected_episodes = tuple(range(1, self.completed_episode + 1))
        if tuple(record.episode for record in self.records) != expected_episodes:
            raise CheckpointError("episode history must be complete and contiguous")
        for record in self.records:
            if record.steps < 1 or record.termination_reason not in {"crash", "lap", "timeout", "stalled"}:
                raise CheckpointError("episode history contains an invalid terminal record")
            numeric = (record.simulated_duration, record.total_return, record.furthest_progress)
            if record.lap_time is not None:
                numeric += (record.lap_time,)
            if any(not isfinite(value) for value in numeric):
                raise CheckpointError("episode history must contain finite metrics")
            if record.lap_completed != (record.termination_reason == "lap"):
                raise CheckpointError("episode lap status does not match its termination reason")
        evaluation_numbers = tuple(record.training_episode for record in self.evaluations)
        replay_numbers = tuple(replay.training_episode for replay in self.replays)
        if evaluation_numbers != tuple(sorted(set(evaluation_numbers))):
            raise CheckpointError("evaluation history must be ordered and unique")
        if replay_numbers != evaluation_numbers:
            raise CheckpointError("each evaluation must have exactly one matching replay")
        if evaluation_numbers and evaluation_numbers[-1] > self.completed_episode:
            raise CheckpointError("evaluation history cannot exceed the completed episode")
        for record in self.evaluations:
            if record.episodes < 1 or record.lap_completions < 0:
                raise CheckpointError("evaluation history contains invalid counts")
            if any(
                not isfinite(value)
                for value in (record.mean_return, record.mean_progress, record.best_progress)
            ):
                raise CheckpointError("evaluation history must contain finite metrics")
        _validated_rng_state(self.rng_state)
        QTable.from_snapshot(self.q_table, bucket_count=self.config.bucket_count)

    def restore_agent(self) -> QLearningAgent:
        rng = Random()
        rng.setstate(self.rng_state)
        agent = QLearningAgent(
            rng,
            self.config,
            q_table=QTable.from_snapshot(self.q_table, bucket_count=self.config.bucket_count),
        )
        agent.restore_exploration(
            self.training_steps,
            self.epsilon_schedule_start_step,
            self.epsilon_schedule_start_value,
        )
        return agent


def checkpoint_from_agent(
    agent: QLearningAgent,
    *,
    seed: int,
    completed_episode: int,
    evaluate_every: int,
    evaluation_episodes: int,
    evaluation_seed: int,
    training_wall_time: float,
    records: tuple[EpisodeRecord, ...],
    evaluations: tuple[EvaluationRecord, ...],
    replays: tuple[EvaluationReplay, ...],
    curriculum: dict[str, object] | None = None,
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        seed=seed,
        completed_episode=completed_episode,
        config=agent.config,
        evaluate_every=evaluate_every,
        evaluation_episodes=evaluation_episodes,
        evaluation_seed=evaluation_seed,
        training_wall_time=training_wall_time,
        epsilon=agent.epsilon,
        rng_state=agent.rng.getstate(),
        q_table=agent.q_table.snapshot(),
        records=records,
        evaluations=evaluations,
        replays=replays,
        curriculum=curriculum,
        training_steps=agent.training_steps,
        epsilon_schedule_start_step=agent.epsilon_schedule_start_step,
        epsilon_schedule_start_value=agent.epsilon_schedule_start_value,
    )


def save_checkpoint(checkpoint: TrainingCheckpoint, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint_payload(checkpoint)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path) -> TrainingCheckpoint:
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"unable to read checkpoint {source}: {error}") from error
    try:
        return _checkpoint_from_payload(_mapping(payload, "checkpoint"))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"invalid checkpoint {source}: {error}") from error


def load_checkpoint_replays(path: str | Path) -> tuple[int, str, tuple[EvaluationReplay, ...]]:
    """Load browser replays from either a legacy tabular or v2 learner manifest."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        version = _integer(payload.get("schema_version"), "schema_version")
        if version == SCHEMA_VERSION:
            return version, "tabular", load_checkpoint(source).replays
        if version == 2:
            history = _mapping(payload["history"], "history")
            replays = tuple(
                _evaluation_replay(value) for value in _list(history["replays"], "history.replays")
            )
            return version, str(payload.get("algorithm", "dqn")), replays
        raise CheckpointError(f"unsupported checkpoint schema version {version}")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"invalid checkpoint {source}: {error}") from error


def _checkpoint_payload(checkpoint: TrainingCheckpoint) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "seed": checkpoint.seed,
            "completed_episode": checkpoint.completed_episode,
            "evaluate_every": checkpoint.evaluate_every,
            "evaluation_episodes": checkpoint.evaluation_episodes,
            "evaluation_seed": checkpoint.evaluation_seed,
            "training_wall_time": checkpoint.training_wall_time,
            "curriculum": checkpoint.curriculum,
        },
        "agent": {
            "config": asdict(checkpoint.config),
            "epsilon": checkpoint.epsilon,
            "training_steps": checkpoint.training_steps,
            "epsilon_schedule_start_step": checkpoint.epsilon_schedule_start_step,
            "epsilon_schedule_start_value": checkpoint.epsilon_schedule_start_value,
            "rng_state": checkpoint.rng_state,
            "q_table": [
                {"state": state, "values": values}
                for state, values in sorted(checkpoint.q_table.items())
            ],
        },
        "history": {
            "episodes": [asdict(record) for record in checkpoint.records],
            "evaluations": [asdict(record) for record in checkpoint.evaluations],
            "replays": [asdict(replay) for replay in checkpoint.replays],
        },
    }


def _checkpoint_from_payload(payload: Mapping[str, Any]) -> TrainingCheckpoint:
    version = _integer(payload.get("schema_version"), "schema_version")
    if version != SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema version {version}")
    run = _mapping(payload["run"], "run")
    agent = _mapping(payload["agent"], "agent")
    history = _mapping(payload["history"], "history")
    config = QLearningConfig(**dict(_mapping(agent["config"], "agent.config")))
    q_table: dict[DiscreteState, tuple[float, ...]] = {}
    for item in _list(agent["q_table"], "agent.q_table"):
        row = _mapping(item, "Q-table row")
        state = tuple(_integer(value, "Q-table state value") for value in _list(row["state"], "Q-table state"))
        if state in q_table:
            raise CheckpointError("Q-table contains a duplicate state")
        q_table[state] = tuple(float(value) for value in _list(row["values"], "Q-table values"))
    records = tuple(_episode_record(item) for item in _list(history["episodes"], "history.episodes"))
    evaluations = tuple(
        _evaluation_record(item) for item in _list(history["evaluations"], "history.evaluations")
    )
    replays = tuple(_evaluation_replay(item) for item in _list(history["replays"], "history.replays"))
    return TrainingCheckpoint(
        seed=_integer(run["seed"], "run.seed"),
        completed_episode=_integer(run["completed_episode"], "run.completed_episode"),
        config=config,
        evaluate_every=_integer(run["evaluate_every"], "run.evaluate_every"),
        evaluation_episodes=_integer(run["evaluation_episodes"], "run.evaluation_episodes"),
        evaluation_seed=_integer(run["evaluation_seed"], "run.evaluation_seed"),
        training_wall_time=float(run["training_wall_time"]),
        epsilon=float(agent["epsilon"]),
        rng_state=_nested_tuple(agent["rng_state"]),
        q_table=q_table,
        records=records,
        evaluations=evaluations,
        replays=replays,
        curriculum=_curriculum_snapshot(run.get("curriculum")),
        training_steps=_integer(agent.get("training_steps", sum(record.steps for record in records)), "training_steps"),
        epsilon_schedule_start_step=_integer(
            agent.get("epsilon_schedule_start_step", 0), "epsilon_schedule_start_step"
        ),
        epsilon_schedule_start_value=float(
            agent.get("epsilon_schedule_start_value", config.epsilon_start)
        ),
    )


def _curriculum_snapshot(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    item = dict(_mapping(value, "run.curriculum"))
    if "rng_state" in item:
        item["rng_state"] = _nested_tuple(item["rng_state"])
    return item


def _episode_record(value: object) -> EpisodeRecord:
    item = _mapping(value, "episode record")
    return EpisodeRecord(
        episode=_integer(item["episode"], "episode"),
        steps=_integer(item["steps"], "steps"),
        simulated_duration=float(item["simulated_duration"]),
        total_return=float(item["total_return"]),
        furthest_progress=float(item["furthest_progress"]),
        termination_reason=str(item["termination_reason"]),
        lap_completed=_boolean(item["lap_completed"], "lap_completed"),
        lap_time=None if item["lap_time"] is None else float(item["lap_time"]),
    )


def _evaluation_record(value: object) -> EvaluationRecord:
    item = _mapping(value, "evaluation record")
    return EvaluationRecord(
        training_episode=_integer(item["training_episode"], "training_episode"),
        episodes=_integer(item["episodes"], "episodes"),
        mean_return=float(item["mean_return"]),
        mean_progress=float(item["mean_progress"]),
        best_progress=float(item["best_progress"]),
        lap_completions=_integer(item["lap_completions"], "lap_completions"),
    )


def _evaluation_replay(value: object) -> EvaluationReplay:
    item = _mapping(value, "evaluation replay")
    return EvaluationReplay(
        training_episode=_integer(item["training_episode"], "training_episode"),
        evaluation_episode=_integer(item["evaluation_episode"], "evaluation_episode"),
        total_return=float(item["total_return"]),
        furthest_progress=float(item["furthest_progress"]),
        simulated_duration=float(item["simulated_duration"]),
        steps=_integer(item["steps"], "steps"),
        termination_reason=str(item["termination_reason"]),
        lap_completed=_boolean(item["lap_completed"], "lap_completed"),
        initial_state=_replay_state(item["initial_state"]),
        transitions=tuple(
            _replay_transition(transition)
            for transition in _list(item["transitions"], "replay transitions")
        ),
    )


def _replay_transition(value: object) -> ReplayTransition:
    item = _mapping(value, "replay transition")
    return ReplayTransition(
        action=_integer(item["action"], "action"),
        action_name=str(item["action_name"]),
        q_values=tuple(float(value) for value in _list(item["q_values"], "q_values")),
        reward=float(item["reward"]),
        state=_replay_state(item["state"]),
    )


def _replay_state(value: object) -> ReplayState:
    item = _mapping(value, "replay state")
    return ReplayState(
        tick=_integer(item["tick"], "tick"),
        x=float(item["x"]),
        y=float(item["y"]),
        heading=float(item["heading"]),
        speed=float(item["speed"]),
        crashed=_boolean(item["crashed"], "crashed"),
        sensors=tuple(float(value) for value in _list(item["sensors"], "sensors")),
        current_progress=float(item["current_progress"]),
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CheckpointError(f"{name} must be a list")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CheckpointError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise CheckpointError(f"{name} must be a boolean")
    return value


def _nested_tuple(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise CheckpointError("agent.rng_state must be an encoded tuple")
    return tuple(_nested_tuple(item) if isinstance(item, list) else item for item in value)


def _validated_rng_state(value: tuple[object, ...]) -> None:
    try:
        Random().setstate(value)
    except (TypeError, ValueError) as error:
        raise CheckpointError("agent RNG state is invalid") from error
