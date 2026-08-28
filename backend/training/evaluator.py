from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from .agents import Agent
from .replay import EvaluationReplay, ReplayEnvironment, ReplayPolicy, record_evaluation_episode, select_best_replay
from .runner import Environment, run_episodes


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    training_episode: int
    episodes: int
    mean_return: float
    mean_progress: float
    best_progress: float
    lap_completions: int


def evaluate_policy(
    environment: Environment,
    policy: Agent,
    episodes: int,
    *,
    training_episode: int,
) -> EvaluationRecord:
    records, _ = run_episodes(environment, policy, episodes)
    return EvaluationRecord(
        training_episode=training_episode,
        episodes=episodes,
        mean_return=fmean(record.total_return for record in records),
        mean_progress=fmean(record.furthest_progress for record in records),
        best_progress=max(record.furthest_progress for record in records),
        lap_completions=sum(record.lap_completed for record in records),
    )


def evaluate_policy_with_replay(
    environment: ReplayEnvironment,
    policy: ReplayPolicy,
    episodes: int,
    *,
    training_episode: int,
) -> tuple[EvaluationRecord, EvaluationReplay]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    records = []
    replays = []
    for evaluation_episode in range(1, episodes + 1):
        record, replay = record_evaluation_episode(
            environment,
            policy,
            training_episode=training_episode,
            evaluation_episode=evaluation_episode,
        )
        records.append(record)
        replays.append(replay)
    evaluation = EvaluationRecord(
        training_episode=training_episode,
        episodes=episodes,
        mean_return=fmean(record.total_return for record in records),
        mean_progress=fmean(record.furthest_progress for record in records),
        best_progress=max(record.furthest_progress for record in records),
        lap_completions=sum(record.lap_completed for record in records),
    )
    return evaluation, select_best_replay(replays)
