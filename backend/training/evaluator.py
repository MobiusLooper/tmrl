from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from .agents import Agent
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
