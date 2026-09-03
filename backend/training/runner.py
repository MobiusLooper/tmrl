from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from time import perf_counter
from typing import Callable, Protocol

from backend.env.environment import DiscreteAction, Observation, StepResult

from .agents import Agent


class Environment(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: DiscreteAction) -> StepResult: ...


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode: int
    steps: int
    simulated_duration: float
    total_return: float
    furthest_progress: float
    termination_reason: str
    lap_completed: bool
    lap_time: float | None


@dataclass(frozen=True, slots=True)
class RunSummary:
    seed: int
    episodes: int
    total_steps: int
    wall_time: float
    steps_per_second: float
    mean_return: float
    mean_progress: float
    best_progress: float
    lap_completions: int
    crash_count: int
    timeout_count: int
    stalled_count: int


ProgressCallback = Callable[[EpisodeRecord, int, float], None]


def run_episode(
    environment: Environment,
    agent: Agent,
    episode: int,
    *,
    action_repeat: int = 1,
) -> EpisodeRecord:
    if action_repeat < 1:
        raise ValueError("action_repeat must be positive")
    start_episode = getattr(agent, "start_episode", None)
    if callable(start_episode):
        start_episode()
    observation = environment.reset()
    while True:
        action = agent.choose_action(observation)
        for _ in range(action_repeat):
            result = environment.step(action)
            observation = result.observation
            if result.done:
                break
        if not result.done:
            continue

        reason = result.info["termination_reason"]
        if not isinstance(reason, str):
            raise ValueError("a terminal step must include a termination reason")
        elapsed_time = float(result.info["elapsed_time"])
        return EpisodeRecord(
            episode=episode,
            steps=int(result.info["steps"]),
            simulated_duration=elapsed_time,
            total_return=float(result.info["episode_return"]),
            furthest_progress=float(result.info["furthest_progress"]),
            termination_reason=reason,
            lap_completed=reason == "lap",
            lap_time=elapsed_time if reason == "lap" else None,
        )


def run_episodes(
    environment: Environment,
    agent: Agent,
    episodes: int,
    *,
    on_episode: ProgressCallback | None = None,
    action_repeat: int = 1,
) -> tuple[list[EpisodeRecord], float]:
    if episodes < 1:
        raise ValueError("episodes must be positive")

    records: list[EpisodeRecord] = []
    total_steps = 0
    started_at = perf_counter()
    for episode in range(1, episodes + 1):
        record = run_episode(environment, agent, episode, action_repeat=action_repeat)
        records.append(record)
        total_steps += record.steps
        if on_episode is not None:
            on_episode(record, total_steps, perf_counter() - started_at)
    return records, perf_counter() - started_at


def summarize_run(records: list[EpisodeRecord], *, seed: int, wall_time: float) -> RunSummary:
    if not records:
        raise ValueError("at least one episode record is required")
    if wall_time < 0:
        raise ValueError("wall_time cannot be negative")

    total_steps = sum(record.steps for record in records)
    return RunSummary(
        seed=seed,
        episodes=len(records),
        total_steps=total_steps,
        wall_time=wall_time,
        steps_per_second=total_steps / wall_time if wall_time else 0.0,
        mean_return=fmean(record.total_return for record in records),
        mean_progress=fmean(record.furthest_progress for record in records),
        best_progress=max(record.furthest_progress for record in records),
        lap_completions=sum(record.lap_completed for record in records),
        crash_count=sum(record.termination_reason == "crash" for record in records),
        timeout_count=sum(record.termination_reason == "timeout" for record in records),
        stalled_count=sum(record.termination_reason == "stalled" for record in records),
    )
