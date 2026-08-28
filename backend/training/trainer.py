from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Callable

from backend.rl.agent import GreedyPolicy, QLearningAgent

from .evaluator import EvaluationRecord, evaluate_policy
from .runner import Environment, EpisodeRecord

TrainingProgressCallback = Callable[[EpisodeRecord, int, float, float], None]
EvaluationCallback = Callable[[EvaluationRecord], None]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    records: tuple[EpisodeRecord, ...]
    evaluations: tuple[EvaluationRecord, ...]
    training_wall_time: float


def run_training_episode(
    environment: Environment,
    agent: QLearningAgent,
    episode: int,
) -> EpisodeRecord:
    agent.start_episode(episode)
    observation = environment.reset()
    while True:
        action = agent.choose_action(observation)
        result = environment.step(action)
        agent.update(observation, action, result.reward, result.observation, result.done)
        observation = result.observation
        if result.done:
            return _record_terminal_episode(result.info, episode)


def run_training(
    environment: Environment,
    agent: QLearningAgent,
    episodes: int,
    *,
    evaluate_every: int = 50,
    evaluation_episodes: int = 10,
    evaluation_seed: int = 1_000_000,
    on_episode: TrainingProgressCallback | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> TrainingResult:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if evaluate_every < 1:
        raise ValueError("evaluate_every must be positive")
    if evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be positive")

    records: list[EpisodeRecord] = []
    evaluations: list[EvaluationRecord] = []
    total_steps = 0
    training_wall_time = 0.0

    for episode in range(1, episodes + 1):
        started_at = perf_counter()
        record = run_training_episode(environment, agent, episode)
        training_wall_time += perf_counter() - started_at
        records.append(record)
        total_steps += record.steps
        if on_episode is not None:
            on_episode(record, total_steps, training_wall_time, agent.epsilon)

        if episode % evaluate_every != 0 and episode != episodes:
            continue
        policy = GreedyPolicy(agent, Random(evaluation_seed + episode))
        evaluation = evaluate_policy(
            environment,
            policy,
            evaluation_episodes,
            training_episode=episode,
        )
        evaluations.append(evaluation)
        if on_evaluation is not None:
            on_evaluation(evaluation)

    return TrainingResult(tuple(records), tuple(evaluations), training_wall_time)


def _record_terminal_episode(info: dict[str, object], episode: int) -> EpisodeRecord:
    reason = info["termination_reason"]
    if not isinstance(reason, str):
        raise ValueError("a terminal step must include a termination reason")
    elapsed_time = float(info["elapsed_time"])
    return EpisodeRecord(
        episode=episode,
        steps=int(info["steps"]),
        simulated_duration=elapsed_time,
        total_return=float(info["episode_return"]),
        furthest_progress=float(info["furthest_progress"]),
        termination_reason=reason,
        lap_completed=reason == "lap",
        lap_time=elapsed_time if reason == "lap" else None,
    )
