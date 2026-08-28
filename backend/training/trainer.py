from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Callable

from backend.rl.agent import GreedyPolicy, QLearningAgent

from .evaluator import EvaluationRecord, evaluate_policy, evaluate_policy_with_replay
from .replay import EvaluationReplay, ReplayEnvironment
from .runner import Environment, EpisodeRecord
from .curriculum import AdaptiveCurriculum, CurriculumEnvironment

TrainingProgressCallback = Callable[[EpisodeRecord, int, float, float], None]
EvaluationCallback = Callable[[EvaluationRecord], None]
ReplayCallback = Callable[[EvaluationReplay], None]
StopRequested = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    records: tuple[EpisodeRecord, ...]
    evaluations: tuple[EvaluationRecord, ...]
    replays: tuple[EvaluationReplay, ...]
    training_wall_time: float
    stopped_early: bool = False


def run_training_episode(
    environment: Environment,
    agent: QLearningAgent,
    episode: int,
    curriculum: AdaptiveCurriculum | None = None,
) -> EpisodeRecord:
    agent.start_episode(episode)
    if curriculum is not None:
        if not isinstance(environment, CurriculumEnvironment):
            raise TypeError("curriculum requires reset_at_progress()")
        observation = curriculum.reset(environment)
    else:
        observation = environment.reset()
    while True:
        action = agent.choose_action(observation)
        result = environment.step(action)
        agent.update(observation, action, result.reward, result.observation, result.done)
        observation = result.observation
        if result.done:
            record = _record_terminal_episode(result.info, episode)
            if curriculum is not None:
                curriculum.observe(record.lap_completed)
            return record


def run_training(
    environment: Environment,
    agent: QLearningAgent,
    episodes: int,
    *,
    evaluate_every: int = 50,
    evaluation_episodes: int = 10,
    evaluation_seed: int = 1_000_000,
    start_episode: int = 1,
    record_replays: bool = False,
    on_episode: TrainingProgressCallback | None = None,
    on_evaluation: EvaluationCallback | None = None,
    on_replay: ReplayCallback | None = None,
    stop_requested: StopRequested | None = None,
    curriculum: AdaptiveCurriculum | None = None,
) -> TrainingResult:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if evaluate_every < 1:
        raise ValueError("evaluate_every must be positive")
    if evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be positive")
    if start_episode < 1 or start_episode > episodes:
        raise ValueError("start_episode must be between 1 and the target episode")

    records: list[EpisodeRecord] = []
    evaluations: list[EvaluationRecord] = []
    replays: list[EvaluationReplay] = []
    total_steps = 0
    training_wall_time = 0.0

    stopped_early = False
    for episode in range(start_episode, episodes + 1):
        started_at = perf_counter()
        record = run_training_episode(environment, agent, episode, curriculum)
        training_wall_time += perf_counter() - started_at
        records.append(record)
        total_steps += record.steps
        if on_episode is not None:
            on_episode(record, total_steps, training_wall_time, agent.epsilon)

        if stop_requested is not None and stop_requested():
            stopped_early = True
            break

        if episode % evaluate_every != 0 and episode != episodes:
            continue
        policy = GreedyPolicy(agent, Random(evaluation_seed + episode))
        if record_replays:
            if not isinstance(environment, ReplayEnvironment):
                raise TypeError("record_replays requires an environment with render_state()")
            evaluation, replay = evaluate_policy_with_replay(
                environment,
                policy,
                evaluation_episodes,
                training_episode=episode,
            )
            replays.append(replay)
            if on_replay is not None:
                on_replay(replay)
        else:
            evaluation = evaluate_policy(
                environment,
                policy,
                evaluation_episodes,
                training_episode=episode,
            )
        evaluations.append(evaluation)
        if on_evaluation is not None:
            on_evaluation(evaluation)
        if stop_requested is not None and stop_requested():
            stopped_early = True
            break

    return TrainingResult(
        tuple(records),
        tuple(evaluations),
        tuple(replays),
        training_wall_time,
        stopped_early,
    )


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
