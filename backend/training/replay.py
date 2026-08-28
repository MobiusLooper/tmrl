from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

from backend.env.environment import DiscreteAction, Observation, StepResult

from .runner import EpisodeRecord


@runtime_checkable
class ReplayEnvironment(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: DiscreteAction) -> StepResult: ...

    def render_state(self) -> dict[str, object]: ...


class ReplayPolicy(Protocol):
    def choose_action(self, observation: Observation) -> DiscreteAction: ...

    def q_values(self, observation: Observation) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class ReplayState:
    tick: int
    x: float
    y: float
    heading: float
    speed: float
    crashed: bool
    sensors: tuple[float, ...]
    current_progress: float

    def __post_init__(self) -> None:
        numeric = (self.x, self.y, self.heading, self.speed, self.current_progress, *self.sensors)
        if self.tick < 0 or len(self.sensors) != 5 or any(not isfinite(value) for value in numeric):
            raise ValueError("replay states must contain a valid tick and finite physical values")


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    action: int
    action_name: str
    q_values: tuple[float, ...]
    reward: float
    state: ReplayState

    def __post_init__(self) -> None:
        if self.action not in range(len(DiscreteAction)):
            raise ValueError("replay action is outside the discrete action space")
        if self.action_name != DiscreteAction(self.action).name:
            raise ValueError("replay action name does not match its value")
        if len(self.q_values) != len(DiscreteAction) or any(not isfinite(value) for value in self.q_values):
            raise ValueError("replay transitions must contain nine finite Q-values")
        if not isfinite(self.reward):
            raise ValueError("replay rewards must be finite")


@dataclass(frozen=True, slots=True)
class EvaluationReplay:
    training_episode: int
    evaluation_episode: int
    total_return: float
    furthest_progress: float
    simulated_duration: float
    steps: int
    termination_reason: str
    lap_completed: bool
    initial_state: ReplayState
    transitions: tuple[ReplayTransition, ...]

    def __post_init__(self) -> None:
        if self.training_episode < 1 or self.evaluation_episode < 1 or self.steps < 1:
            raise ValueError("replay episode numbers and step count must be positive")
        if self.steps != len(self.transitions):
            raise ValueError("replay step count must match its transitions")
        if self.termination_reason not in {"crash", "lap", "timeout", "stalled"}:
            raise ValueError("replay has an invalid termination reason")
        if self.lap_completed != (self.termination_reason == "lap"):
            raise ValueError("replay lap status does not match its termination reason")
        if any(not isfinite(value) for value in (self.total_return, self.furthest_progress, self.simulated_duration)):
            raise ValueError("replay metadata must be finite")


@dataclass(frozen=True, slots=True)
class SteeringMetrics:
    changes: int
    direct_reversals: int
    changes_per_second: float
    direct_reversals_per_second: float


def record_evaluation_episode(
    environment: ReplayEnvironment,
    policy: ReplayPolicy,
    *,
    training_episode: int,
    evaluation_episode: int,
    action_repeat: int = 1,
) -> tuple[EpisodeRecord, EvaluationReplay]:
    if action_repeat < 1:
        raise ValueError("action_repeat must be positive")
    start_episode = getattr(policy, "start_episode", None)
    if callable(start_episode):
        start_episode()
    observation = environment.reset()
    initial_state = replay_state(environment.render_state())
    transitions: list[ReplayTransition] = []
    while True:
        q_values = policy.q_values(observation)
        action = policy.choose_action(observation)
        for _ in range(action_repeat):
            result = environment.step(action)
            transitions.append(
                ReplayTransition(
                    action=int(action),
                    action_name=action.name,
                    q_values=q_values,
                    reward=result.reward,
                    state=replay_state(environment.render_state()),
                )
            )
            observation = result.observation
            if result.done:
                break
        if not result.done:
            continue
        record = _episode_record(result.info, evaluation_episode)
        return record, EvaluationReplay(
            training_episode=training_episode,
            evaluation_episode=evaluation_episode,
            total_return=record.total_return,
            furthest_progress=record.furthest_progress,
            simulated_duration=record.simulated_duration,
            steps=record.steps,
            termination_reason=record.termination_reason,
            lap_completed=record.lap_completed,
            initial_state=initial_state,
            transitions=tuple(transitions),
        )


def select_best_replay(replays: list[EvaluationReplay]) -> EvaluationReplay:
    if not replays:
        raise ValueError("at least one replay is required")
    return max(
        replays,
        key=lambda replay: (
            replay.furthest_progress,
            replay.total_return,
            -replay.simulated_duration,
            -replay.evaluation_episode,
        ),
    )


def steering_metrics(replay: EvaluationReplay) -> SteeringMetrics:
    previous = 0
    changes = 0
    reversals = 0
    for transition in replay.transitions:
        current = _steering_state(DiscreteAction(transition.action))
        if current != previous:
            changes += 1
            if current * previous == -1:
                reversals += 1
        previous = current
    duration = replay.simulated_duration
    return SteeringMetrics(
        changes=changes,
        direct_reversals=reversals,
        changes_per_second=changes / duration if duration else 0.0,
        direct_reversals_per_second=reversals / duration if duration else 0.0,
    )


def replay_state(snapshot: dict[str, object]) -> ReplayState:
    sensors = snapshot.get("sensors")
    if not isinstance(sensors, (list, tuple)) or len(sensors) != 5:
        raise ValueError("render state must contain five sensors")
    return ReplayState(
        tick=_integer(snapshot.get("tick"), "tick"),
        x=float(snapshot["x"]),
        y=float(snapshot["y"]),
        heading=float(snapshot["heading"]),
        speed=float(snapshot["speed"]),
        crashed=snapshot.get("crashed") is True,
        sensors=tuple(float(value) for value in sensors),
        current_progress=float(snapshot.get("current_progress", 0.0)),
    )


def _episode_record(info: dict[str, object], episode: int) -> EpisodeRecord:
    reason = info.get("termination_reason")
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


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"render state {name} must be an integer")
    return value


def _steering_state(action: DiscreteAction) -> int:
    if action in {
        DiscreteAction.LEFT,
        DiscreteAction.LEFT_THROTTLE,
        DiscreteAction.LEFT_BRAKE,
    }:
        return -1
    if action in {
        DiscreteAction.RIGHT,
        DiscreteAction.RIGHT_THROTTLE,
        DiscreteAction.RIGHT_BRAKE,
    }:
        return 1
    return 0
