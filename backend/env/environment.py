from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

from .geometry import Point
from .simulation import Action, DT, MAX_SPEED, RacingSimulation
from .track import TRACK, Gate, Track, progress_gates

Observation: TypeAlias = tuple[float, ...]


class DiscreteAction(IntEnum):
    COAST = 0
    THROTTLE = 1
    BRAKE = 2
    LEFT = 3
    LEFT_THROTTLE = 4
    LEFT_BRAKE = 5
    RIGHT = 6
    RIGHT_THROTTLE = 7
    RIGHT_BRAKE = 8

    def controls(self) -> Action:
        throttle, brake, left, right = _ACTION_CONTROLS[self]
        return Action(throttle=throttle, brake=brake, left=left, right=right)


_ACTION_CONTROLS = {
    DiscreteAction.COAST: (False, False, False, False),
    DiscreteAction.THROTTLE: (True, False, False, False),
    DiscreteAction.BRAKE: (False, True, False, False),
    DiscreteAction.LEFT: (False, False, True, False),
    DiscreteAction.LEFT_THROTTLE: (True, False, True, False),
    DiscreteAction.LEFT_BRAKE: (False, True, True, False),
    DiscreteAction.RIGHT: (False, False, False, True),
    DiscreteAction.RIGHT_THROTTLE: (True, False, False, True),
    DiscreteAction.RIGHT_BRAKE: (False, True, False, True),
}


@dataclass(frozen=True, slots=True)
class RewardConfig:
    step: float = -0.01
    checkpoint: float = 1.0
    reverse_checkpoint: float = -1.0
    crash: float = -10.0
    lap: float = 50.0


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, object]


class RacingEnv:
    def __init__(
        self,
        track: Track = TRACK,
        *,
        rewards: RewardConfig = RewardConfig(),
        max_steps: int = 1_200,
        checkpoint_count: int = 100,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if checkpoint_count < 2:
            raise ValueError("checkpoint_count must be at least 2")
        self.track = track
        self.rewards = rewards
        self.max_steps = max_steps
        self.checkpoint_count = checkpoint_count
        self.checkpoints: tuple[Gate, ...] = progress_gates(track, checkpoint_count)
        self.simulation = RacingSimulation(track)
        self._reset_episode_state()

    def reset(self) -> Observation:
        snapshot = self.simulation.reset()
        self._reset_episode_state()
        return _observation_from_snapshot(snapshot)

    def step(self, action: DiscreteAction) -> StepResult:
        if self.done:
            raise RuntimeError("reset() must be called before stepping a finished episode")
        if not isinstance(action, DiscreteAction):
            raise TypeError("action must be a DiscreteAction")

        previous_position = self.simulation.car.position
        snapshot = self.simulation.step(action.controls())
        current_position = self.simulation.car.position
        self.steps += 1
        reward = self.rewards.step
        termination_reason: str | None = None

        if self.simulation.crashed:
            reward += self.rewards.crash
            termination_reason = "crash"
        else:
            reward += self._update_progress(previous_position, current_position)
            if self.current_checkpoint == self.checkpoint_count - 1 and self.track.finish_gate.contains_crossing(
                previous_position, current_position, self.track.half_width
            ):
                self.current_checkpoint = self.checkpoint_count
                self.furthest_checkpoint = self.checkpoint_count
                reward += self.rewards.checkpoint + self.rewards.lap
                termination_reason = "lap"
            elif self.steps >= self.max_steps:
                termination_reason = "timeout"

        self.episode_return += reward
        self.done = termination_reason is not None
        self.termination_reason = termination_reason
        return StepResult(
            observation=_observation_from_snapshot(snapshot),
            reward=reward,
            done=self.done,
            info=self._info(),
        )

    def _reset_episode_state(self) -> None:
        self.steps = 0
        self.current_checkpoint = 0
        self.furthest_checkpoint = 0
        self.episode_return = 0.0
        self.done = False
        self.termination_reason: str | None = None

    def _update_progress(self, previous: Point, current: Point) -> float:
        if self.current_checkpoint < len(self.checkpoints):
            next_gate = self.checkpoints[self.current_checkpoint]
            if next_gate.contains_crossing(previous, current, self.track.half_width):
                self.current_checkpoint += 1
                self.furthest_checkpoint = max(self.furthest_checkpoint, self.current_checkpoint)
                return self.rewards.checkpoint

        if self.current_checkpoint > 0:
            previous_gate = self.checkpoints[self.current_checkpoint - 1]
            if previous_gate.contains_reverse_crossing(previous, current, self.track.half_width):
                self.current_checkpoint -= 1
                return self.rewards.reverse_checkpoint
        return 0.0

    def _info(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "elapsed_time": self.steps * DT,
            "current_progress": self.current_checkpoint / self.checkpoint_count,
            "furthest_progress": self.furthest_checkpoint / self.checkpoint_count,
            "episode_return": self.episode_return,
            "termination_reason": self.termination_reason,
        }


def _observation_from_snapshot(snapshot: dict[str, object]) -> Observation:
    sensors = snapshot["sensors"]
    if not isinstance(sensors, list) or len(sensors) != 5:
        raise ValueError("simulation snapshot must contain five sensor readings")
    return (*(float(value) for value in sensors), float(snapshot["speed"]) / MAX_SPEED)
