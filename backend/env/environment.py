from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import atan2, pi, sin, cos
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
    pace: float = 1.0
    checkpoint_speed: float = 0.25
    reverse_checkpoint: float = -1.0
    crash: float = -15.0
    timeout: float = -15.0
    stalled: float = -15.0
    lap: float = 100.0
    target_lap_time: float = 40.0
    start_allowance: float = 2.0


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
        stall_steps: int = 100,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if checkpoint_count < 2:
            raise ValueError("checkpoint_count must be at least 2")
        if stall_steps < 1:
            raise ValueError("stall_steps must be positive")
        self.track = track
        self.rewards = rewards
        self.max_steps = max_steps
        self.checkpoint_count = checkpoint_count
        self.stall_steps = stall_steps
        self.checkpoints: tuple[Gate, ...] = progress_gates(track, checkpoint_count)
        self.simulation = RacingSimulation(track)
        self._reset_episode_state()

    def reset(self) -> Observation:
        snapshot = self.simulation.reset()
        self._reset_episode_state()
        return self._observation(snapshot)

    def reset_at_progress(self, progress: float, *, speed: float = 3.0) -> Observation:
        """Reset at an ordered gate for curriculum training."""
        if not 0 < progress < 1:
            raise ValueError("curriculum progress must be between zero and one")
        checkpoint = min(self.checkpoint_count - 1, max(1, round(progress * self.checkpoint_count)))
        gate = self.checkpoints[checkpoint - 1]
        snapshot = self.simulation.reset_pose(gate.center, atan2(gate.tangent.y, gate.tangent.x), speed)
        self._reset_episode_state()
        self.current_checkpoint = checkpoint
        self.furthest_checkpoint = checkpoint
        return self._observation(snapshot)

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
            progress_reward, advanced = self._update_progress(previous_position, current_position)
            reward += progress_reward
            self._steps_since_progress = 0 if advanced else self._steps_since_progress + 1
            if self.current_checkpoint == self.checkpoint_count - 1 and self.track.finish_gate.contains_crossing(
                previous_position, current_position, self.track.half_width
            ):
                self.current_checkpoint = self.checkpoint_count
                self.furthest_checkpoint = self.checkpoint_count
                reward += self._new_progress_reward(1.0) + self.rewards.lap
                termination_reason = "lap"
            elif self.steps >= self.max_steps:
                reward += self.rewards.timeout
                termination_reason = "timeout"
            elif self._steps_since_progress >= self.stall_steps:
                reward += self.rewards.stalled
                termination_reason = "stalled"

        self.episode_return += reward
        self.done = termination_reason is not None
        self.termination_reason = termination_reason
        return StepResult(
            observation=self._observation(snapshot),
            reward=reward,
            done=self.done,
            info=self._info(),
        )

    def render_state(self) -> dict[str, object]:
        """Return the browser-facing physical state for replay recording."""
        return {
            **self.simulation.snapshot(),
            "current_progress": self.current_checkpoint / self.checkpoint_count,
        }

    def _reset_episode_state(self) -> None:
        self.steps = 0
        self.current_checkpoint = 0
        self.furthest_checkpoint = 0
        self.episode_return = 0.0
        self.done = False
        self.termination_reason: str | None = None
        self._steps_since_progress = 0

    def _update_progress(self, previous: Point, current: Point) -> tuple[float, bool]:
        if self.current_checkpoint < len(self.checkpoints):
            next_gate = self.checkpoints[self.current_checkpoint]
            if next_gate.contains_crossing(previous, current, self.track.half_width):
                self.current_checkpoint += 1
                if self.current_checkpoint > self.furthest_checkpoint:
                    self.furthest_checkpoint = self.current_checkpoint
                    return self._new_progress_reward(self.current_checkpoint / self.checkpoint_count), True
                return 0.0, False

        if self.current_checkpoint > 0:
            previous_gate = self.checkpoints[self.current_checkpoint - 1]
            if previous_gate.contains_reverse_crossing(previous, current, self.track.half_width):
                self.current_checkpoint -= 1
                return self.rewards.reverse_checkpoint, False
        return 0.0, False

    def _new_progress_reward(self, progress: float) -> float:
        deadline = self.rewards.start_allowance + (
            self.rewards.target_lap_time - self.rewards.start_allowance
        ) * progress
        elapsed = self.steps * DT
        pace = max(0.0, min(1.0, (deadline - elapsed) / deadline))
        speed = self.simulation.car.speed / MAX_SPEED
        return self.rewards.checkpoint + self.rewards.pace * pace + self.rewards.checkpoint_speed * speed

    def _observation(self, snapshot: dict[str, object]) -> Observation:
        sensors = snapshot["sensors"]
        if not isinstance(sensors, list) or len(sensors) != 5:
            raise ValueError("simulation snapshot must contain five sensor readings")
        gate = self.track.finish_gate if self.current_checkpoint >= len(self.checkpoints) else self.checkpoints[self.current_checkpoint]
        normal = Point(-gate.tangent.y, gate.tangent.x)
        lateral = max(
            -1.0,
            min(1.0, (self.simulation.car.position - gate.center).dot(normal) / self.track.half_width),
        )
        tangent_heading = atan2(gate.tangent.y, gate.tangent.x)
        heading_error = (self.simulation.car.heading - tangent_heading + pi) % (2 * pi) - pi
        return (
            *(float(value) for value in sensors),
            float(snapshot["speed"]) / MAX_SPEED,
            self.current_checkpoint / self.checkpoint_count,
            lateral,
            sin(heading_error),
            cos(heading_error),
        )

    def _info(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "elapsed_time": self.steps * DT,
            "current_progress": self.current_checkpoint / self.checkpoint_count,
            "furthest_progress": self.furthest_checkpoint / self.checkpoint_count,
            "episode_return": self.episode_return,
            "termination_reason": self.termination_reason,
        }
