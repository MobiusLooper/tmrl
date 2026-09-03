from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from .geometry import Point
from .sensors import raw_observation, sensor_readings
from .track import TRACK, Track

DT = 0.05
ACCELERATION = 4.0
BRAKING = 7.0
MAX_SPEED = 12.0
STEERING_RATE = 1.4
CAR_RADIUS = 0.4


@dataclass(slots=True)
class Action:
    throttle: bool = False
    brake: bool = False
    left: bool = False
    right: bool = False


@dataclass(slots=True)
class CarState:
    x: float
    y: float
    heading: float
    speed: float = 0.0

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)


class RacingSimulation:
    def __init__(self, track: Track = TRACK) -> None:
        self.track = track
        self.best_lap_time: float | None = None
        self.reset(clear_best=False)

    def reset(self, *, clear_best: bool = False) -> dict[str, object]:
        if clear_best:
            self.best_lap_time = None
        self.car = CarState(
            x=self.track.start_position.x,
            y=self.track.start_position.y,
            heading=self.track.start_heading,
        )
        self.tick = 0
        self.crashed = False
        self.laps = 0
        self.current_lap_time = 0.0
        self.last_lap_time: float | None = None
        self.passed_halfway = False
        self.lap_started = False
        return self.snapshot()

    def reset_pose(self, position: Point, heading: float, speed: float = 0.0) -> dict[str, object]:
        """Reset episode physics at a deterministic curriculum pose."""
        self.reset(clear_best=False)
        self.car = CarState(position.x, position.y, heading, max(0.0, min(MAX_SPEED, speed)))
        return self.snapshot()

    def step(self, action: Action) -> dict[str, object]:
        if self.crashed:
            return self.snapshot()

        previous_position = self.car.position

        if action.throttle != action.brake:
            acceleration = ACCELERATION if action.throttle else -BRAKING
            self.car.speed = max(0.0, min(MAX_SPEED, self.car.speed + acceleration * DT))

        if self.car.speed > 0 and action.left != action.right:
            direction = 1.0 if action.left else -1.0
            self.car.heading += direction * STEERING_RATE * DT

        self.car.x += cos(self.car.heading) * self.car.speed * DT
        self.car.y += sin(self.car.heading) * self.car.speed * DT
        self.tick += 1
        if self.car.speed > 0:
            self.lap_started = True
        if self.lap_started:
            self.current_lap_time += DT

        current_position = self.car.position
        if not self.track.is_on_track(current_position, CAR_RADIUS):
            self.crashed = True
            return self.snapshot()

        if not self.passed_halfway and self.track.halfway_gate.contains_crossing(
            previous_position, current_position, self.track.half_width
        ):
            self.passed_halfway = True

        if self.passed_halfway and self.track.finish_gate.contains_crossing(
            previous_position, current_position, self.track.half_width
        ):
            self.laps += 1
            self.last_lap_time = self.current_lap_time
            if self.best_lap_time is None or self.last_lap_time < self.best_lap_time:
                self.best_lap_time = self.last_lap_time
            self.current_lap_time = 0.0
            self.passed_halfway = False

        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        return {
            "type": "state",
            "tick": self.tick,
            "x": self.car.x,
            "y": self.car.y,
            "heading": self.car.heading,
            "speed": self.car.speed,
            "crashed": self.crashed,
            "laps": self.laps,
            "current_lap_time": self.current_lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "sensors": list(sensor_readings(self.track, self.car.position, self.car.heading)),
        }

    def observation(self) -> tuple[float, ...]:
        return raw_observation(
            self.track,
            self.car.position,
            self.car.heading,
            self.car.speed,
            MAX_SPEED,
        )
