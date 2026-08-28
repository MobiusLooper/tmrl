from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import atan2, isfinite, pi
from typing import TypeAlias, cast

from backend.env.environment import Observation

DiscreteState: TypeAlias = tuple[int, ...]

SENSOR_THRESHOLDS = (0.04, 0.08, 0.15, 0.30, 0.60)
SPEED_THRESHOLDS = (0.10, 0.25, 0.45, 0.70)
LATERAL_THRESHOLDS = (-0.60, -0.20, 0.20, 0.60)


@dataclass(frozen=True, slots=True)
class StateDiscretizer:
    bucket_count: int = 5

    def __post_init__(self) -> None:
        if self.bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")

    def discretize(self, observation: Observation) -> DiscreteState:
        if len(observation) != 10:
            raise ValueError("observation must contain ten values")
        for value in observation:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("observation values must be finite")
        sensors = tuple(bisect_right(SENSOR_THRESHOLDS, min(1.0, max(0.0, value))) for value in observation[:5])
        speed = bisect_right(SPEED_THRESHOLDS, min(1.0, max(0.0, observation[5])))
        progress = min(19, int(min(1.0, max(0.0, observation[6])) * 20))
        lateral = bisect_right(LATERAL_THRESHOLDS, min(1.0, max(-1.0, observation[7])))
        angle = atan2(observation[8], observation[9])
        heading = min(7, int((angle + pi) / (2 * pi) * 8))
        return cast(DiscreteState, (*sensors, speed, progress, lateral, heading))
