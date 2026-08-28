from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import isfinite
from typing import TypeAlias

from backend.env.environment import Observation

DiscreteState: TypeAlias = tuple[int, ...]

SENSOR_THRESHOLDS = (0.04, 0.08, 0.15, 0.30, 0.60)
SPEED_THRESHOLDS = (0.10, 0.25, 0.45, 0.70)
PROGRESS_SECTORS = 4
TABULAR_STATE_COUNT = 6**5 * 5 * PROGRESS_SECTORS


@dataclass(frozen=True, slots=True)
class StateDiscretizer:
    bucket_count: int = 5

    def __post_init__(self) -> None:
        if self.bucket_count != 5:
            raise ValueError("the smooth tabular architecture requires five sensor thresholds")

    def discretize(self, observation: Observation) -> DiscreteState:
        if len(observation) != 10:
            raise ValueError("observation must contain ten values")
        for value in observation:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("observation values must be finite")
        sensors = tuple(
            bisect_right(SENSOR_THRESHOLDS, min(1.0, max(0.0, value)))
            for value in observation[:5]
        )
        speed = bisect_right(SPEED_THRESHOLDS, min(1.0, max(0.0, observation[5])))
        progress = min(
            PROGRESS_SECTORS - 1,
            int(min(1.0, max(0.0, observation[6])) * PROGRESS_SECTORS),
        )
        return (*sensors, speed, progress)
