from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import atan2, isfinite, pi
from typing import TypeAlias

from backend.env.environment import Observation
from backend.env.sensors import SENSOR_COUNT

DiscreteState: TypeAlias = tuple[int, ...]

SENSOR_THRESHOLDS = (0.04, 0.08, 0.15, 0.30, 0.60)
SPEED_THRESHOLDS = (0.10, 0.25, 0.45, 0.70)
LATERAL_THRESHOLDS = (-0.5, 0.5)
HEADING_THRESHOLDS = (-pi / 3, -pi / 12, pi / 12, pi / 3)
TABULAR_STATE_COUNT = 6**5 * 5 * 3 * 5
OBSERVATION_SIZE = SENSOR_COUNT + 5
TABULAR_SENSOR_SLICE = slice(1, 6)
SPEED_INDEX = SENSOR_COUNT
PROGRESS_INDEX = SENSOR_COUNT + 1
LATERAL_INDEX = SENSOR_COUNT + 2
HEADING_SIN_INDEX = SENSOR_COUNT + 3
HEADING_COS_INDEX = SENSOR_COUNT + 4


@dataclass(frozen=True, slots=True)
class StateDiscretizer:
    bucket_count: int = 5

    def __post_init__(self) -> None:
        if self.bucket_count != 5:
            raise ValueError("the local tabular architecture requires five sensor thresholds")

    def discretize(self, observation: Observation) -> DiscreteState:
        if len(observation) != OBSERVATION_SIZE:
            raise ValueError(f"observation must contain {OBSERVATION_SIZE} values")
        for value in observation:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("observation values must be finite")
        sensors = tuple(
            bisect_right(SENSOR_THRESHOLDS, min(1.0, max(0.0, value)))
            for value in observation[TABULAR_SENSOR_SLICE]
        )
        speed = bisect_right(SPEED_THRESHOLDS, min(1.0, max(0.0, observation[SPEED_INDEX])))
        lateral = bisect_right(
            LATERAL_THRESHOLDS,
            min(1.0, max(-1.0, observation[LATERAL_INDEX])),
        )
        heading = bisect_right(
            HEADING_THRESHOLDS,
            atan2(observation[HEADING_SIN_INDEX], observation[HEADING_COS_INDEX]),
        )
        return (*sensors, speed, lateral, heading)
