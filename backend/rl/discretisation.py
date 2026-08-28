from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias, cast

from backend.env.environment import Observation

DiscreteState: TypeAlias = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class StateDiscretizer:
    bucket_count: int = 5

    def __post_init__(self) -> None:
        if self.bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")

    def discretize(self, observation: Observation) -> DiscreteState:
        if len(observation) != 6:
            raise ValueError("observation must contain six values")

        buckets: list[int] = []
        for value in observation:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("observation values must be finite")
            clamped = min(1.0, max(0.0, numeric))
            buckets.append(min(int(clamped * self.bucket_count), self.bucket_count - 1))
        return cast(DiscreteState, tuple(buckets))
