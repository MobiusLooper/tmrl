from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from random import Random
from typing import Protocol, runtime_checkable

from backend.env.environment import Observation


@runtime_checkable
class CurriculumEnvironment(Protocol):
    def reset(self) -> Observation: ...

    def reset_at_progress(self, progress: float, *, speed: float = 3.0) -> Observation: ...


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    canonical_probability: float = 0.25
    initial_floor: float = 0.75
    promotion_window: int = 50
    promotion_rate: float = 0.35
    bounded_stages: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.canonical_probability <= 1:
            raise ValueError("canonical_probability must be in [0, 1]")
        if self.initial_floor not in {0.0, 0.25, 0.5, 0.75}:
            raise ValueError("initial_floor must be one of 0, 0.25, 0.5, or 0.75")
        if self.promotion_window < 1:
            raise ValueError("promotion_window must be positive")
        if not 0 <= self.promotion_rate <= 1:
            raise ValueError("promotion_rate must be in [0, 1]")


class AdaptiveCurriculum:
    """Seeded backwards curriculum that eventually returns to canonical-only starts."""

    def __init__(self, rng: Random, config: CurriculumConfig = CurriculumConfig()) -> None:
        self.rng = rng
        self.config = config
        self.floor = config.initial_floor
        self._outcomes: deque[bool] = deque(maxlen=config.promotion_window)
        self.last_start_progress = 0.0

    def reset(self, environment: CurriculumEnvironment) -> Observation:
        if self.floor <= 0 or self.rng.random() < self.config.canonical_probability:
            self.last_start_progress = 0.0
            return environment.reset()
        ceiling = self._stage_ceiling()
        choices = [
            value / 100
            for value in range(round(self.floor * 100), round(ceiling * 100) + 1, 5)
        ]
        self.last_start_progress = self.rng.choice(choices)
        speed = self.rng.uniform(2.0, 5.0)
        return environment.reset_at_progress(self.last_start_progress, speed=speed)

    def observe(self, lap_completed: bool) -> bool:
        if self.last_start_progress == 0:
            return False
        self._outcomes.append(lap_completed)
        if len(self._outcomes) < self.config.promotion_window:
            return False
        if sum(self._outcomes) / len(self._outcomes) < self.config.promotion_rate:
            return False
        self.floor = max(0.0, self.floor - 0.25)
        self._outcomes.clear()
        return True

    def snapshot(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "floor": self.floor,
            "outcomes": list(self._outcomes),
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> AdaptiveCurriculum:
        raw_config = snapshot.get("config")
        config = (
            CurriculumConfig()
            if raw_config is None
            else CurriculumConfig(**dict(raw_config))  # type: ignore[arg-type]
        )
        curriculum = cls(Random(), config)
        curriculum.floor = float(snapshot["floor"])
        curriculum._outcomes.extend(bool(value) for value in snapshot["outcomes"])  # type: ignore[union-attr]
        curriculum.rng.setstate(snapshot["rng_state"])  # type: ignore[arg-type]
        return curriculum

    def _stage_ceiling(self) -> float:
        if not self.config.bounded_stages:
            return 0.90
        if self.floor >= 0.75:
            return 0.90
        if self.floor >= 0.50:
            return 0.70
        return 0.45
