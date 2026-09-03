from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from backend.env.environment import DiscreteAction, Observation


class Agent(Protocol):
    def choose_action(self, observation: Observation) -> DiscreteAction: ...


@dataclass(slots=True)
class RandomAgent:
    rng: Random

    def choose_action(self, observation: Observation) -> DiscreteAction:
        del observation
        return self.rng.choice(tuple(DiscreteAction))
