from __future__ import annotations

from dataclasses import dataclass
from random import Random

from backend.env.environment import DiscreteAction, Observation

from .discretisation import StateDiscretizer
from .q_table import QTable

ACTIONS = tuple(DiscreteAction)


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    bucket_count: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= minimum <= start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if self.bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")


class QLearningAgent:
    def __init__(
        self,
        rng: Random,
        config: QLearningConfig = QLearningConfig(),
        *,
        q_table: QTable | None = None,
    ) -> None:
        self.rng = rng
        self.config = config
        self.discretizer = StateDiscretizer(config.bucket_count)
        self.q_table = q_table if q_table is not None else QTable()
        self.epsilon = config.epsilon_start

    def start_episode(self, episode: int) -> None:
        if episode < 1:
            raise ValueError("episode must be positive")
        self.epsilon = max(
            self.config.epsilon_min,
            self.config.epsilon_start * self.config.epsilon_decay ** (episode - 1),
        )

    def choose_action(self, observation: Observation) -> DiscreteAction:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        return _choose_max_action(self.q_values(observation), self.rng)

    def update(
        self,
        observation: Observation,
        action: DiscreteAction,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        if not isinstance(action, DiscreteAction):
            raise TypeError("action must be a DiscreteAction")
        state = self.discretizer.discretize(observation)
        current = self.q_table.value(state, action)
        future = 0.0 if done else max(self.q_values(next_observation))
        target = float(reward) + self.config.discount * future
        updated = current + self.config.learning_rate * (target - current)
        self.q_table.set_value(state, action, updated)

    def q_values(self, observation: Observation) -> tuple[float, ...]:
        return self.q_table.values(self.discretizer.discretize(observation))

    @property
    def visited_states(self) -> int:
        return self.q_table.visited_states


@dataclass(slots=True)
class GreedyPolicy:
    agent: QLearningAgent
    rng: Random

    def choose_action(self, observation: Observation) -> DiscreteAction:
        return _choose_max_action(self.agent.q_values(observation), self.rng)


def _choose_max_action(values: tuple[float, ...], rng: Random) -> DiscreteAction:
    best_value = max(values)
    candidates = [action for action in ACTIONS if values[int(action)] == best_value]
    return rng.choice(candidates)
