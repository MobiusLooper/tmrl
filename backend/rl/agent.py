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
    discount: float = 0.9995
    epsilon_start: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay: float = 0.997
    epsilon_decay_steps: int = 400_000
    epsilon_reheat: float = 0.30
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
        if self.epsilon_decay_steps < 1:
            raise ValueError("epsilon_decay_steps must be positive")
        if not 0 <= self.epsilon_reheat <= 1:
            raise ValueError("epsilon_reheat must be in [0, 1]")
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
        self.training_steps = 0
        self.epsilon_schedule_start_step = 0
        self.epsilon_schedule_start_value = config.epsilon_start

    def start_episode(self, episode: int) -> None:
        if episode < 1:
            raise ValueError("episode must be positive")

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
        self.training_steps += 1
        self._update_epsilon()

    def reheat_epsilon(self) -> None:
        target = min(
            self.config.epsilon_start,
            max(self.config.epsilon_min, self.config.epsilon_reheat),
        )
        if self.epsilon >= target:
            return
        self.epsilon = target
        self.epsilon_schedule_start_step = self.training_steps
        self.epsilon_schedule_start_value = self.epsilon

    def restore_exploration(
        self,
        training_steps: int,
        schedule_start_step: int = 0,
        schedule_start_value: float | None = None,
    ) -> None:
        self.training_steps = training_steps
        self.epsilon_schedule_start_step = schedule_start_step
        self.epsilon_schedule_start_value = (
            self.config.epsilon_start if schedule_start_value is None else schedule_start_value
        )
        self._update_epsilon()

    def _update_epsilon(self) -> None:
        elapsed = max(0, self.training_steps - self.epsilon_schedule_start_step)
        amount = min(1.0, elapsed / self.config.epsilon_decay_steps)
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon_schedule_start_value
            + amount * (self.config.epsilon_min - self.epsilon_schedule_start_value),
        )

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

    def q_values(self, observation: Observation) -> tuple[float, ...]:
        return self.agent.q_values(observation)


def _choose_max_action(values: tuple[float, ...], rng: Random) -> DiscreteAction:
    best_value = max(values)
    candidates = [action for action in ACTIONS if values[int(action)] == best_value]
    return rng.choice(candidates)
