from __future__ import annotations

from dataclasses import dataclass
from random import Random

from backend.env.environment import DiscreteAction, Observation

from .discretisation import StateDiscretizer
from .q_table import QTable

TABULAR_ARCHITECTURE = "tabular-smooth-v3"
LEGACY_TABULAR_ARCHITECTURE = "tabular-legacy"
TABULAR_ACTIONS = (
    DiscreteAction.COAST,
    DiscreteAction.THROTTLE,
    DiscreteAction.BRAKE,
    DiscreteAction.LEFT,
    DiscreteAction.LEFT_THROTTLE,
    DiscreteAction.RIGHT,
    DiscreteAction.RIGHT_THROTTLE,
)
INACTIVE_TABULAR_ACTIONS = (
    DiscreteAction.LEFT_BRAKE,
    DiscreteAction.RIGHT_BRAKE,
)


@dataclass(frozen=True, slots=True)
class QLearningConfig:
    architecture: str = TABULAR_ARCHITECTURE
    learning_rate: float = 0.1
    discount: float = 0.9995
    epsilon_start: float = 1.0
    epsilon_min: float = 0.10
    epsilon_decay: float = 0.997
    epsilon_decay_steps: int = 200_000
    epsilon_reheat: float = 0.30
    bucket_count: int = 5
    action_repeat: int = 2
    sticky_tolerance: float = 0.03
    canonical_start_probability: float = 0.50
    tabular_stall_seconds: float = 10.0

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
        if not self.architecture:
            raise ValueError("architecture must not be empty")
        if self.architecture == TABULAR_ARCHITECTURE and self.bucket_count != 5:
            raise ValueError("the smooth tabular architecture requires five sensor thresholds")
        if self.architecture != TABULAR_ARCHITECTURE and self.bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be positive")
        if self.sticky_tolerance < 0:
            raise ValueError("sticky_tolerance must be non-negative")
        if not 0 <= self.canonical_start_probability <= 1:
            raise ValueError("canonical_start_probability must be in [0, 1]")
        if self.tabular_stall_seconds <= 0:
            raise ValueError("tabular_stall_seconds must be positive")


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
        self.previous_action: DiscreteAction | None = None

    def start_episode(self, episode: int | None = None) -> None:
        if episode is not None and episode < 1:
            raise ValueError("episode must be positive")
        self.previous_action = None

    def choose_action(self, observation: Observation) -> DiscreteAction:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(TABULAR_ACTIONS)
        else:
            action = _choose_tabular_action(
                self.q_values(observation),
                self.previous_action,
                self.config.sticky_tolerance,
            )
        self.previous_action = action
        return action

    def update(
        self,
        observation: Observation,
        action: DiscreteAction,
        reward: float,
        next_observation: Observation,
        done: bool,
        *,
        duration: int = 1,
    ) -> None:
        if not isinstance(action, DiscreteAction):
            raise TypeError("action must be a DiscreteAction")
        if action not in TABULAR_ACTIONS:
            raise ValueError("action is not active in the smooth tabular architecture")
        if duration < 1:
            raise ValueError("duration must be positive")
        state = self.discretizer.discretize(observation)
        current = self.q_table.value(state, action)
        next_values = self.q_values(next_observation)
        future = 0.0 if done else max(next_values[int(action)] for action in TABULAR_ACTIONS)
        target = float(reward) + self.config.discount**duration * future
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
        values = list(self.q_table.values(self.discretizer.discretize(observation)))
        for action in INACTIVE_TABULAR_ACTIONS:
            values[int(action)] = 0.0
        return tuple(values)

    @property
    def visited_states(self) -> int:
        return self.q_table.visited_states


@dataclass(slots=True)
class GreedyPolicy:
    agent: QLearningAgent
    rng: Random | None = None
    previous_action: DiscreteAction | None = None

    def start_episode(self) -> None:
        self.previous_action = None

    def choose_action(self, observation: Observation) -> DiscreteAction:
        action = _choose_tabular_action(
            self.agent.q_values(observation),
            self.previous_action,
            self.agent.config.sticky_tolerance,
        )
        self.previous_action = action
        return action

    def q_values(self, observation: Observation) -> tuple[float, ...]:
        return self.agent.q_values(observation)


def _choose_tabular_action(
    values: tuple[float, ...],
    previous_action: DiscreteAction | None,
    sticky_tolerance: float,
) -> DiscreteAction:
    if len(values) != len(DiscreteAction):
        raise ValueError("tabular action selection requires nine Q-values")
    best_value = max(values[int(action)] for action in TABULAR_ACTIONS)
    if (
        previous_action in TABULAR_ACTIONS
        and values[int(previous_action)] >= best_value - sticky_tolerance
    ):
        return previous_action
    candidates = [
        action for action in TABULAR_ACTIONS if values[int(action)] == best_value
    ]
    if previous_action in candidates:
        return previous_action
    if DiscreteAction.COAST in candidates:
        return DiscreteAction.COAST
    return candidates[0]
