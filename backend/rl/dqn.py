from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from random import Random

import torch
from torch import Tensor, nn

from backend.env.environment import DiscreteAction, Observation
from backend.env.sensors import SENSOR_COUNT

OBSERVATION_SIZE = SENSOR_COUNT + 5
SPEED_INDEX = SENSOR_COUNT
DQN_ACTIONS = tuple(DiscreteAction)
LOW_SPEED_DQN_ACTIONS = (
    DiscreteAction.THROTTLE,
    DiscreteAction.LEFT_THROTTLE,
    DiscreteAction.RIGHT_THROTTLE,
)


@dataclass(frozen=True, slots=True)
class DQNConfig:
    learning_rate: float = 1e-3
    discount: float = 0.9995
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    target_sync_steps: int = 2_000
    n_step: int = 10
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 250_000
    gradient_clip: float = 10.0
    low_speed_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.low_speed_threshold <= 1:
            raise ValueError("low_speed_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ReplayItem:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    done: bool
    discount: float


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(DiscreteAction)),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class DQNAgent:
    def __init__(self, rng: Random, config: DQNConfig = DQNConfig(), *, seed: int = 0) -> None:
        self.rng = rng
        self.config = config
        torch.manual_seed(seed)
        self.online = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.buffer: deque[ReplayItem] = deque(maxlen=config.replay_capacity)
        self.pending: deque[tuple[Observation, DiscreteAction, float, Observation, bool]] = deque()
        self.transitions = 0
        self.updates = 0

    @property
    def epsilon(self) -> float:
        amount = min(1.0, self.transitions / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + amount * (self.config.epsilon_min - self.config.epsilon_start)

    def choose_action(self, observation: Observation) -> DiscreteAction:
        eligible_actions = self.eligible_actions(observation)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(eligible_actions)
        return self.greedy_action(observation)

    def greedy_action(self, observation: Observation) -> DiscreteAction:
        values = self.q_values(observation)
        return max(self.eligible_actions(observation), key=lambda action: values[int(action)])

    def eligible_actions(self, observation: Observation) -> tuple[DiscreteAction, ...]:
        if len(observation) != OBSERVATION_SIZE:
            raise ValueError(f"observation must contain {OBSERVATION_SIZE} values")
        return (
            LOW_SPEED_DQN_ACTIONS
            if float(observation[SPEED_INDEX]) < self.config.low_speed_threshold
            else DQN_ACTIONS
        )

    def q_values(self, observation: Observation) -> tuple[float, ...]:
        with torch.no_grad():
            values = self.online(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))[0]
        return tuple(float(value) for value in values)

    def observe(
        self,
        observation: Observation,
        action: DiscreteAction,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> float | None:
        self.pending.append((observation, action, reward, next_observation, done))
        self.transitions += 1
        if len(self.pending) >= self.config.n_step or done:
            self._emit_pending()
        if done:
            while self.pending:
                self._emit_pending()
        if len(self.buffer) < max(self.config.warmup_steps, self.config.batch_size):
            return None
        return self._optimize()

    def _emit_pending(self) -> None:
        total = 0.0
        next_observation = self.pending[0][3]
        done = False
        used = 0
        for used, (_, _, reward, candidate_next, candidate_done) in enumerate(self.pending, 1):
            total += self.config.discount ** (used - 1) * reward
            next_observation = candidate_next
            done = candidate_done
            if candidate_done or used >= self.config.n_step:
                break
        observation, action, _, _, _ = self.pending.popleft()
        self.buffer.append(
            ReplayItem(observation, int(action), total, next_observation, done, self.config.discount**used)
        )

    def _optimize(self) -> float:
        batch = self.rng.sample(tuple(self.buffer), self.config.batch_size)
        observations = torch.tensor([item.observation for item in batch], dtype=torch.float32)
        actions = torch.tensor([item.action for item in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32)
        next_observations = torch.tensor([item.next_observation for item in batch], dtype=torch.float32)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32)
        discounts = torch.tensor([item.discount for item in batch], dtype=torch.float32)

        predicted = self.online(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.online(next_observations)
            low_speed = next_observations[:, SPEED_INDEX] < self.config.low_speed_threshold
            ineligible = torch.tensor(
                [action not in LOW_SPEED_DQN_ACTIONS for action in DQN_ACTIONS],
                dtype=torch.bool,
                device=next_values.device,
            )
            eligible_next_values = next_values.masked_fill(
                low_speed.unsqueeze(1) & ineligible.unsqueeze(0),
                -torch.inf,
            )
            next_actions = eligible_next_values.argmax(dim=1, keepdim=True)
            future = self.target(next_observations).gather(1, next_actions).squeeze(1)
            expected = rewards + (1 - dones) * discounts * future
        loss = nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def snapshot(self) -> dict[str, object]:
        return {
            "observation_size": OBSERVATION_SIZE,
            "config": asdict(self.config),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": list(self.buffer),
            "pending": list(self.pending),
            "transitions": self.transitions,
            "updates": self.updates,
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> DQNAgent:
        observation_size = snapshot.get("observation_size", 10)
        if observation_size != OBSERVATION_SIZE:
            raise ValueError(
                f"DQN checkpoint uses {observation_size} observation values; "
                f"the current seven-ray architecture requires {OBSERVATION_SIZE}"
            )
        config = DQNConfig(**snapshot["config"])  # type: ignore[arg-type]
        agent = cls(Random(), config)
        agent.online.load_state_dict(snapshot["online"])  # type: ignore[arg-type]
        agent.target.load_state_dict(snapshot["target"])  # type: ignore[arg-type]
        agent.optimizer.load_state_dict(snapshot["optimizer"])  # type: ignore[arg-type]
        agent.buffer.extend(snapshot["buffer"])  # type: ignore[arg-type]
        agent.pending.extend(snapshot["pending"])  # type: ignore[arg-type]
        agent.transitions = int(snapshot["transitions"])
        agent.updates = int(snapshot["updates"])
        agent.rng.setstate(snapshot["rng_state"])  # type: ignore[arg-type]
        torch.set_rng_state(snapshot["torch_rng_state"])  # type: ignore[arg-type]
        return agent


@dataclass(slots=True)
class DQNPolicy:
    agent: DQNAgent

    def choose_action(self, observation: Observation) -> DiscreteAction:
        return self.agent.greedy_action(observation)

    def q_values(self, observation: Observation) -> tuple[float, ...]:
        return self.agent.q_values(observation)
