from random import Random

import pytest
import torch

from backend.env.environment import DiscreteAction, RacingEnv
from backend.rl.dqn import DQNAgent, DQNConfig, DQNPolicy, QNetwork
from backend.training.curriculum import AdaptiveCurriculum, CurriculumConfig


OBSERVATION = (0.2, 0.3, 0.4, 0.3, 0.2, 0.25, 0.1, 0.0, 0.0, 1.0)


def test_q_network_has_one_value_per_action() -> None:
    assert QNetwork()(torch.zeros((4, 10))).shape == (4, 9)


def test_dqn_n_step_transition_and_terminal_flush() -> None:
    config = DQNConfig(n_step=3, warmup_steps=100, batch_size=2)
    agent = DQNAgent(Random(1), config, seed=1)
    for index in range(3):
        agent.observe(OBSERVATION, DiscreteAction.THROTTLE, 1.0, OBSERVATION, index == 2)

    assert len(agent.pending) == 0
    assert len(agent.buffer) == 3
    assert agent.buffer[0].reward == pytest.approx(1 + config.discount + config.discount**2)
    assert agent.buffer[0].done


def test_dqn_policy_is_greedy_and_snapshot_restores() -> None:
    agent = DQNAgent(Random(2), DQNConfig(warmup_steps=100), seed=2)
    action = DQNPolicy(agent).choose_action(OBSERVATION)
    restored = DQNAgent.from_snapshot(agent.snapshot())

    assert action == DQNPolicy(restored).choose_action(OBSERVATION)
    assert restored.q_values(OBSERVATION) == pytest.approx(agent.q_values(OBSERVATION))


def test_curriculum_is_seeded_and_promotes_to_canonical() -> None:
    config = CurriculumConfig(canonical_probability=0, promotion_window=2, promotion_rate=1)
    first = AdaptiveCurriculum(Random(5), config)
    second = AdaptiveCurriculum(Random(5), config)
    first_env = RacingEnv()
    second_env = RacingEnv()

    assert first.reset(first_env) == second.reset(second_env)
    assert first.last_start_progress == second.last_start_progress
    first.observe(True)
    first.reset(first_env)
    first.observe(True)
    assert first.floor == 0.5


def test_default_curriculum_promotes_at_thirty_five_percent() -> None:
    curriculum = AdaptiveCurriculum(Random(1))
    curriculum.last_start_progress = 0.75
    promoted = False
    for index in range(50):
        promoted = curriculum.observe(index < 18)
    assert promoted
    assert curriculum.floor == 0.5


def test_stall_terminates_early_with_explicit_penalty() -> None:
    environment = RacingEnv(stall_steps=2)
    environment.step(DiscreteAction.COAST)
    result = environment.step(DiscreteAction.COAST)

    assert result.done
    assert result.info["termination_reason"] == "stalled"
    assert result.reward == pytest.approx(-15.01)


def test_near_wall_sensor_values_use_different_tabular_states() -> None:
    from backend.rl.discretisation import StateDiscretizer

    safer = (0.064, 0.3, 0.3, 0.3, 0.3, 0.2, 0.45, 0.0, 0.0, 1.0)
    danger = (0.038, 0.3, 0.3, 0.3, 0.3, 0.2, 0.45, 0.0, 0.0, 1.0)
    assert StateDiscretizer().discretize(safer) != StateDiscretizer().discretize(danger)
