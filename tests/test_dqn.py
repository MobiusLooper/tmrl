from random import Random

import pytest
import torch

from backend.env.environment import DiscreteAction, RacingEnv
from backend.rl.dqn import DQNAgent, DQNConfig, DQNPolicy, QNetwork
from backend.training.curriculum import AdaptiveCurriculum, CurriculumConfig


OBSERVATION = (0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.25, 0.1, 0.0, 0.0, 1.0)


def test_q_network_has_one_value_per_action() -> None:
    assert QNetwork()(torch.zeros((4, 12))).shape == (4, 9)


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


def test_legacy_five_ray_snapshot_requires_a_fresh_dqn_run() -> None:
    snapshot = DQNAgent(Random(2), DQNConfig(warmup_steps=100), seed=2).snapshot()
    snapshot["observation_size"] = 10

    with pytest.raises(ValueError, match="seven-ray architecture requires 12"):
        DQNAgent.from_snapshot(snapshot)


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


@pytest.mark.parametrize(
    ("floor", "lower", "upper"),
    [(0.75, 0.75, 0.90), (0.50, 0.50, 0.70), (0.25, 0.25, 0.45)],
)
def test_smooth_tabular_curriculum_uses_bounded_stage_bands(
    floor: float,
    lower: float,
    upper: float,
) -> None:
    curriculum = AdaptiveCurriculum(
        Random(3),
        CurriculumConfig(canonical_probability=0, bounded_stages=True),
    )
    curriculum.floor = floor
    environment = RacingEnv()
    starts = []
    for _ in range(100):
        curriculum.reset(environment)
        starts.append(curriculum.last_start_progress)

    assert min(starts) == lower
    assert max(starts) == upper
    assert all(round(value * 100) % 5 == 0 for value in starts)


def test_smooth_tabular_curriculum_samples_half_canonical_starts() -> None:
    curriculum = AdaptiveCurriculum(
        Random(14),
        CurriculumConfig(canonical_probability=0.5, bounded_stages=True),
    )
    environment = RacingEnv()
    canonical = 0
    for _ in range(1_000):
        curriculum.reset(environment)
        canonical += curriculum.last_start_progress == 0

    assert canonical / 1_000 == pytest.approx(0.5, abs=0.04)


def test_curriculum_snapshot_preserves_smooth_configuration() -> None:
    curriculum = AdaptiveCurriculum(
        Random(9),
        CurriculumConfig(canonical_probability=0.5, bounded_stages=True),
    )
    restored = AdaptiveCurriculum.from_snapshot(curriculum.snapshot())
    assert restored.config == curriculum.config
    assert restored.rng.getstate() == curriculum.rng.getstate()


def test_stall_terminates_early_with_explicit_penalty() -> None:
    environment = RacingEnv(stall_steps=2)
    environment.step(DiscreteAction.COAST)
    result = environment.step(DiscreteAction.COAST)

    assert result.done
    assert result.info["termination_reason"] == "stalled"
    assert result.reward == pytest.approx(-15.01)


def test_smooth_tabular_default_stall_limit_is_ten_seconds() -> None:
    from backend.rl import QLearningConfig
    from backend.training.q_learning import _stall_steps

    config = QLearningConfig()
    assert _stall_steps(config) == 200
    environment = RacingEnv(stall_steps=_stall_steps(config))
    result = None
    for _ in range(200):
        result = environment.step(DiscreteAction.COAST)
    assert result is not None
    assert result.done
    assert result.info["termination_reason"] == "stalled"
    assert result.info["elapsed_time"] == pytest.approx(10.0)


def test_near_wall_sensor_values_use_different_tabular_states() -> None:
    from backend.rl.discretisation import StateDiscretizer

    safer = (0.1, 0.064, 0.3, 0.3, 0.3, 0.3, 0.1, 0.2, 0.45, 0.0, 0.0, 1.0)
    danger = (0.1, 0.038, 0.3, 0.3, 0.3, 0.3, 0.1, 0.2, 0.45, 0.0, 0.0, 1.0)
    assert StateDiscretizer().discretize(safer) != StateDiscretizer().discretize(danger)
