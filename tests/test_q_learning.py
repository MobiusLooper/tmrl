from __future__ import annotations

from dataclasses import dataclass
from math import inf, nan
from random import Random

import pytest

from backend.env.environment import DiscreteAction, Observation, RacingEnv, StepResult
from backend.rl import (
    TABULAR_ACTIONS,
    TABULAR_STATE_COUNT,
    GreedyPolicy,
    QLearningAgent,
    QLearningConfig,
    StateDiscretizer,
)
from backend.training import evaluate_policy, run_training, run_training_episode

ZERO_OBSERVATION: Observation = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
NEXT_OBSERVATION: Observation = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 1.0)


def test_discretizer_clamps_values_and_handles_bucket_boundaries() -> None:
    observation: Observation = (-0.1, 0.039, 0.04, 0.3, 1.0, 1.2, 1.2, -1.2, 0.0, 1.0)

    assert StateDiscretizer(5).discretize(observation) == (0, 0, 1, 4, 5, 4, 3)
    assert StateDiscretizer(5).discretize((1.0,) * 10) == (5, 5, 5, 5, 5, 4, 3)
    assert TABULAR_STATE_COUNT == 155_520


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0399, 0),
        (0.04, 1),
        (0.0799, 1),
        (0.08, 2),
        (0.1499, 2),
        (0.15, 3),
        (0.2999, 3),
        (0.30, 4),
        (0.5999, 4),
        (0.60, 5),
    ],
)
def test_sensor_bucket_boundaries(value: float, expected: int) -> None:
    observation: Observation = (value, 1, 1, 1, 1, 0, 0, 0, 0, 1)
    assert StateDiscretizer().discretize(observation)[0] == expected


@pytest.mark.parametrize(
    ("speed", "expected"),
    [(0.0999, 0), (0.10, 1), (0.2499, 1), (0.25, 2), (0.70, 4)],
)
def test_speed_bucket_boundaries(speed: float, expected: int) -> None:
    observation: Observation = (0, 0, 0, 0, 0, speed, 0, 0, 0, 1)
    assert StateDiscretizer().discretize(observation)[5] == expected


@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0.2499, 0), (0.25, 1), (0.4999, 1), (0.50, 2), (0.75, 3), (1.0, 3)],
)
def test_progress_sector_boundaries(progress: float, expected: int) -> None:
    observation: Observation = (0, 0, 0, 0, 0, 0, progress, 0, 0, 1)
    assert StateDiscretizer().discretize(observation)[6] == expected


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_discretizer_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        StateDiscretizer().discretize((value, 0, 0, 0, 0, 0, 0, 0, 0, 1))


def test_discretizer_validates_shape_and_bucket_count() -> None:
    with pytest.raises(ValueError, match="requires five"):
        StateDiscretizer(1)
    with pytest.raises(ValueError, match="ten"):
        StateDiscretizer().discretize((0.0,) * 5)


def test_q_learning_update_bootstraps_only_non_terminal_transitions() -> None:
    agent = QLearningAgent(
        Random(0),
        QLearningConfig(learning_rate=0.5, discount=0.9, epsilon_start=0, epsilon_min=0),
    )
    state = agent.discretizer.discretize(ZERO_OBSERVATION)
    next_state = agent.discretizer.discretize(NEXT_OBSERVATION)
    agent.q_table.set_value(state, DiscreteAction.THROTTLE, 1.0)
    agent.q_table.set_value(next_state, DiscreteAction.LEFT, 4.0)

    agent.update(ZERO_OBSERVATION, DiscreteAction.THROTTLE, 2.0, NEXT_OBSERVATION, False)
    assert agent.q_table.value(state, DiscreteAction.THROTTLE) == pytest.approx(3.3)

    agent.update(ZERO_OBSERVATION, DiscreteAction.THROTTLE, 2.0, NEXT_OBSERVATION, True)
    assert agent.q_table.value(state, DiscreteAction.THROTTLE) == pytest.approx(2.65)


def test_q_learning_uses_macro_duration_for_bootstrap_discount() -> None:
    agent = QLearningAgent(
        Random(0),
        QLearningConfig(
            learning_rate=1,
            discount=0.5,
            epsilon_start=0,
            epsilon_min=0,
        ),
    )
    next_state = agent.discretizer.discretize(NEXT_OBSERVATION)
    agent.q_table.set_value(next_state, DiscreteAction.LEFT, 4)

    agent.update(
        ZERO_OBSERVATION,
        DiscreteAction.THROTTLE,
        1.5,
        NEXT_OBSERVATION,
        False,
        duration=2,
    )
    state = agent.discretizer.discretize(ZERO_OBSERVATION)
    assert agent.q_table.value(state, DiscreteAction.THROTTLE) == 2.5


def test_epsilon_schedule_uses_training_transitions_and_clamps() -> None:
    agent = QLearningAgent(
        Random(0),
        QLearningConfig(epsilon_start=0.8, epsilon_min=0.2, epsilon_decay_steps=2),
    )

    agent.start_episode(1)
    assert agent.epsilon == 0.8
    agent.update(ZERO_OBSERVATION, DiscreteAction.COAST, 0, NEXT_OBSERVATION, False)
    assert agent.epsilon == pytest.approx(0.5)
    agent.update(ZERO_OBSERVATION, DiscreteAction.COAST, 0, NEXT_OBSERVATION, False)
    assert agent.epsilon == 0.2
    agent.start_episode(20)
    assert agent.epsilon == 0.2
    with pytest.raises(ValueError, match="positive"):
        agent.start_episode(0)


def test_epsilon_reheats_and_starts_a_new_transition_schedule() -> None:
    agent = QLearningAgent(
        Random(0),
        QLearningConfig(epsilon_min=0.1, epsilon_reheat=0.3, epsilon_decay_steps=2),
    )
    agent.restore_exploration(2)
    assert agent.epsilon == 0.1
    agent.reheat_epsilon()
    assert agent.epsilon == 0.3
    agent.update(ZERO_OBSERVATION, DiscreteAction.COAST, 0, NEXT_OBSERVATION, False)
    assert agent.epsilon == pytest.approx(0.2)


def test_seeded_exploration_uses_only_the_seven_active_actions() -> None:
    config = QLearningConfig(epsilon_start=1, epsilon_min=1)
    first = QLearningAgent(Random(12), config)
    second = QLearningAgent(Random(12), config)
    assert [first.choose_action(ZERO_OBSERVATION) for _ in range(20)] == [
        second.choose_action(ZERO_OBSERVATION) for _ in range(20)
    ]

    assert set(first.choose_action(ZERO_OBSERVATION) for _ in range(500)) == set(TABULAR_ACTIONS)


def test_greedy_policy_is_sticky_and_tie_breaking_is_deterministic() -> None:
    agent = QLearningAgent(
        Random(1),
        QLearningConfig(epsilon_start=0, epsilon_min=0, sticky_tolerance=0.03),
    )
    state = agent.discretizer.discretize(ZERO_OBSERVATION)
    agent.q_table.set_value(state, DiscreteAction.RIGHT, 1.0)
    agent.q_table.set_value(state, DiscreteAction.LEFT, 0.98)
    policy = GreedyPolicy(agent, previous_action=DiscreteAction.LEFT)
    assert policy.choose_action(ZERO_OBSERVATION) == DiscreteAction.LEFT

    agent.q_table.set_value(state, DiscreteAction.LEFT, 0.96)
    assert policy.choose_action(ZERO_OBSERVATION) == DiscreteAction.RIGHT

    agent.q_table.set_value(state, DiscreteAction.LEFT, 1.0)
    policy.previous_action = None
    assert policy.choose_action(ZERO_OBSERVATION) == DiscreteAction.LEFT
    policy.previous_action = DiscreteAction.RIGHT
    assert policy.choose_action(ZERO_OBSERVATION) == DiscreteAction.RIGHT

    empty_policy = GreedyPolicy(QLearningAgent(Random(2)))
    assert empty_policy.choose_action(ZERO_OBSERVATION) == DiscreteAction.COAST
    empty_policy.start_episode()
    assert empty_policy.previous_action is None


def test_inactive_actions_are_masked_and_excluded_from_bootstrapping() -> None:
    agent = QLearningAgent(
        Random(1),
        QLearningConfig(
            learning_rate=1,
            discount=1,
            epsilon_start=0,
            epsilon_min=0,
        ),
    )
    next_state = agent.discretizer.discretize(NEXT_OBSERVATION)
    for action in TABULAR_ACTIONS:
        agent.q_table.set_value(next_state, action, -2)
    agent.q_table.set_value(next_state, DiscreteAction.LEFT_BRAKE, 100)
    agent.q_table.set_value(next_state, DiscreteAction.RIGHT_BRAKE, 100)

    assert agent.q_values(NEXT_OBSERVATION)[5] == 0
    assert agent.q_values(NEXT_OBSERVATION)[8] == 0
    agent.update(ZERO_OBSERVATION, DiscreteAction.THROTTLE, 0, NEXT_OBSERVATION, False)
    state = agent.discretizer.discretize(ZERO_OBSERVATION)
    assert agent.q_table.value(state, DiscreteAction.THROTTLE) == -2


@dataclass
class TwoStepEnvironment:
    step_number: int = 0

    def reset(self) -> Observation:
        self.step_number = 0
        return ZERO_OBSERVATION

    def step(self, action: DiscreteAction) -> StepResult:
        del action
        self.step_number += 1
        done = self.step_number == 2
        observation = NEXT_OBSERVATION if not done else (0.7,) * 10
        return StepResult(
            observation=observation,
            reward=float(self.step_number),
            done=done,
            info={
                "steps": self.step_number,
                "elapsed_time": self.step_number * 0.05,
                "furthest_progress": self.step_number / 10,
                "episode_return": 3.0 if done else 1.0,
                "termination_reason": "timeout" if done else None,
            },
        )


def test_training_aggregates_two_ticks_into_one_discounted_update() -> None:
    agent = QLearningAgent(
        Random(2),
        QLearningConfig(
            learning_rate=1,
            epsilon_start=0.8,
            epsilon_min=0.2,
            discount=0.5,
            epsilon_decay_steps=2,
        ),
    )
    record = run_training_episode(TwoStepEnvironment(), agent, 1)

    assert record.steps == 2
    assert agent.visited_states == 1
    assert agent.training_steps == 1
    state = agent.discretizer.discretize(ZERO_OBSERVATION)
    assert agent.previous_action is not None
    assert agent.q_table.value(state, agent.previous_action) == 2.0
    assert agent.epsilon == pytest.approx(0.5)
    before = agent.q_table.snapshot()
    epsilon = agent.epsilon
    evaluation = evaluate_policy(
        TwoStepEnvironment(),
        GreedyPolicy(agent, Random(9)),
        3,
        training_episode=1,
        action_repeat=2,
    )

    assert evaluation.mean_progress == pytest.approx(0.2)
    assert agent.q_table.snapshot() == before
    assert agent.epsilon == epsilon


def test_macro_action_stops_after_a_terminal_first_tick() -> None:
    agent = QLearningAgent(
        Random(1),
        QLearningConfig(epsilon_start=0, epsilon_min=0, action_repeat=2),
    )
    record = run_training_episode(RacingEnv(max_steps=1), agent, 1)

    assert record.steps == 1
    assert record.termination_reason == "timeout"
    assert agent.training_steps == 1


def test_training_episode_resets_previous_action() -> None:
    agent = QLearningAgent(Random(1))
    agent.previous_action = DiscreteAction.RIGHT
    run_training_episode(RacingEnv(max_steps=1), agent, 1)
    assert agent.previous_action in TABULAR_ACTIONS
    agent.start_episode(2)
    assert agent.previous_action is None


def test_short_real_training_run_is_seed_reproducible() -> None:
    config = QLearningConfig(epsilon_decay=0.8)
    first_agent = QLearningAgent(Random(7), config)
    second_agent = QLearningAgent(Random(7), config)

    first = run_training(
        RacingEnv(max_steps=4),
        first_agent,
        3,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=99,
    )
    second = run_training(
        RacingEnv(max_steps=4),
        second_agent,
        3,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=99,
    )

    assert first.records == second.records
    assert first.evaluations == second.evaluations
    assert first_agent.q_table.snapshot() == second_agent.q_table.snapshot()
    assert len(first.evaluations) == 2
