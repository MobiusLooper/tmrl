from __future__ import annotations

from dataclasses import dataclass
from math import cos, inf, nan, pi, sin
from random import Random

import pytest

from backend.env.environment import DiscreteAction, Observation, RacingEnv, StepResult
from backend.rl import (
    LOW_SPEED_TABULAR_ACTIONS,
    TABULAR_ACTIONS,
    TABULAR_STATE_COUNT,
    GreedyPolicy,
    QLearningAgent,
    QLearningConfig,
    StateDiscretizer,
)
from backend.training import evaluate_policy, run_training, run_training_episode
from backend.training.q_learning import TABULAR_REWARDS


def with_side_rays(observation: tuple[float, ...]) -> Observation:
    """Expand a legacy five-ray observation with neutral side-ray values."""
    assert len(observation) == 10
    return (0.0, *observation[:5], 0.0, *observation[5:])


ZERO_OBSERVATION = with_side_rays((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
NEXT_OBSERVATION = with_side_rays((0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 1.0))
MOVING_OBSERVATION = with_side_rays((0.2, 0.3, 0.7, 0.4, 0.2, 0.2, 0.25, 0.1, 0.0, 1.0))


def test_discretizer_clamps_values_and_handles_bucket_boundaries() -> None:
    observation = with_side_rays((-0.1, 0.039, 0.04, 0.3, 1.0, 1.2, 1.2, -1.2, 0.0, 1.0))

    assert StateDiscretizer(5).discretize(observation) == (0, 0, 1, 4, 5, 4, 0, 2)
    assert StateDiscretizer(5).discretize((1.0,) * 12) == (5, 5, 5, 5, 5, 4, 2, 3)
    assert TABULAR_STATE_COUNT == 583_200


def test_local_v6_rewards_penalize_late_progress_and_timeout_more_strongly() -> None:
    assert TABULAR_REWARDS.pace_floor == -1.0
    result = RacingEnv(rewards=TABULAR_REWARDS, max_steps=1).step(DiscreteAction.COAST)
    assert result.reward == pytest.approx(-30.01)
    assert result.info["termination_reason"] == "timeout"


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
    observation = with_side_rays((value, 1, 1, 1, 1, 0, 0, 0, 0, 1))
    assert StateDiscretizer().discretize(observation)[0] == expected


@pytest.mark.parametrize(
    ("speed", "expected"),
    [(0.0999, 0), (0.10, 1), (0.2499, 1), (0.25, 2), (0.70, 4)],
)
def test_speed_bucket_boundaries(speed: float, expected: int) -> None:
    observation = with_side_rays((0, 0, 0, 0, 0, speed, 0, 0, 0, 1))
    assert StateDiscretizer().discretize(observation)[5] == expected


@pytest.mark.parametrize(
    ("lateral", "expected"),
    [(-1, 0), (-0.5001, 0), (-0.5, 1), (0.4999, 1), (0.5, 2), (1, 2)],
)
def test_lateral_bucket_boundaries(lateral: float, expected: int) -> None:
    observation = with_side_rays((0, 0, 0, 0, 0, 0, 0, lateral, 0, 1))
    assert StateDiscretizer().discretize(observation)[6] == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (-pi, 0),
        (-pi / 3, 1),
        (-pi / 12, 2),
        (pi / 12, 3),
        (pi / 3, 4),
        (pi, 4),
    ],
)
def test_heading_bucket_boundaries(angle: float, expected: int) -> None:
    observation = with_side_rays((0, 0, 0, 0, 0, 0, 0, 0, sin(angle), cos(angle)))
    assert StateDiscretizer().discretize(observation)[7] == expected


def test_absolute_progress_does_not_change_local_tabular_state() -> None:
    before = with_side_rays((0.2, 0.3, 0.7, 0.4, 0.2, 0.2, 0.2499, 0.1, 0.0, 1.0))
    after = with_side_rays((0.2, 0.3, 0.7, 0.4, 0.2, 0.2, 0.25, 0.1, 0.0, 1.0))
    assert StateDiscretizer().discretize(before) == StateDiscretizer().discretize(after)


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_discretizer_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        StateDiscretizer().discretize(with_side_rays((value, 0, 0, 0, 0, 0, 0, 0, 0, 1)))


def test_discretizer_validates_shape_and_bucket_count() -> None:
    with pytest.raises(ValueError, match="requires five"):
        StateDiscretizer(1)
    with pytest.raises(ValueError, match="12"):
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


def test_seeded_exploration_respects_speed_eligible_actions() -> None:
    config = QLearningConfig(epsilon_start=1, epsilon_min=1)
    first = QLearningAgent(Random(12), config)
    second = QLearningAgent(Random(12), config)
    assert [first.choose_action(ZERO_OBSERVATION) for _ in range(20)] == [
        second.choose_action(ZERO_OBSERVATION) for _ in range(20)
    ]

    assert set(first.choose_action(ZERO_OBSERVATION) for _ in range(500)) == set(
        LOW_SPEED_TABULAR_ACTIONS
    )
    assert set(first.choose_action(MOVING_OBSERVATION) for _ in range(500)) == set(TABULAR_ACTIONS)


def test_greedy_policy_is_sticky_and_tie_breaking_is_deterministic() -> None:
    agent = QLearningAgent(
        Random(1),
        QLearningConfig(epsilon_start=0, epsilon_min=0, sticky_tolerance=0.03),
    )
    state = agent.discretizer.discretize(MOVING_OBSERVATION)
    agent.q_table.set_value(state, DiscreteAction.RIGHT, 1.0)
    agent.q_table.set_value(state, DiscreteAction.LEFT, 0.98)
    policy = GreedyPolicy(agent, previous_action=DiscreteAction.LEFT)
    assert policy.choose_action(MOVING_OBSERVATION) == DiscreteAction.LEFT

    agent.q_table.set_value(state, DiscreteAction.LEFT, 0.96)
    assert policy.choose_action(MOVING_OBSERVATION) == DiscreteAction.RIGHT

    agent.q_table.set_value(state, DiscreteAction.LEFT, 1.0)
    policy.previous_action = None
    assert policy.choose_action(MOVING_OBSERVATION) == DiscreteAction.LEFT
    policy.previous_action = DiscreteAction.RIGHT
    assert policy.choose_action(MOVING_OBSERVATION) == DiscreteAction.RIGHT

    empty_policy = GreedyPolicy(QLearningAgent(Random(2)))
    assert empty_policy.choose_action(ZERO_OBSERVATION) == DiscreteAction.THROTTLE
    empty_policy.start_episode()
    assert empty_policy.previous_action is None


def test_unseen_state_does_not_stick_with_brake_across_the_25_percent_boundary() -> None:
    agent = QLearningAgent(Random(2), QLearningConfig(epsilon_start=0, epsilon_min=0))
    policy = GreedyPolicy(agent, previous_action=DiscreteAction.BRAKE)

    assert policy.choose_action(MOVING_OBSERVATION) == DiscreteAction.COAST

    stopped = (*MOVING_OBSERVATION[:7], 0.0, *MOVING_OBSERVATION[8:])
    policy.previous_action = DiscreteAction.BRAKE
    assert policy.choose_action(stopped) == DiscreteAction.THROTTLE


def test_low_speed_known_state_selects_and_bootstraps_only_propulsive_actions() -> None:
    config = QLearningConfig(
        learning_rate=1,
        discount=1,
        epsilon_start=0,
        epsilon_min=0,
    )
    agent = QLearningAgent(Random(3), config)
    low_next = with_side_rays((0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.3, 0.0, 0.0, 1.0))
    next_state = agent.discretizer.discretize(low_next)
    agent.q_table.set_value(next_state, DiscreteAction.BRAKE, 100)
    agent.q_table.set_value(next_state, DiscreteAction.LEFT_THROTTLE, 4)

    policy = GreedyPolicy(agent, previous_action=DiscreteAction.BRAKE)
    assert policy.choose_action(low_next) == DiscreteAction.LEFT_THROTTLE

    agent.update(MOVING_OBSERVATION, DiscreteAction.COAST, 0, low_next, False)
    state = agent.discretizer.discretize(MOVING_OBSERVATION)
    assert agent.q_table.value(state, DiscreteAction.COAST) == 4


def test_brake_and_steer_actions_are_selected_and_used_for_bootstrapping() -> None:
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
    agent.q_table.set_value(next_state, DiscreteAction.LEFT_BRAKE, 4)
    agent.q_table.set_value(next_state, DiscreteAction.RIGHT_BRAKE, 3)

    assert GreedyPolicy(agent).choose_action(NEXT_OBSERVATION) == DiscreteAction.LEFT_BRAKE
    assert agent.q_values(NEXT_OBSERVATION)[5] == 4
    assert agent.q_values(NEXT_OBSERVATION)[8] == 3
    agent.update(ZERO_OBSERVATION, DiscreteAction.THROTTLE, 0, NEXT_OBSERVATION, False)
    state = agent.discretizer.discretize(ZERO_OBSERVATION)
    assert agent.q_table.value(state, DiscreteAction.THROTTLE) == 4


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
        observation = NEXT_OBSERVATION if not done else (0.7,) * 12
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
