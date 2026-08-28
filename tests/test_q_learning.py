from __future__ import annotations

from dataclasses import dataclass
from math import inf, nan
from random import Random

import pytest

from backend.env.environment import DiscreteAction, Observation, RacingEnv, StepResult
from backend.rl import GreedyPolicy, QLearningAgent, QLearningConfig, StateDiscretizer
from backend.training import evaluate_policy, run_training, run_training_episode

ZERO_OBSERVATION: Observation = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
NEXT_OBSERVATION: Observation = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 1.0)


def test_discretizer_clamps_values_and_handles_bucket_boundaries() -> None:
    observation: Observation = (-0.1, 0.039, 0.04, 0.3, 1.0, 1.2, 1.2, -1.2, 0.0, 1.0)

    assert StateDiscretizer(5).discretize(observation) == (0, 0, 1, 4, 5, 4, 19, 0, 4)
    assert StateDiscretizer(5).discretize((1.0,) * 10) == (5, 5, 5, 5, 5, 4, 19, 4, 5)


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_discretizer_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        StateDiscretizer().discretize((value, 0, 0, 0, 0, 0, 0, 0, 0, 1))


def test_discretizer_validates_shape_and_bucket_count() -> None:
    with pytest.raises(ValueError, match="at least 2"):
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


def test_seeded_exploration_and_greedy_tie_breaking_are_reproducible() -> None:
    config = QLearningConfig(epsilon_start=1, epsilon_min=1)
    first = QLearningAgent(Random(12), config)
    second = QLearningAgent(Random(12), config)
    assert [first.choose_action(ZERO_OBSERVATION) for _ in range(20)] == [
        second.choose_action(ZERO_OBSERVATION) for _ in range(20)
    ]

    state = first.discretizer.discretize(ZERO_OBSERVATION)
    first.q_table.set_value(state, DiscreteAction.LEFT, 3)
    first.q_table.set_value(state, DiscreteAction.RIGHT, 3)
    policy_a = GreedyPolicy(first, Random(4))
    policy_b = GreedyPolicy(first, Random(4))
    actions_a = [policy_a.choose_action(ZERO_OBSERVATION) for _ in range(20)]
    actions_b = [policy_b.choose_action(ZERO_OBSERVATION) for _ in range(20)]
    assert actions_a == actions_b
    assert set(actions_a) == {DiscreteAction.LEFT, DiscreteAction.RIGHT}


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


def test_training_updates_each_transition_and_evaluation_is_read_only() -> None:
    agent = QLearningAgent(
        Random(2),
        QLearningConfig(epsilon_start=0, epsilon_min=0, discount=0),
    )
    record = run_training_episode(TwoStepEnvironment(), agent, 1)

    assert record.steps == 2
    assert agent.visited_states == 2
    before = agent.q_table.snapshot()
    epsilon = agent.epsilon
    evaluation = evaluate_policy(
        TwoStepEnvironment(),
        GreedyPolicy(agent, Random(9)),
        3,
        training_episode=1,
    )

    assert evaluation.mean_progress == pytest.approx(0.2)
    assert agent.q_table.snapshot() == before
    assert agent.epsilon == epsilon


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
