from math import atan2, pi

import pytest

from backend.env.environment import DiscreteAction, RacingEnv
from backend.env.track import Gate, TRACK


EXPECTED_CONTROLS = {
    DiscreteAction.COAST: (False, False, False, False),
    DiscreteAction.THROTTLE: (True, False, False, False),
    DiscreteAction.BRAKE: (False, True, False, False),
    DiscreteAction.LEFT: (False, False, True, False),
    DiscreteAction.LEFT_THROTTLE: (True, False, True, False),
    DiscreteAction.LEFT_BRAKE: (False, True, True, False),
    DiscreteAction.RIGHT: (False, False, False, True),
    DiscreteAction.RIGHT_THROTTLE: (True, False, False, True),
    DiscreteAction.RIGHT_BRAKE: (False, True, False, True),
}


def cross_gate(environment: RacingEnv, gate: Gate, *, forward: bool = True):
    direction = gate.tangent if forward else gate.tangent * -1
    before = gate.center - direction * 0.01
    environment.simulation.car.x = before.x
    environment.simulation.car.y = before.y
    environment.simulation.car.heading = atan2(direction.y, direction.x)
    environment.simulation.car.speed = 1.0
    return environment.step(DiscreteAction.COAST)


def test_reset_returns_only_normalized_observation_and_clears_episode() -> None:
    environment = RacingEnv(max_steps=1)
    terminal = environment.step(DiscreteAction.COAST)
    assert terminal.done

    observation = environment.reset()

    assert len(observation) == 6
    assert all(0.0 <= value <= 1.0 for value in observation)
    assert environment.steps == 0
    assert environment.current_checkpoint == 0
    assert environment.furthest_checkpoint == 0
    assert environment.episode_return == 0
    assert not environment.done


def test_discrete_actions_have_stable_order_and_controls() -> None:
    assert list(DiscreteAction) == [DiscreteAction(index) for index in range(9)]
    for discrete_action, expected in EXPECTED_CONTROLS.items():
        controls = discrete_action.controls()
        assert (controls.throttle, controls.brake, controls.left, controls.right) == expected


def test_action_replay_is_deterministic() -> None:
    actions = [
        DiscreteAction.THROTTLE,
        DiscreteAction.LEFT_THROTTLE,
        DiscreteAction.COAST,
        DiscreteAction.RIGHT_BRAKE,
    ] * 5
    first = RacingEnv()
    second = RacingEnv()

    first_results = [first.step(action) for action in actions]
    second_results = [second.step(action) for action in actions]

    assert first_results == second_results


def test_checkpoints_are_ordered_and_reverse_crossings_remove_progress() -> None:
    environment = RacingEnv()

    out_of_order = cross_gate(environment, environment.checkpoints[1])
    assert out_of_order.reward == pytest.approx(-0.01)
    assert out_of_order.info["current_progress"] == 0

    forward = cross_gate(environment, environment.checkpoints[0])
    assert forward.reward == pytest.approx(0.99)
    assert forward.info["current_progress"] == pytest.approx(0.01)
    assert forward.info["furthest_progress"] == pytest.approx(0.01)

    reverse = cross_gate(environment, environment.checkpoints[0], forward=False)
    assert reverse.reward == pytest.approx(-1.01)
    assert reverse.info["current_progress"] == 0
    assert reverse.info["furthest_progress"] == pytest.approx(0.01)
    assert forward.reward + reverse.reward == pytest.approx(-0.02)


def test_crash_reward_and_terminal_step_guard() -> None:
    environment = RacingEnv()
    environment.simulation.car.x = 1_000
    environment.simulation.car.y = 1_000

    result = environment.step(DiscreteAction.COAST)

    assert result.reward == pytest.approx(-10.01)
    assert result.done
    assert result.info["termination_reason"] == "crash"
    with pytest.raises(RuntimeError, match="reset"):
        environment.step(DiscreteAction.COAST)


def test_timeout_reward() -> None:
    result = RacingEnv(max_steps=1).step(DiscreteAction.COAST)

    assert result.reward == pytest.approx(-0.01)
    assert result.done
    assert result.info["termination_reason"] == "timeout"
    assert result.info["elapsed_time"] == pytest.approx(0.05)


def test_lap_requires_all_checkpoints_and_has_additive_reward() -> None:
    incomplete = RacingEnv()
    incomplete_result = cross_gate(incomplete, TRACK.finish_gate)
    assert not incomplete_result.done
    assert incomplete_result.reward == pytest.approx(-0.01)

    complete = RacingEnv()
    complete.current_checkpoint = 99
    complete.furthest_checkpoint = 99
    complete_result = cross_gate(complete, TRACK.finish_gate)

    assert complete_result.reward == pytest.approx(50.99)
    assert complete_result.done
    assert complete_result.info["termination_reason"] == "lap"
    assert complete_result.info["current_progress"] == 1.0
    assert complete_result.info["furthest_progress"] == 1.0


def test_observation_never_exposes_privileged_state() -> None:
    environment = RacingEnv()
    environment.simulation.car.x = 20
    environment.simulation.car.y = 6
    environment.simulation.car.heading = pi

    result = environment.step(DiscreteAction.COAST)

    assert isinstance(result.observation, tuple)
    assert len(result.observation) == 6
    assert set(result.info) == {
        "steps",
        "elapsed_time",
        "current_progress",
        "furthest_progress",
        "episode_return",
        "termination_reason",
    }
