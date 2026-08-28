from __future__ import annotations

from dataclasses import dataclass
from random import Random

import pytest

from backend.env.environment import DiscreteAction, Observation, RacingEnv, StepResult
from backend.training.agents import RandomAgent
from backend.training.runner import EpisodeRecord, run_episode, run_episodes, summarize_run


OBSERVATION: Observation = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class ScriptedAgent:
    action: DiscreteAction = DiscreteAction.COAST

    def choose_action(self, observation: Observation) -> DiscreteAction:
        assert len(observation) == 6
        return self.action


@dataclass
class TerminalEnvironment:
    reason: str

    def reset(self) -> Observation:
        return OBSERVATION

    def step(self, action: DiscreteAction) -> StepResult:
        del action
        return StepResult(
            observation=OBSERVATION,
            reward=1.0,
            done=True,
            info={
                "steps": 1,
                "elapsed_time": 0.05,
                "current_progress": 1.0 if self.reason == "lap" else 0.0,
                "furthest_progress": 1.0 if self.reason == "lap" else 0.25,
                "episode_return": 1.0,
                "termination_reason": self.reason,
            },
        )


def test_random_agent_is_seeded_and_uniform_over_action_space() -> None:
    first = RandomAgent(Random(7))
    second = RandomAgent(Random(7))
    different = RandomAgent(Random(8))

    first_actions = [first.choose_action(OBSERVATION) for _ in range(30)]
    second_actions = [second.choose_action(OBSERVATION) for _ in range(30)]
    different_actions = [different.choose_action(OBSERVATION) for _ in range(30)]

    assert first_actions == second_actions
    assert first_actions != different_actions
    assert set(first_actions) <= set(DiscreteAction)


@pytest.mark.parametrize("reason", ["crash", "lap", "timeout"])
def test_runner_records_every_terminal_reason(reason: str) -> None:
    record = run_episode(TerminalEnvironment(reason), ScriptedAgent(), episode=4)

    assert record.episode == 4
    assert record.steps == 1
    assert record.termination_reason == reason
    assert record.lap_completed is (reason == "lap")
    assert record.lap_time == (0.05 if reason == "lap" else None)


def test_scripted_agent_runs_to_environment_step_limit() -> None:
    records, _ = run_episodes(RacingEnv(max_steps=3), ScriptedAgent(), 2)

    assert [record.steps for record in records] == [3, 3]
    assert [record.termination_reason for record in records] == ["timeout", "timeout"]


def test_seed_reproduces_complete_episode_records() -> None:
    first, _ = run_episodes(RacingEnv(max_steps=8), RandomAgent(Random(11)), 3)
    second, _ = run_episodes(RacingEnv(max_steps=8), RandomAgent(Random(11)), 3)

    assert first == second


def test_summary_aggregates_hand_checked_records() -> None:
    records = [
        EpisodeRecord(1, 10, 0.5, -2.0, 0.2, "crash", False, None),
        EpisodeRecord(2, 20, 1.0, 4.0, 1.0, "lap", True, 1.0),
        EpisodeRecord(3, 30, 1.5, 1.0, 0.4, "timeout", False, None),
    ]

    summary = summarize_run(records, seed=5, wall_time=2.0)

    assert summary.seed == 5
    assert summary.episodes == 3
    assert summary.total_steps == 60
    assert summary.steps_per_second == 30
    assert summary.mean_return == 1
    assert summary.mean_progress == pytest.approx(1.6 / 3)
    assert summary.best_progress == 1
    assert summary.lap_completions == 1
    assert summary.crash_count == 1
    assert summary.timeout_count == 1


def test_runner_and_summary_reject_empty_runs() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_episodes(RacingEnv(), ScriptedAgent(), 0)
    with pytest.raises(ValueError, match="one episode"):
        summarize_run([], seed=0, wall_time=0)
