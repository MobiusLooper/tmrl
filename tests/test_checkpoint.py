from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest

from backend.env.environment import DiscreteAction, RacingEnv
from backend.rl import QLearningAgent, QLearningConfig, QTable
from backend.training.checkpoint import (
    CheckpointError,
    checkpoint_from_agent,
    load_checkpoint,
    save_checkpoint,
    load_checkpoint_replays,
)
from backend.training.q_learning import _training_setup, build_parser
from backend.training.replay import (
    EvaluationReplay,
    ReplayState,
    ReplayTransition,
    select_best_replay,
    steering_metrics,
)
from backend.training.trainer import run_training


def test_checkpoint_round_trip_restores_complete_agent_and_history(tmp_path: Path) -> None:
    agent = QLearningAgent(Random(7), QLearningConfig(epsilon_decay=0.8))
    result = run_training(
        RacingEnv(max_steps=4),
        agent,
        2,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=99,
        record_replays=True,
    )
    checkpoint = checkpoint_from_agent(
        agent,
        seed=7,
        completed_episode=2,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=99,
        training_wall_time=result.training_wall_time,
        records=result.records,
        evaluations=result.evaluations,
        replays=result.replays,
    )
    path = tmp_path / "nested" / "latest.json"

    save_checkpoint(checkpoint, path)
    restored = load_checkpoint(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert restored == checkpoint
    assert payload["agent"]["config"]["architecture"] == "tabular-local-v6"
    assert payload["agent"]["config"]["action_repeat"] == 2
    assert payload["agent"]["config"]["sticky_tolerance"] == 0.03
    restored_agent = restored.restore_agent()
    assert restored_agent.q_table.snapshot() == agent.q_table.snapshot()
    assert restored_agent.rng.getstate() == agent.rng.getstate()
    assert restored_agent.epsilon == agent.epsilon
    assert not list(path.parent.glob("*.tmp"))


def test_resumed_training_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    config = QLearningConfig(epsilon_decay=0.8)
    continuous_agent = QLearningAgent(Random(11), config)
    continuous = run_training(
        RacingEnv(max_steps=5),
        continuous_agent,
        4,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=123,
        record_replays=True,
    )

    split_agent = QLearningAgent(Random(11), config)
    first = run_training(
        RacingEnv(max_steps=5),
        split_agent,
        2,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=123,
        record_replays=True,
    )
    checkpoint = checkpoint_from_agent(
        split_agent,
        seed=11,
        completed_episode=2,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=123,
        training_wall_time=first.training_wall_time,
        records=first.records,
        evaluations=first.evaluations,
        replays=first.replays,
    )
    path = tmp_path / "resume.json"
    save_checkpoint(checkpoint, path)
    resumed_agent = load_checkpoint(path).restore_agent()
    second = run_training(
        RacingEnv(max_steps=5),
        resumed_agent,
        4,
        start_episode=3,
        evaluate_every=2,
        evaluation_episodes=2,
        evaluation_seed=123,
        record_replays=True,
    )

    assert first.records + second.records == continuous.records
    assert first.evaluations + second.evaluations == continuous.evaluations
    assert first.replays + second.replays == continuous.replays
    assert resumed_agent.q_table.snapshot() == continuous_agent.q_table.snapshot()
    assert resumed_agent.rng.getstate() == continuous_agent.rng.getstate()
    assert resumed_agent.epsilon == continuous_agent.epsilon


def test_replay_contains_causal_transitions_and_nine_q_values() -> None:
    agent = QLearningAgent(Random(3), QLearningConfig(epsilon_start=0, epsilon_min=0))
    result = run_training(
        RacingEnv(max_steps=3),
        agent,
        1,
        evaluate_every=1,
        evaluation_episodes=1,
        evaluation_seed=4,
        record_replays=True,
    )

    replay = result.replays[0]
    assert replay.steps == len(replay.transitions) == 3
    assert replay.initial_state.tick == 0
    assert [transition.state.tick for transition in replay.transitions] == [1, 2, 3]
    assert replay.transitions[0].action == replay.transitions[1].action
    assert replay.transitions[0].q_values == replay.transitions[1].q_values
    assert all(len(transition.q_values) == len(DiscreteAction) for transition in replay.transitions)
    assert replay.transitions[-1].state.current_progress >= 0
    assert replay.simulated_duration == pytest.approx(0.15)


def test_best_replay_selection_uses_progress_return_duration_then_earliest_episode() -> None:
    state = ReplayState(0, 0, 0, 0, 0, False, (1, 1, 1, 1, 1), 0)
    transition = ReplayTransition(0, "COAST", (0,) * 9, 0, state)

    def candidate(evaluation_episode: int, progress: float, total_return: float, duration: float) -> EvaluationReplay:
        return EvaluationReplay(
            training_episode=10,
            evaluation_episode=evaluation_episode,
            total_return=total_return,
            furthest_progress=progress,
            simulated_duration=duration,
            steps=1,
            termination_reason="timeout",
            lap_completed=False,
            initial_state=state,
            transitions=(transition,),
        )

    replays = [
        candidate(1, 0.4, 2, 3),
        candidate(2, 0.5, 1, 3),
        candidate(3, 0.5, 2, 4),
        candidate(4, 0.5, 2, 2),
        candidate(5, 0.5, 2, 2),
    ]

    assert select_best_replay(replays).evaluation_episode == 4


def test_steering_metrics_count_changes_and_direct_reversals() -> None:
    state = ReplayState(0, 0, 0, 0, 0, False, (1, 1, 1, 1, 1), 0)
    actions = [
        DiscreteAction.LEFT,
        DiscreteAction.LEFT,
        DiscreteAction.RIGHT,
        DiscreteAction.RIGHT,
        DiscreteAction.COAST,
    ]
    transitions = tuple(
        ReplayTransition(int(action), action.name, (0,) * 9, 0, state)
        for action in actions
    )
    replay = EvaluationReplay(
        10,
        1,
        0,
        0,
        0.25,
        5,
        "timeout",
        False,
        state,
        transitions,
    )

    metrics = steering_metrics(replay)
    assert metrics.changes == 3
    assert metrics.direct_reversals == 1
    assert metrics.changes_per_second == 12
    assert metrics.direct_reversals_per_second == 4


def test_training_stops_at_an_episode_boundary_before_evaluation() -> None:
    seen = []
    result = run_training(
        RacingEnv(max_steps=2),
        QLearningAgent(Random(1)),
        5,
        evaluate_every=5,
        on_episode=lambda record, *_: seen.append(record.episode),
        stop_requested=lambda: len(seen) == 1,
    )

    assert [record.episode for record in result.records] == [1]
    assert result.evaluations == ()
    assert result.stopped_early


def test_checkpoint_rejects_unsupported_schema_and_invalid_q_table(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(CheckpointError, match="unsupported"):
        load_checkpoint(path)

    with pytest.raises(ValueError, match="six or nine legacy.*eight local"):
        QTable.from_snapshot({(0,): (0,) * 9}, bucket_count=5)
    with pytest.raises(ValueError, match="9 finite"):
        QTable.from_snapshot({(0,) * 6: (0,) * 8}, bucket_count=5)


def test_resume_cli_uses_a_total_target_and_rejects_conflicting_settings(tmp_path: Path) -> None:
    agent = QLearningAgent(Random(5))
    result = run_training(RacingEnv(max_steps=1), agent, 1, evaluate_every=1)
    checkpoint = checkpoint_from_agent(
        agent,
        seed=5,
        completed_episode=1,
        evaluate_every=1,
        evaluation_episodes=10,
        evaluation_seed=1_000_005,
        training_wall_time=result.training_wall_time,
        records=result.records,
        evaluations=(),
        replays=(),
    )
    path = tmp_path / "resume.json"
    save_checkpoint(checkpoint, path)
    parser = build_parser()

    setup = _training_setup(parser.parse_args(["--resume", str(path), "--episodes", "2"]))
    assert setup.completed_episode == 1

    with pytest.raises(ValueError, match="must exceed checkpoint episode 1"):
        _training_setup(parser.parse_args(["--resume", str(path), "--episodes", "1"]))
    with pytest.raises(ValueError, match="conflicts with saved value"):
        _training_setup(
            parser.parse_args(
                ["--resume", str(path), "--episodes", "2", "--learning-rate", "0.2"]
            )
        )


def test_fresh_cli_accepts_smooth_control_overrides() -> None:
    parser = build_parser()
    setup = _training_setup(
        parser.parse_args(
            [
                "--episodes",
                "2",
                "--action-repeat",
                "3",
                "--sticky-tolerance",
                "0.05",
                "--canonical-start-probability",
                "0.6",
                "--tabular-stall-seconds",
                "12",
            ]
        )
    )

    assert setup.config.action_repeat == 3
    assert setup.config.sticky_tolerance == 0.05
    assert setup.config.canonical_start_probability == 0.6
    assert setup.config.tabular_stall_seconds == 12


def test_previous_nine_part_checkpoint_is_replayable_but_cannot_resume(tmp_path: Path) -> None:
    agent = QLearningAgent(Random(8))
    result = run_training(
        RacingEnv(max_steps=2),
        agent,
        1,
        evaluate_every=1,
        evaluation_episodes=1,
        record_replays=True,
    )
    checkpoint = checkpoint_from_agent(
        agent,
        seed=8,
        completed_episode=1,
        evaluate_every=1,
        evaluation_episodes=1,
        evaluation_seed=1_000_008,
        training_wall_time=result.training_wall_time,
        records=result.records,
        evaluations=result.evaluations,
        replays=result.replays,
    )
    path = tmp_path / "legacy-nine.json"
    save_checkpoint(checkpoint, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["agent"]["config"].pop("architecture")
    for row in payload["agent"]["q_table"]:
        row["state"].append(0)
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, algorithm, replays = load_checkpoint_replays(path)
    assert algorithm == "tabular"
    assert replays == result.replays
    parser = build_parser()
    with pytest.raises(ValueError, match="remains replayable.*fresh run"):
        _training_setup(parser.parse_args(["--resume", str(path), "--episodes", "2"]))


def test_smooth_v3_checkpoint_is_replayable_but_cannot_resume(tmp_path: Path) -> None:
    agent = QLearningAgent(Random(9))
    result = run_training(
        RacingEnv(max_steps=2),
        agent,
        1,
        evaluate_every=1,
        evaluation_episodes=1,
        record_replays=True,
    )
    checkpoint = checkpoint_from_agent(
        agent,
        seed=9,
        completed_episode=1,
        evaluate_every=1,
        evaluation_episodes=1,
        evaluation_seed=1_000_009,
        training_wall_time=result.training_wall_time,
        records=result.records,
        evaluations=result.evaluations,
        replays=result.replays,
    )
    path = tmp_path / "smooth-v3.json"
    save_checkpoint(checkpoint, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["agent"]["config"]["architecture"] = "tabular-smooth-v3"
    for row in payload["agent"]["q_table"]:
        row["state"] = row["state"][:7]
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, algorithm, replays = load_checkpoint_replays(path)
    assert algorithm == "tabular"
    assert replays == result.replays
    parser = build_parser()
    with pytest.raises(ValueError, match="remains replayable.*fresh run"):
        _training_setup(parser.parse_args(["--resume", str(path), "--episodes", "2"]))


@pytest.mark.parametrize("architecture", ["tabular-local-v4", "tabular-local-v5"])
def test_previous_local_checkpoint_is_replayable_but_cannot_resume(
    tmp_path: Path, architecture: str
) -> None:
    agent = QLearningAgent(Random(10))
    result = run_training(
        RacingEnv(max_steps=2),
        agent,
        1,
        evaluate_every=1,
        evaluation_episodes=1,
        record_replays=True,
    )
    checkpoint = checkpoint_from_agent(
        agent,
        seed=10,
        completed_episode=1,
        evaluate_every=1,
        evaluation_episodes=1,
        evaluation_seed=1_000_010,
        training_wall_time=result.training_wall_time,
        records=result.records,
        evaluations=result.evaluations,
        replays=result.replays,
    )
    path = tmp_path / f"{architecture}.json"
    save_checkpoint(checkpoint, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["agent"]["config"]["architecture"] = architecture
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, algorithm, replays = load_checkpoint_replays(path)
    assert algorithm == "tabular"
    assert replays == result.replays
    parser = build_parser()
    with pytest.raises(ValueError, match="remains replayable.*fresh run"):
        _training_setup(parser.parse_args(["--resume", str(path), "--episodes", "2"]))
