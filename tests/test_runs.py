import json
import os
from pathlib import Path
from random import Random

import pytest

from backend.env.environment import RacingEnv
from backend.rl import QLearningAgent
from backend.rl.dqn import DQNAgent, DQNConfig
from backend.training.checkpoint import checkpoint_from_agent, load_checkpoint, save_checkpoint
from backend.training.curriculum import AdaptiveCurriculum
from backend.training.dqn_training import _resume_model_path, _save as save_dqn, main as dqn_main
from backend.training.q_learning import main as tabular_main
from backend.training.run_catalog import discover_runs, inspect_run
from backend.training.run_storage import RunMetadata, create_run
from backend.training.trainer import run_training


def test_automatic_run_directories_are_unique_and_timestamped(tmp_path: Path) -> None:
    first, first_checkpoint = create_run("tabular", 7, runs_dir=tmp_path / "runs")
    second, second_checkpoint = create_run("tabular", 7, runs_dir=tmp_path / "runs")

    assert first.run_id != second.run_id
    assert "-tabular-seed7-" in first.run_id
    assert first_checkpoint == tmp_path / "runs" / first.run_id / "checkpoint.json"
    assert first_checkpoint.parent.is_dir()
    assert second_checkpoint.parent.is_dir()


def test_tabular_run_metadata_round_trips_and_updates(tmp_path: Path) -> None:
    checkpoint, _ = _tabular_checkpoint()
    path = tmp_path / "checkpoint.json"
    first = RunMetadata("run-one", "2026-08-28T10:00:00Z", "2026-08-28T10:01:00Z")
    save_checkpoint(_with_metadata(checkpoint, first), path)
    assert load_checkpoint(path).updated_at == first.updated_at

    second = RunMetadata(first.run_id, first.created_at, "2026-08-28T10:02:00Z")
    save_checkpoint(_with_metadata(checkpoint, second), path)
    restored = load_checkpoint(path)
    assert restored.run_id == first.run_id
    assert restored.created_at == first.created_at
    assert restored.updated_at == second.updated_at


def test_run_discovery_combines_new_and_legacy_checkpoints(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    checkpoint, _ = _tabular_checkpoint()
    tracked_path = artifacts / "runs" / "tracked-run" / "checkpoint.json"
    tracked = RunMetadata("tracked-run", "2026-08-28T10:00:00Z", "2026-08-28T12:00:00Z")
    save_checkpoint(_with_metadata(checkpoint, tracked), tracked_path)
    save_checkpoint(_with_metadata(checkpoint, tracked), tracked_path.with_name("best.json"))
    legacy_path = artifacts / "latest.json"
    save_checkpoint(checkpoint, legacy_path)
    os.utime(legacy_path, (1, 1))
    (artifacts / "broken.json").write_text("not json", encoding="utf-8")

    catalog = discover_runs(artifacts)

    assert catalog.runs[0].run_id == "tracked-run"
    assert catalog.runs[1].run_id.startswith("legacy-latest-")
    assert catalog.default_run_id == "tracked-run"
    assert len(catalog.runs) == 2

    configured = discover_runs(artifacts, configured_checkpoint=legacy_path)
    assert configured.default_run_id == catalog.runs[1].run_id


def test_dqn_run_uses_colocated_manifest_model_files(tmp_path: Path) -> None:
    _, result = _tabular_checkpoint()
    path = tmp_path / "checkpoint.json"
    metadata = RunMetadata("dqn-run", "2026-08-28T10:00:00Z", "2026-08-28T10:05:00Z")
    save_dqn(
        path,
        DQNAgent(Random(3), DQNConfig(warmup_steps=100), seed=3),
        AdaptiveCurriculum(Random(4)),
        list(result.records),
        list(result.evaluations),
        list(result.replays),
        1,
        result.training_wall_time,
        3,
        metadata,
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["model_file"] == "model.pt"
    assert manifest["best_model_file"] == "best-model.pt"
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "best-model.pt").is_file()
    assert _resume_model_path(path, manifest) == tmp_path / "model.pt"
    assert _resume_model_path(tmp_path / "legacy.json", {}) == tmp_path / "legacy.pt"
    assert inspect_run(path).algorithm == "dqn"


def test_training_rejects_resume_with_checkpoint_override(tmp_path: Path) -> None:
    arguments = ["--resume", str(tmp_path / "old.json"), "--checkpoint", str(tmp_path / "new.json")]
    with pytest.raises(SystemExit, match="cannot be combined"):
        tabular_main(arguments)
    with pytest.raises(SystemExit, match="cannot be combined"):
        dqn_main(arguments)


def _tabular_checkpoint():
    agent = QLearningAgent(Random(2))
    result = run_training(
        RacingEnv(max_steps=1),
        agent,
        1,
        evaluate_every=1,
        evaluation_episodes=1,
        evaluation_seed=8,
        record_replays=True,
    )
    checkpoint = checkpoint_from_agent(
        agent,
        seed=2,
        completed_episode=1,
        evaluate_every=1,
        evaluation_episodes=1,
        evaluation_seed=8,
        training_wall_time=result.training_wall_time,
        records=result.records,
        evaluations=result.evaluations,
        replays=result.replays,
    )
    return checkpoint, result


def _with_metadata(checkpoint, metadata: RunMetadata):
    return checkpoint_from_agent(
        checkpoint.restore_agent(),
        seed=checkpoint.seed,
        completed_episode=checkpoint.completed_episode,
        evaluate_every=checkpoint.evaluate_every,
        evaluation_episodes=checkpoint.evaluation_episodes,
        evaluation_seed=checkpoint.evaluation_seed,
        training_wall_time=checkpoint.training_wall_time,
        records=checkpoint.records,
        evaluations=checkpoint.evaluations,
        replays=checkpoint.replays,
        curriculum=checkpoint.curriculum,
        run_metadata=metadata,
    )
