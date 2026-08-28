from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .checkpoint import CheckpointError, load_checkpoint_replays
from .replay import EvaluationReplay
from .run_storage import inferred_run_id


@dataclass(frozen=True, slots=True)
class TrainingRun:
    run_id: str
    created_at: str
    updated_at: str
    algorithm: str
    seed: int
    completed_episode: int
    schema_version: int
    checkpoint_path: Path
    replays: tuple[EvaluationReplay, ...]

    def as_dict(self) -> dict[str, object]:
        latest_progress = self.replays[-1].furthest_progress if self.replays else None
        best_progress = max((replay.furthest_progress for replay in self.replays), default=None)
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "completed_episode": self.completed_episode,
            "evaluation_count": len(self.replays),
            "latest_progress": latest_progress,
            "best_progress": best_progress,
            "lap_completed": any(replay.lap_completed for replay in self.replays),
        }


@dataclass(frozen=True, slots=True)
class RunCatalog:
    runs: tuple[TrainingRun, ...]
    default_run_id: str | None

    def find(self, run_id: str) -> TrainingRun | None:
        return next((run for run in self.runs if run.run_id == run_id), None)

    @property
    def default(self) -> TrainingRun | None:
        return None if self.default_run_id is None else self.find(self.default_run_id)


def discover_runs(
    artifacts_dir: Path,
    *,
    configured_checkpoint: Path | None = None,
) -> RunCatalog:
    candidates = _checkpoint_candidates(artifacts_dir)
    configured_resolved: Path | None = None
    if configured_checkpoint is not None:
        if not configured_checkpoint.is_file():
            raise FileNotFoundError(configured_checkpoint)
        configured_resolved = configured_checkpoint.resolve()
        candidates.add(configured_resolved)

    runs_by_id: dict[str, TrainingRun] = {}
    configured_run_id: str | None = None
    for candidate in sorted(candidates):
        try:
            run = inspect_run(candidate)
        except CheckpointError:
            if configured_resolved is not None and candidate.resolve() == configured_resolved:
                raise
            continue
        is_configured = configured_resolved is not None and candidate.resolve() == configured_resolved
        if run.run_id not in runs_by_id or is_configured:
            runs_by_id[run.run_id] = run
        if is_configured:
            configured_run_id = run.run_id

    runs = list(runs_by_id.values())
    runs.sort(key=lambda run: (run.updated_at, run.run_id), reverse=True)
    default_run_id = configured_run_id or (runs[0].run_id if runs else None)
    return RunCatalog(tuple(runs), default_run_id)


def inspect_run(path: Path) -> TrainingRun:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"unable to read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointError(f"invalid checkpoint {path}: expected an object")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise CheckpointError(f"invalid checkpoint {path}: run must be an object")

    schema_version, algorithm, replays = load_checkpoint_replays(path)
    seed = run.get("seed")
    completed_episode = run.get("completed_episode")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CheckpointError(f"invalid checkpoint {path}: run.seed must be an integer")
    if not isinstance(completed_episode, int) or isinstance(completed_episode, bool):
        raise CheckpointError(f"invalid checkpoint {path}: run.completed_episode must be an integer")

    modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    run_id = run.get("run_id")
    created_at = _timestamp(run.get("created_at"), modified_at)
    updated_at = _timestamp(run.get("updated_at"), modified_at)
    return TrainingRun(
        run_id=run_id if isinstance(run_id, str) and run_id else inferred_run_id(path),
        created_at=created_at,
        updated_at=updated_at,
        algorithm=algorithm,
        seed=seed,
        completed_episode=completed_episode,
        schema_version=schema_version,
        checkpoint_path=path,
        replays=replays,
    )


def _checkpoint_candidates(artifacts_dir: Path) -> set[Path]:
    if not artifacts_dir.is_dir():
        return set()
    candidates = {path.resolve() for path in (artifacts_dir / "runs").glob("*/checkpoint.json")}
    candidates.update(
        path.resolve()
        for path in artifacts_dir.glob("*.json")
        if path.name != "best.json" and not path.stem.endswith("-best")
    )
    return candidates


def _timestamp(value: object, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        return fallback
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
