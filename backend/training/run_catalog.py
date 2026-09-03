from __future__ import annotations

import json
import mmap
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


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    created_at: str
    updated_at: str
    algorithm: str
    seed: int
    completed_episode: int
    evaluation_count: int
    latest_progress: float | None
    best_progress: float | None
    lap_completed: bool
    checkpoint_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "completed_episode": self.completed_episode,
            "evaluation_count": self.evaluation_count,
            "latest_progress": self.latest_progress,
            "best_progress": self.best_progress,
            "lap_completed": self.lap_completed,
        }


@dataclass(frozen=True, slots=True)
class RunSummaryCatalog:
    runs: tuple[RunSummary, ...]
    default_run_id: str | None

    def find(self, run_id: str) -> RunSummary | None:
        return next((run for run in self.runs if run.run_id == run_id), None)

    @property
    def default(self) -> RunSummary | None:
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


def discover_run_summaries(
    artifacts_dir: Path,
    *,
    configured_checkpoint: Path | None = None,
) -> RunSummaryCatalog:
    """Discover runs without deserializing Q-tables or replay transitions."""
    candidates = _checkpoint_candidates(artifacts_dir)
    configured_resolved: Path | None = None
    if configured_checkpoint is not None:
        if not configured_checkpoint.is_file():
            raise FileNotFoundError(configured_checkpoint)
        configured_resolved = configured_checkpoint.resolve()
        candidates.add(configured_resolved)

    runs_by_id: dict[str, RunSummary] = {}
    configured_run_id: str | None = None
    for candidate in sorted(candidates):
        try:
            run = inspect_run_summary(candidate)
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
    return RunSummaryCatalog(tuple(runs), default_run_id)


def inspect_run_summary(path: Path) -> RunSummary:
    """Read just the small run and evaluation sections of a checkpoint."""
    try:
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            version = _mapped_json_value(data, b'\n  "schema_version": ')
            run = _mapped_json_value(data, b'\n  "run": ')
            evaluations = _mapped_json_value(data, b'\n    "evaluations": ')
            algorithm = "tabular" if version == 1 else _mapped_json_value(
                data, b'\n  "algorithm": '
            )
    except ValueError:
        # Older or hand-authored checkpoints may be valid JSON without the
        # pretty-printed layout used by the current writers.
        return _summary_from_run(inspect_run(path))
    except OSError as error:
        raise CheckpointError(f"unable to inspect checkpoint {path}: {error}") from error

    if version not in {1, 2}:
        raise CheckpointError(f"unsupported checkpoint schema version {version}")
    if not isinstance(run, dict) or not isinstance(evaluations, list):
        raise CheckpointError(f"invalid checkpoint {path}: run and evaluations must be present")
    seed = run.get("seed")
    completed_episode = run.get("completed_episode")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CheckpointError(f"invalid checkpoint {path}: run.seed must be an integer")
    if not isinstance(completed_episode, int) or isinstance(completed_episode, bool):
        raise CheckpointError(f"invalid checkpoint {path}: run.completed_episode must be an integer")

    progress = [item.get("best_progress") for item in evaluations if isinstance(item, dict)]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in progress):
        raise CheckpointError(f"invalid checkpoint {path}: evaluation progress must be numeric")
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    run_id = run.get("run_id")
    return RunSummary(
        run_id=run_id if isinstance(run_id, str) and run_id else inferred_run_id(path),
        created_at=_timestamp(run.get("created_at"), modified_at),
        updated_at=_timestamp(run.get("updated_at"), modified_at),
        algorithm=algorithm if isinstance(algorithm, str) else "unknown",
        seed=seed,
        completed_episode=completed_episode,
        evaluation_count=len(evaluations),
        latest_progress=None if not progress else float(progress[-1]),
        best_progress=None if not progress else float(max(progress)),
        lap_completed=any(
            isinstance(item, dict)
            and isinstance(item.get("lap_completions"), int)
            and item["lap_completions"] > 0
            for item in evaluations
        ),
        checkpoint_path=path,
    )


def _mapped_json_value(data: mmap.mmap, marker: bytes) -> object:
    marker_at = data.find(marker)
    if marker_at < 0:
        raise ValueError(f"missing {marker.decode().strip()}")
    start = marker_at + len(marker)
    first = data[start]
    if first not in (ord("["), ord("{")):
        end = data.find(b"\n", start)
        if end < 0:
            end = len(data)
        return json.loads(data[start:end].rstrip(b",\r").decode("utf-8"))

    closing = ord("]") if first == ord("[") else ord("}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(data)):
        byte = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte == first:
            depth += 1
        elif byte == closing:
            depth -= 1
            if depth == 0:
                return json.loads(data[start:index + 1].decode("utf-8"))
    raise ValueError(f"unterminated {marker.decode().strip()}")


def _summary_from_run(run: TrainingRun) -> RunSummary:
    return RunSummary(
        run_id=run.run_id,
        created_at=run.created_at,
        updated_at=run.updated_at,
        algorithm=run.algorithm,
        seed=run.seed,
        completed_episode=run.completed_episode,
        evaluation_count=len(run.replays),
        latest_progress=run.replays[-1].furthest_progress if run.replays else None,
        best_progress=max((replay.furthest_progress for replay in run.replays), default=None),
        lap_completed=any(replay.lap_completed for replay in run.replays),
        checkpoint_path=run.checkpoint_path,
    )


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
