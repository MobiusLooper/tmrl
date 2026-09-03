from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .replay import EvaluationReplay, ReplayState, steering_metrics

TRAJECTORY_SCHEMA_VERSION = 1


def trajectory_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.name == "checkpoint.json":
        return checkpoint.with_name("trajectories.json")
    return checkpoint.with_name(f"{checkpoint.stem}-trajectories.json")


def save_trajectory_catalog(
    checkpoint_path: str | Path,
    run_id: str,
    replays: tuple[EvaluationReplay, ...] | list[EvaluationReplay],
) -> Path:
    destination = trajectory_path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = trajectory_catalog(run_id, replays)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def trajectory_catalog(
    run_id: str,
    replays: tuple[EvaluationReplay, ...] | list[EvaluationReplay],
) -> dict[str, object]:
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "trajectory_count": len(replays),
        "run_id": run_id,
        "latest_training_episode": replays[-1].training_episode if replays else None,
        "trajectories": [trajectory(replay) for replay in replays],
    }


def trajectory(replay: EvaluationReplay) -> dict[str, object]:
    smoothness = steering_metrics(replay)
    return {
        "training_episode": replay.training_episode,
        "evaluation_episode": replay.evaluation_episode,
        "total_return": replay.total_return,
        "furthest_progress": replay.furthest_progress,
        "simulated_duration": replay.simulated_duration,
        "steps": replay.steps,
        "termination_reason": replay.termination_reason,
        "lap_completed": replay.lap_completed,
        "steering_changes_per_second": smoothness.changes_per_second,
        "direct_steering_reversals_per_second": smoothness.direct_reversals_per_second,
        "states": [
            trajectory_state(replay.initial_state),
            *(trajectory_state(transition.state) for transition in replay.transitions),
        ],
    }


def trajectory_state(state: ReplayState) -> dict[str, object]:
    return {
        "tick": state.tick,
        "x": state.x,
        "y": state.y,
        "heading": state.heading,
        "crashed": state.crashed,
    }


def saved_trajectory_count(path: Path) -> int | None:
    """Read the fixed-size catalog header without parsing the route states."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(128)
    except OSError:
        return None
    marker = b'"trajectory_count":'
    start = prefix.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = start
    while end < len(prefix) and prefix[end:end + 1].isdigit():
        end += 1
    try:
        return int(prefix[start:end])
    except ValueError:
        return None
