from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.env.simulation import Action, DT, RacingSimulation
from backend.env.sensors import sensor_config
from backend.env.track import TRACK
from backend.training.checkpoint import CheckpointError
from backend.training.demonstrations import (
    DemonstrationError,
    DemonstrationStore,
    LapRecorder,
)
from backend.training.replay import EvaluationReplay, steering_metrics
from backend.training.run_catalog import (
    RunSummaryCatalog,
    TrainingRun,
    discover_run_summaries,
    inspect_run,
)
from backend.training.trajectory import (
    save_trajectory_catalog,
    saved_trajectory_count,
    trajectory_path,
)

app = FastAPI(title="RL Racer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEMONSTRATIONS_DIR = ARTIFACTS_DIR / "demonstrations"
DEMONSTRATION_STORE = DemonstrationStore(DEMONSTRATIONS_DIR)
_configured_checkpoint = os.environ.get("RL_RACER_CHECKPOINT")
CHECKPOINT_PATH = Path(_configured_checkpoint).expanduser() if _configured_checkpoint else None


@app.get("/api/track")
async def get_track() -> dict[str, object]:
    return {**TRACK.as_dict(), "sensors": sensor_config()}


@app.get("/api/demonstrations/current")
async def get_current_demonstration_library() -> dict[str, object]:
    try:
        payload = await asyncio.to_thread(DEMONSTRATION_STORE.current)
        return DEMONSTRATION_STORE.summary(payload)
    except DemonstrationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/replays")
async def get_replays() -> dict[str, object]:
    return _replay_catalog(_default_run())


@app.get("/api/replays/latest")
async def get_latest_replay() -> dict[str, object]:
    return asdict(_latest_replay(_default_run()))


@app.get("/api/replays/{training_episode}")
async def get_replay(training_episode: int) -> dict[str, object]:
    return asdict(_find_replay(_default_run(), training_episode))


@app.get("/api/runs")
async def get_runs() -> dict[str, object]:
    catalog = _run_catalog()
    return {
        "default_run_id": catalog.default_run_id,
        "runs": [run.as_dict() for run in catalog.runs],
    }


@app.get("/api/runs/{run_id}/replays")
async def get_run_replays(run_id: str) -> dict[str, object]:
    summary = _run_catalog().find(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"no training run exists with ID {run_id}")
    run = _load_run(summary.checkpoint_path)
    _ensure_trajectory_catalog(summary.checkpoint_path, run)
    return _replay_catalog(run)


@app.get("/api/runs/{run_id}/replays/latest")
async def get_run_latest_replay(run_id: str) -> dict[str, object]:
    return asdict(_latest_replay(_get_run(run_id)))


@app.get("/api/runs/{run_id}/replays/{training_episode}")
async def get_run_replay(run_id: str, training_episode: int) -> dict[str, object]:
    return asdict(_find_replay(_get_run(run_id), training_episode))


@app.get("/api/runs/{run_id}/trajectories", response_model=None)
async def get_run_trajectories(run_id: str) -> FileResponse | dict[str, object]:
    summary = _run_catalog().find(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"no training run exists with ID {run_id}")
    saved_path = trajectory_path(summary.checkpoint_path)
    saved_count = saved_trajectory_count(saved_path)
    if saved_count == 0:
        raise HTTPException(status_code=404, detail="the training checkpoint does not contain any replays")
    if (
        saved_count is not None
        and saved_path.stat().st_mtime_ns >= summary.checkpoint_path.stat().st_mtime_ns
    ):
        return FileResponse(saved_path, media_type="application/json")
    run = _load_run(summary.checkpoint_path)
    if not run.replays:
        raise HTTPException(status_code=404, detail="the training checkpoint does not contain any replays")
    generated_path = _ensure_trajectory_catalog(summary.checkpoint_path, run)
    return FileResponse(generated_path, media_type="application/json")


@app.websocket("/ws/play")
async def play(websocket: WebSocket) -> None:
    await websocket.accept()
    simulation = RacingSimulation()
    action = Action()
    recorder = LapRecorder()
    latest_sequence = -1
    next_tick = monotonic() + DT
    try:
        library = await asyncio.to_thread(DEMONSTRATION_STORE.current)
    except DemonstrationError as error:
        await websocket.send_json(simulation.snapshot())
        await websocket.send_json(
            {"type": "recording_status", "status": "error", "message": str(error)}
        )
        library = None
    else:
        await websocket.send_json(recorder.start(simulation, reset=False))
        await websocket.send_json(
            _recording_status("started", library, "Recording attempt · 0.0s")
        )
    try:
        while True:
            timeout = max(0.0, next_tick - monotonic())
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
            except TimeoutError:
                snapshot, recording_event = recorder.step(simulation, action)
                await websocket.send_json(snapshot)
                if recording_event is not None:
                    if library is None:
                        await websocket.send_json(
                            {
                                "type": "recording_status",
                                "status": "error",
                                "message": "Demonstration library is unavailable",
                            }
                        )
                    elif recording_event.status == "qualifying" and recording_event.lap is not None:
                        lap_time = float(recording_event.lap_time)
                        await websocket.send_json(
                            _recording_status(
                                "saving", library, recording_event.message, lap_time
                            )
                        )
                        try:
                            library = await asyncio.to_thread(
                                DEMONSTRATION_STORE.append, recording_event.lap
                            )
                        except DemonstrationError as error:
                            await websocket.send_json(
                                _recording_status("error", library, str(error))
                            )
                        else:
                            await websocket.send_json(
                                _recording_status(
                                    "saved",
                                    library,
                                    f"Lap {lap_time:.1f}s saved · "
                                    f"{DEMONSTRATION_STORE.summary(library)['lap_count']} "
                                    "qualifying laps · press R for another",
                                    lap_time,
                                )
                            )
                        next_tick = monotonic()
                    else:
                        await websocket.send_json(
                            _recording_status(
                                recording_event.status,
                                library,
                                recording_event.message,
                                recording_event.lap_time,
                            )
                        )
                next_tick += DT
                if next_tick < monotonic() - DT:
                    next_tick = monotonic() + DT
                continue

            message_type = message.get("type")
            if message_type == "reset":
                recorder.cancel("Attempt discarded by restart")
                action = Action()
                snapshot = simulation.reset()
                library_error: str | None = None
                try:
                    library = await asyncio.to_thread(DEMONSTRATION_STORE.current)
                except DemonstrationError as error:
                    library = None
                    library_error = str(error)
                await websocket.send_json(snapshot)
                if library is None:
                    await websocket.send_json(
                        {
                            "type": "recording_status",
                            "status": "error",
                            "message": library_error or "Demonstration library is unavailable",
                        }
                    )
                else:
                    recorder.start(simulation, reset=False)
                    await websocket.send_json(
                        _recording_status("started", library, "Recording attempt · 0.0s")
                    )
                next_tick = monotonic() + DT
                continue
            if message_type == "manual_pause":
                event = recorder.cancel("Attempt discarded after leaving manual mode")
                action = Action()
                if event is not None and library is not None:
                    await websocket.send_json(
                        _recording_status(event.status, library, event.message)
                    )
                continue
            if message_type != "input":
                continue

            sequence = message.get("seq")
            if not isinstance(sequence, int) or sequence <= latest_sequence:
                continue
            latest_sequence = sequence
            action = Action(
                throttle=message.get("throttle") is True,
                brake=message.get("brake") is True,
                left=message.get("left") is True,
                right=message.get("right") is True,
            )
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        recorder.cancel("Recording discarded after disconnect")


def _recording_status(
    status: str,
    library: dict[str, object],
    message: str,
    lap_time: float | None = None,
) -> dict[str, object]:
    return {
        "type": "recording_status",
        **DEMONSTRATION_STORE.summary(library),
        "status": status,
        "message": message,
        "lap_time": lap_time,
    }


def _run_catalog() -> RunSummaryCatalog:
    try:
        return discover_run_summaries(ARTIFACTS_DIR, configured_checkpoint=CHECKPOINT_PATH)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"configured training checkpoint does not exist: {error}",
        ) from error
    except CheckpointError as error:
        raise HTTPException(status_code=503, detail=f"training checkpoint is invalid: {error}") from error


def _default_run() -> TrainingRun:
    summary = _run_catalog().default
    if summary is None:
        raise HTTPException(status_code=404, detail="no training checkpoint is available yet")
    return _load_run(summary.checkpoint_path)


def _get_run(run_id: str) -> TrainingRun:
    summary = _run_catalog().find(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"no training run exists with ID {run_id}")
    return _load_run(summary.checkpoint_path)


def _load_run(path: Path) -> TrainingRun:
    try:
        return inspect_run(path)
    except CheckpointError as error:
        raise HTTPException(status_code=503, detail=f"training checkpoint is invalid: {error}") from error


def _ensure_trajectory_catalog(checkpoint_path: Path, run: TrainingRun) -> Path:
    saved_path = trajectory_path(checkpoint_path)
    if (
        saved_trajectory_count(saved_path) is not None
        and saved_path.stat().st_mtime_ns >= checkpoint_path.stat().st_mtime_ns
    ):
        return saved_path
    return save_trajectory_catalog(checkpoint_path, run.run_id, run.replays)


def _replay_catalog(run: TrainingRun) -> dict[str, object]:
    if not run.replays:
        raise HTTPException(status_code=404, detail="the training checkpoint does not contain any replays")
    return {
        "run_id": run.run_id,
        "schema_version": run.schema_version,
        "algorithm": run.algorithm,
        "latest_training_episode": run.replays[-1].training_episode,
        "replays": [_replay_metadata(replay) for replay in run.replays],
    }


def _latest_replay(run: TrainingRun) -> EvaluationReplay:
    if not run.replays:
        raise HTTPException(status_code=404, detail="the training checkpoint does not contain any replays")
    return run.replays[-1]


def _find_replay(run: TrainingRun, training_episode: int) -> EvaluationReplay:
    for replay in run.replays:
        if replay.training_episode == training_episode:
            return replay
    raise HTTPException(status_code=404, detail=f"no replay exists for training episode {training_episode}")


def _replay_metadata(replay: EvaluationReplay) -> dict[str, object]:
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
    }


DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST_DIR.exists():
    assets = DIST_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        candidate = (DIST_DIR / path).resolve()
        if path and candidate.is_relative_to(DIST_DIR) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")
