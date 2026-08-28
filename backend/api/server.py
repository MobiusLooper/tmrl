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
from backend.training.replay import EvaluationReplay, ReplayState, steering_metrics
from backend.training.run_catalog import RunCatalog, TrainingRun, discover_runs

app = FastAPI(title="RL Racer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
_configured_checkpoint = os.environ.get("RL_RACER_CHECKPOINT")
CHECKPOINT_PATH = Path(_configured_checkpoint).expanduser() if _configured_checkpoint else None


@app.get("/api/track")
async def get_track() -> dict[str, object]:
    return {**TRACK.as_dict(), "sensors": sensor_config()}


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
    return _replay_catalog(_get_run(run_id))


@app.get("/api/runs/{run_id}/replays/latest")
async def get_run_latest_replay(run_id: str) -> dict[str, object]:
    return asdict(_latest_replay(_get_run(run_id)))


@app.get("/api/runs/{run_id}/replays/{training_episode}")
async def get_run_replay(run_id: str, training_episode: int) -> dict[str, object]:
    return asdict(_find_replay(_get_run(run_id), training_episode))


@app.get("/api/runs/{run_id}/trajectories")
async def get_run_trajectories(run_id: str) -> dict[str, object]:
    return _trajectory_catalog(_get_run(run_id))


@app.websocket("/ws/play")
async def play(websocket: WebSocket) -> None:
    await websocket.accept()
    simulation = RacingSimulation()
    action = Action()
    latest_sequence = -1
    next_tick = monotonic() + DT
    await websocket.send_json(simulation.snapshot())
    try:
        while True:
            timeout = max(0.0, next_tick - monotonic())
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
            except TimeoutError:
                await websocket.send_json(simulation.step(action))
                next_tick += DT
                if next_tick < monotonic() - DT:
                    next_tick = monotonic() + DT
                continue

            message_type = message.get("type")
            if message_type == "reset":
                simulation.reset()
                action = Action()
                await websocket.send_json(simulation.snapshot())
                next_tick = monotonic() + DT
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
        pass


def _run_catalog() -> RunCatalog:
    try:
        return discover_runs(ARTIFACTS_DIR, configured_checkpoint=CHECKPOINT_PATH)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=f"configured training checkpoint does not exist: {error}",
        ) from error
    except CheckpointError as error:
        raise HTTPException(status_code=503, detail=f"training checkpoint is invalid: {error}") from error


def _default_run() -> TrainingRun:
    run = _run_catalog().default
    if run is None:
        raise HTTPException(status_code=404, detail="no training checkpoint is available yet")
    return run


def _get_run(run_id: str) -> TrainingRun:
    run = _run_catalog().find(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no training run exists with ID {run_id}")
    return run


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


def _trajectory_catalog(run: TrainingRun) -> dict[str, object]:
    if not run.replays:
        raise HTTPException(status_code=404, detail="the training checkpoint does not contain any replays")
    return {
        "run_id": run.run_id,
        "latest_training_episode": run.replays[-1].training_episode,
        "trajectories": [
            {
                **_replay_metadata(replay),
                "states": [
                    _trajectory_state(replay.initial_state),
                    *(_trajectory_state(transition.state) for transition in replay.transitions),
                ],
            }
            for replay in run.replays
        ],
    }


def _trajectory_state(state: ReplayState) -> dict[str, object]:
    return {
        "tick": state.tick,
        "x": state.x,
        "y": state.y,
        "heading": state.heading,
        "crashed": state.crashed,
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
