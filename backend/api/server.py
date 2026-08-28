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
from backend.training.checkpoint import DEFAULT_CHECKPOINT_PATH, CheckpointError, load_checkpoint_replays
from backend.training.replay import EvaluationReplay

app = FastAPI(title="RL Racer")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = Path(
    os.environ.get("RL_RACER_CHECKPOINT", str(PROJECT_ROOT / DEFAULT_CHECKPOINT_PATH))
).expanduser()


@app.get("/api/track")
async def get_track() -> dict[str, object]:
    return {**TRACK.as_dict(), "sensors": sensor_config()}


@app.get("/api/replays")
async def get_replays() -> dict[str, object]:
    schema_version, algorithm, replays = _load_replays()
    return {
        "schema_version": schema_version,
        "algorithm": algorithm,
        "latest_training_episode": replays[-1].training_episode,
        "replays": [_replay_metadata(replay) for replay in replays],
    }


@app.get("/api/replays/latest")
async def get_latest_replay() -> dict[str, object]:
    return asdict(_load_replays()[2][-1])


@app.get("/api/replays/{training_episode}")
async def get_replay(training_episode: int) -> dict[str, object]:
    for replay in _load_replays()[2]:
        if replay.training_episode == training_episode:
            return asdict(replay)
    raise HTTPException(status_code=404, detail=f"no replay exists for training episode {training_episode}")


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


def _load_replays() -> tuple[int, str, tuple[EvaluationReplay, ...]]:
    if not CHECKPOINT_PATH.is_file():
        raise HTTPException(status_code=404, detail="no training checkpoint is available yet")
    try:
        schema_version, algorithm, replays = load_checkpoint_replays(CHECKPOINT_PATH)
    except CheckpointError as error:
        raise HTTPException(status_code=503, detail=f"training checkpoint is invalid: {error}") from error
    if not replays:
        raise HTTPException(status_code=404, detail="the training checkpoint does not contain any replays")
    return schema_version, algorithm, replays


def _replay_metadata(replay: EvaluationReplay) -> dict[str, object]:
    return {
        "training_episode": replay.training_episode,
        "evaluation_episode": replay.evaluation_episode,
        "total_return": replay.total_return,
        "furthest_progress": replay.furthest_progress,
        "simulated_duration": replay.simulated_duration,
        "steps": replay.steps,
        "termination_reason": replay.termination_reason,
        "lap_completed": replay.lap_completed,
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
