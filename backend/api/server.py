from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.env.simulation import Action, DT, RacingSimulation
from backend.env.sensors import sensor_config
from backend.env.track import TRACK

app = FastAPI(title="RL Racer Physics Sandbox")


@app.get("/api/track")
async def get_track() -> dict[str, object]:
    return {**TRACK.as_dict(), "sensors": sensor_config()}


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
