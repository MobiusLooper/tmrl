from pathlib import Path
from random import Random

from fastapi.testclient import TestClient

import backend.api.server as server_module
from backend.env.environment import RacingEnv
from backend.rl import QLearningAgent
from backend.api.server import app
from backend.training.checkpoint import checkpoint_from_agent, save_checkpoint
from backend.training.trainer import run_training


client = TestClient(app)


def test_track_endpoint() -> None:
    response = client.get("/api/track")
    assert response.status_code == 200
    assert response.json()["track_width"] > 0
    assert response.json()["sensors"] == {
        "angles": [-60.0, -30.0, 0.0, 30.0, 60.0],
        "max_range": 12.0,
    }


def test_websocket_accepts_input_and_reset() -> None:
    with client.websocket_connect("/ws/play") as websocket:
        initial = websocket.receive_json()
        assert len(initial["sensors"]) == 5
        assert all(0 <= reading <= 1 for reading in initial["sensors"])
        websocket.send_json(
            {"type": "input", "seq": 1, "throttle": True, "brake": False, "left": False, "right": False}
        )
        moving = websocket.receive_json()
        while moving["speed"] == 0:
            moving = websocket.receive_json()
        assert moving["tick"] > initial["tick"]
        assert moving["speed"] > 0
        assert len(moving["sensors"]) == 5

        websocket.send_json({"type": "reset"})
        reset = websocket.receive_json()
        while reset["speed"] != 0:
            reset = websocket.receive_json()
        assert reset["speed"] == 0
        assert not reset["crashed"]
        assert len(reset["sensors"]) == 5


def test_websocket_sessions_are_isolated() -> None:
    with client.websocket_connect("/ws/play") as first, client.websocket_connect("/ws/play") as second:
        first.receive_json()
        second.receive_json()
        first.send_json(
            {"type": "input", "seq": 1, "throttle": True, "brake": False, "left": False, "right": False}
        )
        first_state = first.receive_json()
        while first_state["speed"] == 0:
            first_state = first.receive_json()
        second_state = second.receive_json()
        assert first_state["speed"] > 0
        assert second_state["speed"] == 0


def test_replay_catalog_latest_and_episode_endpoints(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = _write_replay_checkpoint(tmp_path)
    monkeypatch.setattr(server_module, "CHECKPOINT_PATH", checkpoint_path)

    catalog = client.get("/api/replays")
    assert catalog.status_code == 200
    assert catalog.json()["schema_version"] == 1
    assert catalog.json()["latest_training_episode"] == 1
    assert catalog.json()["replays"][0]["training_episode"] == 1

    latest = client.get("/api/replays/latest")
    selected = client.get("/api/replays/1")
    assert latest.status_code == selected.status_code == 200
    assert latest.json() == selected.json()
    assert latest.json()["steps"] == 1
    assert len(latest.json()["transitions"][0]["q_values"]) == 9
    assert client.get("/api/replays/999").status_code == 404

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    run_id = runs.json()["default_run_id"]
    selected_run = next(run for run in runs.json()["runs"] if run["run_id"] == run_id)
    assert selected_run["algorithm"] == "tabular"
    assert selected_run["evaluation_count"] == 1

    scoped_catalog = client.get(f"/api/runs/{run_id}/replays")
    scoped_latest = client.get(f"/api/runs/{run_id}/replays/latest")
    scoped_selected = client.get(f"/api/runs/{run_id}/replays/1")
    assert scoped_catalog.status_code == 200
    assert scoped_catalog.json()["run_id"] == run_id
    assert scoped_latest.json() == latest.json()
    assert scoped_selected.json() == selected.json()
    assert client.get("/api/runs/missing/replays").status_code == 404


def test_replay_endpoints_distinguish_missing_and_invalid_checkpoints(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(server_module, "CHECKPOINT_PATH", missing)
    assert client.get("/api/replays").status_code == 404

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(server_module, "CHECKPOINT_PATH", invalid)
    response = client.get("/api/replays")
    assert response.status_code == 503
    assert "invalid" in response.json()["detail"]


def _write_replay_checkpoint(tmp_path: Path) -> Path:
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
    path = tmp_path / "latest.json"
    save_checkpoint(checkpoint, path)
    return path
