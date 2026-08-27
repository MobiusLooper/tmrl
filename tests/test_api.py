from fastapi.testclient import TestClient

from backend.api.server import app


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
