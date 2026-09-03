from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from math import atan2
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import backend.api.server as server_module
import backend.training.demonstrations as demonstration_module
from backend.api.server import app
from backend.env.environment import DiscreteAction, RacingEnv
from backend.env.simulation import Action, DT, RacingSimulation
from backend.training.demonstrations import (
    MAX_QUALIFYING_LAP_TIME,
    DemonstrationConflictError,
    DemonstrationDataset,
    DemonstrationError,
    DemonstrationLap,
    DemonstrationStore,
    DemonstrationTransition,
    LapRecorder,
    RecordingEvent,
    active_dataset_id,
    aggregate_demonstration_laps,
    environment_metadata,
    load_demonstration_dataset,
    normalize_action,
    validate_lap,
)

OBSERVATION = (0.1,) * 12


def _fake_lap(lap_id: str, lap_time: float = 20.0) -> dict[str, object]:
    return {
        "lap_id": lap_id,
        "recorded_at": "2026-08-30T10:00:00Z",
        "lap_time": lap_time,
        "steps": 1,
        "initial_state": {},
        "transitions": [{}],
    }


def _legacy_payload(laps: list[dict[str, object]], *, status: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "human-demonstrations",
        "dataset": {
            "dataset_id": "20260830T100000Z-human-1234abcd",
            "created_at": "2026-08-30T10:00:00Z",
            "updated_at": "2026-08-30T10:00:00Z",
            "status": status,
            "lap_count": len(laps),
        },
        "environment": demonstration_module._legacy_environment_metadata(),
        "laps": laps,
    }


def test_normalize_action_maps_conflicting_axes_to_neutral() -> None:
    assert normalize_action(Action(throttle=True)) == DiscreteAction.THROTTLE
    assert normalize_action(Action(throttle=True, left=True)) == DiscreteAction.LEFT_THROTTLE
    assert normalize_action(Action(throttle=True, brake=True, left=True)) == DiscreteAction.LEFT
    assert normalize_action(Action(left=True, right=True)) == DiscreteAction.COAST


def test_store_creates_one_stable_active_library(tmp_path: Path) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")

    first = store.current()
    second = store.current()
    summary = store.summary(first)

    assert first == second
    assert summary == {
        "dataset_id": active_dataset_id(),
        "created_at": first["dataset"]["created_at"],  # type: ignore[index]
        "updated_at": first["dataset"]["updated_at"],  # type: ignore[index]
        "status": "active",
        "lap_count": 0,
        "max_lap_time_exclusive": 30.0,
        "path": f"artifacts/demonstrations/{active_dataset_id()}.json",
    }
    assert store.list() == [summary]


def test_first_access_merges_compatible_legacy_laps_without_deleting_sources(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "artifacts" / "demonstrations"
    directory.mkdir(parents=True)
    complete_path = directory / "20260830T100000Z-human-1234abcd.json"
    draft_path = directory / "20260830T110000Z-human-5678abcd.json"
    complete_path.write_text(
        json.dumps(_legacy_payload([_fake_lap("lap-a"), _fake_lap("lap-b", 29.9)], status="complete")),
        encoding="utf-8",
    )
    draft = _legacy_payload(
        [_fake_lap("lap-b", 29.9), _fake_lap("lap-c", 22.0), _fake_lap("lap-slow", 30.0)],
        status="draft",
    )
    draft["dataset"]["dataset_id"] = "20260830T110000Z-human-5678abcd"  # type: ignore[index]
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    store = DemonstrationStore(directory)
    active = store.current()

    assert active["schema_version"] == 2
    assert active["dataset"]["status"] == "active"  # type: ignore[index]
    assert active["dataset"]["lap_count"] == 3  # type: ignore[index]
    assert [lap["lap_id"] for lap in active["laps"]] == ["lap-a", "lap-b", "lap-c"]  # type: ignore[index]
    assert complete_path.exists()
    assert draft_path.exists()


def test_store_appends_concurrently_without_lost_laps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")
    store.current()
    monkeypatch.setattr(demonstration_module, "validate_lap", lambda lap: None)

    laps = [_fake_lap(f"lap-{index}") for index in range(20)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store.append, laps))

    active = store.current()
    assert active["dataset"]["lap_count"] == 20  # type: ignore[index]
    assert {lap["lap_id"] for lap in active["laps"]} == {  # type: ignore[index]
        f"lap-{index}" for index in range(20)
    }
    store.append(_fake_lap("lap-0"))
    assert store.summary(store.current())["lap_count"] == 20


def test_store_enforces_strict_sub_30_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")
    monkeypatch.setattr(demonstration_module, "validate_lap", lambda lap: None)

    with pytest.raises(DemonstrationConflictError, match="under 30.0s"):
        store.append(_fake_lap("lap-equal", 30.0))
    assert store.summary(store.current())["lap_count"] == 0


def test_atomic_append_preserves_existing_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")
    store.current()
    before = store.path.read_bytes()
    monkeypatch.setattr(demonstration_module, "validate_lap", lambda lap: None)

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("interrupted")

    monkeypatch.setattr(demonstration_module.os, "replace", fail_replace)
    with pytest.raises(DemonstrationError, match="interrupted"):
        store.append(_fake_lap("lap-test"))
    assert store.path.read_bytes() == before


def test_training_loader_supports_legacy_complete_and_active_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = {
        "tick": 0,
        "x": 1.0,
        "y": 2.0,
        "heading": 0.0,
        "speed": 0.0,
        "crashed": False,
        "sensors": [1.0] * 7,
    }
    final = {**initial, "tick": 1, "x": 1.1, "speed": 1.0}
    lap = {
        "lap_id": "lap-loader",
        "recorded_at": "2026-08-30T10:00:00Z",
        "lap_time": 20.0,
        "steps": 1,
        "initial_state": initial,
        "transitions": [
            {
                "tick": 1,
                "action": int(DiscreteAction.THROTTLE),
                "action_name": DiscreteAction.THROTTLE.name,
                "state": final,
            }
        ],
    }

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.index = 0

        def reset(self) -> tuple[float, ...]:
            self.index = 0
            return OBSERVATION

        def render_state(self) -> dict[str, object]:
            return initial if self.index == 0 else final

        def step(self, action: DiscreteAction) -> SimpleNamespace:
            assert action == DiscreteAction.THROTTLE
            self.index = 1
            return SimpleNamespace(
                reward=1.0,
                observation=OBSERVATION,
                done=True,
                info={"termination_reason": "lap"},
            )

    monkeypatch.setattr(demonstration_module, "validate_lap", lambda raw_lap: None)
    monkeypatch.setattr(demonstration_module, "RacingEnv", FakeEnvironment)

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(_legacy_payload([lap], status="complete")), encoding="utf-8")
    legacy = load_demonstration_dataset(legacy_path)
    assert [item.lap_id for item in legacy.laps] == ["lap-loader"]

    active_path = tmp_path / "active.json"
    active_payload = {
        "schema_version": 2,
        "kind": "human-demonstrations",
        "dataset": {
            "dataset_id": active_dataset_id(),
            "created_at": "2026-08-30T10:00:00Z",
            "updated_at": "2026-08-30T10:00:00Z",
            "status": "active",
            "lap_count": 1,
        },
        "qualification": {"max_lap_time_exclusive": 30.0},
        "environment": environment_metadata(),
        "laps": [lap],
    }
    active_path.write_text(json.dumps(active_payload), encoding="utf-8")
    loaded = load_demonstration_dataset(active_path)
    original_digest = loaded.digest
    active_payload["laps"].append({**lap, "lap_id": "later-lap"})
    active_payload["dataset"]["lap_count"] = 2
    active_path.write_text(json.dumps(active_payload), encoding="utf-8")

    assert loaded.digest == original_digest
    assert [item.lap_id for item in loaded.laps] == ["lap-loader"]


def test_demonstration_aggregation_uses_discount_and_terminal_short_chunk(tmp_path: Path) -> None:
    transitions = (
        DemonstrationTransition(OBSERVATION, DiscreteAction.THROTTLE, 1.0, OBSERVATION, False),
        DemonstrationTransition(OBSERVATION, DiscreteAction.THROTTLE, 2.0, OBSERVATION, False),
        DemonstrationTransition(OBSERVATION, DiscreteAction.LEFT, 3.0, OBSERVATION, True),
    )
    dataset = DemonstrationDataset(
        tmp_path / "demo.json", "demo", "digest", (DemonstrationLap("lap", transitions),)
    )

    laps = aggregate_demonstration_laps(dataset, action_repeat=2, discount=0.5)

    assert len(laps[0].transitions) == 2
    assert laps[0].transitions[0].reward == 2.0
    assert laps[0].transitions[0].duration == 2
    assert laps[0].transitions[1].reward == 3.0
    assert laps[0].transitions[1].duration == 1
    assert laps[0].transitions[1].done


def test_recorder_starts_automatically_holds_actions_and_cancel_discards() -> None:
    recorder = LapRecorder()
    simulation = RacingSimulation()

    snapshot = recorder.start(simulation, reset=False)
    assert snapshot["tick"] == 0
    assert recorder.active
    recorder.step(simulation, Action(throttle=True))
    recorder.step(simulation, Action(right=True))
    assert recorder.transitions is not None
    assert [item["action"] for item in recorder.transitions] == [
        int(DiscreteAction.THROTTLE),
        int(DiscreteAction.THROTTLE),
    ]
    discarded = recorder.cancel()
    assert discarded is not None and discarded.status == "discarded"
    assert not recorder.active


def _position_for_finish(simulation: RacingSimulation, lap_time: float) -> None:
    gate = simulation.track.finish_gate
    simulation.passed_halfway = True
    simulation.lap_started = True
    simulation.current_lap_time = lap_time - DT
    simulation.car.x = gate.center.x - 0.01 * gate.tangent.x
    simulation.car.y = gate.center.y - 0.01 * gate.tangent.y
    simulation.car.heading = atan2(gate.tangent.y, gate.tangent.x)
    simulation.car.speed = 1.0


def test_recorder_discards_crash_and_waits_for_reset_after_finish() -> None:
    simulation = RacingSimulation()
    recorder = LapRecorder()
    recorder.start(simulation, reset=False)
    simulation.car.x = 1_000
    _, event = recorder.step(simulation, Action(throttle=True))
    assert event is not None and event.status == "discarded"
    assert not recorder.active

    recorder.start(simulation)
    _position_for_finish(simulation, 23.6)
    _, event = recorder.step(simulation, Action())
    assert event is not None and event.status == "qualifying"
    assert event.lap_time == pytest.approx(23.6)
    assert event.lap is not None
    assert not recorder.active

    transition_count = len(event.lap["transitions"])  # type: ignore[arg-type]
    recorder.step(simulation, Action(throttle=True))
    assert not recorder.active
    assert len(event.lap["transitions"]) == transition_count  # type: ignore[arg-type]


def test_recorder_rejects_lap_at_exact_threshold() -> None:
    simulation = RacingSimulation()
    recorder = LapRecorder()
    recorder.start(simulation, reset=False)
    _position_for_finish(simulation, MAX_QUALIFYING_LAP_TIME)

    _, event = recorder.step(simulation, Action())

    assert event is not None and event.status == "rejected"
    assert event.lap is None
    assert "30.0s limit" in event.message
    assert not recorder.active


def test_lap_validation_rejects_unpaired_actions_and_incomplete_routes() -> None:
    environment = RacingEnv()
    environment.reset()
    initial = environment.render_state()
    environment.step(DiscreteAction.THROTTLE)
    first_state = environment.render_state()
    environment.step(DiscreteAction.LEFT_THROTTLE)
    second_state = environment.render_state()
    transitions = [
        {
            "action": int(DiscreteAction.THROTTLE),
            "action_name": DiscreteAction.THROTTLE.name,
            "state": first_state,
        },
        {
            "action": int(DiscreteAction.LEFT_THROTTLE),
            "action_name": DiscreteAction.LEFT_THROTTLE.name,
            "state": second_state,
        },
    ]
    lap = {"steps": 2, "initial_state": initial, "transitions": transitions}
    with pytest.raises(DemonstrationError, match="held for two"):
        validate_lap(lap)

    incomplete = {
        "steps": 1,
        "initial_state": initial,
        "transitions": [
            {
                "action": int(DiscreteAction.THROTTLE),
                "action_name": DiscreteAction.THROTTLE.name,
                "state": first_state,
            }
        ],
    }
    with pytest.raises(DemonstrationError, match="do not complete"):
        validate_lap(incomplete)


def test_lap_validation_does_not_apply_training_stall_limit() -> None:
    environment = RacingEnv(stall_steps=1_000)
    environment.reset()
    initial = environment.render_state()
    transitions = []
    for _ in range(102):
        result = environment.step(DiscreteAction.COAST)
        transitions.append(
            {
                "action": int(DiscreteAction.COAST),
                "action_name": DiscreteAction.COAST.name,
                "state": environment.render_state(),
            }
        )
        assert not result.done
    lap = {"steps": len(transitions), "initial_state": initial, "transitions": transitions}

    with pytest.raises(DemonstrationError, match="do not complete"):
        validate_lap(lap)


def test_current_demonstration_api_returns_active_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")
    monkeypatch.setattr(server_module, "DEMONSTRATION_STORE", store)

    response = TestClient(app).get("/api/demonstrations/current")

    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert response.json()["lap_count"] == 0
    assert response.json()["path"].startswith("artifacts/demonstrations/track-")


def test_websocket_auto_records_and_reset_restarts_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")
    monkeypatch.setattr(server_module, "DEMONSTRATION_STORE", store)
    client = TestClient(app)

    with client.websocket_connect("/ws/play") as websocket:
        assert websocket.receive_json()["type"] == "state"
        started = _receive_recording_status(websocket)
        assert started["status"] == "started"
        assert started["lap_count"] == 0

        websocket.send_json({"type": "manual_pause"})
        assert _receive_recording_status(websocket)["status"] == "discarded"
        websocket.send_json({"type": "reset"})
        assert _receive_recording_status(websocket)["status"] == "started"


def test_slow_save_does_not_block_another_websocket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DemonstrationStore(tmp_path / "artifacts" / "demonstrations")
    monkeypatch.setattr(server_module, "DEMONSTRATION_STORE", store)
    monkeypatch.setattr(demonstration_module, "validate_lap", lambda lap: None)
    original_append = store.append

    def slow_append(lap: dict[str, object]) -> dict[str, object]:
        time.sleep(0.5)
        return original_append(lap)

    monkeypatch.setattr(store, "append", slow_append)
    recorders: list[LapRecorder] = []
    original_start = LapRecorder.start
    original_step = LapRecorder.step
    emitted = False

    def tracked_start(
        recorder: LapRecorder, simulation: RacingSimulation, *, reset: bool = True
    ) -> dict[str, object]:
        if recorder not in recorders:
            recorders.append(recorder)
        return original_start(recorder, simulation, reset=reset)

    def qualifying_first(
        recorder: LapRecorder, simulation: RacingSimulation, action: Action
    ) -> tuple[dict[str, object], RecordingEvent | None]:
        nonlocal emitted
        if recorders and recorder is recorders[0] and not emitted:
            emitted = True
            return simulation.step(Action()), RecordingEvent(
                "qualifying", "Saving lap…", _fake_lap("lap-slow-save"), 20.0
            )
        return original_step(recorder, simulation, action)

    monkeypatch.setattr(LapRecorder, "start", tracked_start)
    monkeypatch.setattr(LapRecorder, "step", qualifying_first)
    client = TestClient(app)

    with client.websocket_connect("/ws/play") as first, client.websocket_connect("/ws/play") as second:
        _receive_recording_status(first)
        _receive_recording_status(second)
        assert _receive_recording_status(first)["status"] == "saving"

        started = time.monotonic()
        assert _receive_state(second)["type"] == "state"
        assert time.monotonic() - started < 0.35

        saved = _receive_recording_status(first)
        assert saved["status"] == "saved"
        assert saved["lap_count"] == 1


def _receive_recording_status(websocket: object) -> dict[str, object]:
    receive_json = getattr(websocket, "receive_json")
    while True:
        payload = receive_json()
        if payload.get("type") == "recording_status":
            return payload


def _receive_state(websocket: object) -> dict[str, object]:
    receive_json = getattr(websocket, "receive_json")
    while True:
        payload = receive_json()
        if payload.get("type") == "state":
            return payload
