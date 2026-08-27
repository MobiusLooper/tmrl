import pytest

from backend.env.geometry import Point
from backend.env.simulation import (
    ACCELERATION,
    DT,
    MAX_SPEED,
    STEERING_RATE,
    Action,
    RacingSimulation,
)
from backend.env.track import Gate, Track


def broad_test_track() -> Track:
    centerline = (Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100))
    finish = Gate(Point(0, 0), Point(1, 0), Point(0, 10), Point(0, -10))
    halfway = Gate(Point(100, 100), Point(-1, 0), Point(100, 90), Point(100, 110))
    return Track(centerline, 10, Point(1, 0), 0, finish, halfway)


def test_throttle_brake_coast_and_speed_clamp() -> None:
    simulation = RacingSimulation(broad_test_track())
    simulation.step(Action(throttle=True))
    assert simulation.car.speed == pytest.approx(ACCELERATION * DT)

    coast_speed = simulation.car.speed
    simulation.step(Action())
    assert simulation.car.speed == coast_speed

    simulation.step(Action(brake=True))
    assert simulation.car.speed == 0
    simulation.car.speed = MAX_SPEED
    simulation.step(Action(throttle=True))
    assert simulation.car.speed == MAX_SPEED


def test_contradictory_inputs_are_ignored() -> None:
    simulation = RacingSimulation(broad_test_track())
    simulation.car.speed = 3
    simulation.step(Action(throttle=True, brake=True, left=True, right=True))
    assert simulation.car.speed == 3
    assert simulation.car.heading == 0


def test_stationary_car_cannot_steer() -> None:
    simulation = RacingSimulation(broad_test_track())
    simulation.step(Action(left=True))
    assert simulation.car.heading == simulation.track.start_heading
    assert simulation.current_lap_time == 0


def test_combined_throttle_and_steering_are_deterministic() -> None:
    first = RacingSimulation(broad_test_track())
    second = RacingSimulation(broad_test_track())
    for _ in range(5):
        first.step(Action(throttle=True, left=True))
        second.step(Action(throttle=True, left=True))
    assert first.snapshot() == second.snapshot()
    assert first.car.heading == pytest.approx(STEERING_RATE * DT * 5)
    assert first.car.x != first.track.start_position.x
    assert first.car.y > first.track.start_position.y


def test_crash_freezes_until_reset() -> None:
    simulation = RacingSimulation(broad_test_track())
    simulation.car.x = 50
    simulation.car.y = 50
    simulation.step(Action())
    assert simulation.crashed
    assert simulation.snapshot()["sensors"] == [0.0] * 5
    frozen = simulation.snapshot()
    simulation.step(Action(throttle=True, left=True))
    assert simulation.snapshot() == frozen

    simulation.reset()
    assert not simulation.crashed
    assert simulation.car.speed == 0


def test_observation_contains_sensors_and_normalized_speed() -> None:
    simulation = RacingSimulation(broad_test_track())
    simulation.car.speed = MAX_SPEED / 2

    observation = simulation.observation()

    assert len(observation) == 6
    assert observation[:5] == tuple(simulation.snapshot()["sensors"])
    assert observation[5] == 0.5


def test_finish_requires_halfway_gate_and_forward_direction() -> None:
    simulation = RacingSimulation(broad_test_track())
    simulation.car.x = -0.01
    simulation.car.y = 0
    simulation.car.heading = 0
    simulation.car.speed = 1
    simulation.step(Action())
    assert simulation.laps == 0

    simulation.reset()
    simulation.passed_halfway = True
    simulation.current_lap_time = 5
    simulation.car.x = -0.01
    simulation.car.y = 0
    simulation.car.heading = 0
    simulation.car.speed = 1
    simulation.step(Action())
    assert simulation.laps == 1
    assert simulation.last_lap_time == pytest.approx(5 + DT)
    assert simulation.best_lap_time == simulation.last_lap_time

    simulation.reset()
    simulation.passed_halfway = True
    simulation.car.x = 0.01
    simulation.car.heading = 3.141592653589793
    simulation.car.speed = 1
    simulation.step(Action())
    assert simulation.laps == 0
