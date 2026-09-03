from math import atan2, cos, pi, sin, sqrt

import pytest

from backend.env.geometry import Point
from backend.env.sensors import (
    MAX_SENSOR_RANGE,
    SENSOR_ANGLES_RADIANS,
    raw_observation,
    raycast_track_boundary,
    sensor_readings,
)
from backend.env.track import TRACK, Gate, Track


def rectangular_track() -> Track:
    centerline = (Point(0, 0), Point(20, 0), Point(20, 20), Point(0, 20))
    finish = Gate(Point(0, 0), Point(1, 0), Point(0, 2), Point(0, -2))
    halfway = Gate(Point(20, 20), Point(-1, 0), Point(20, 18), Point(20, 22))
    return Track(centerline, 2, Point(10, 0), 0, finish, halfway)


def test_raycast_finds_perpendicular_and_angled_boundaries() -> None:
    track = rectangular_track()
    origin = Point(10, 0)

    assert raycast_track_boundary(track, origin, pi / 2) == pytest.approx(2, abs=1e-4)
    assert raycast_track_boundary(track, origin, pi / 4) == pytest.approx(2 * sqrt(2), abs=1e-4)


def test_raycast_handles_range_boundary_and_off_track_origins() -> None:
    track = rectangular_track()

    assert raycast_track_boundary(track, Point(10, 0), 0, max_range=5) == 5
    assert raycast_track_boundary(track, Point(10, 2), pi / 2) == pytest.approx(0, abs=1e-4)
    assert raycast_track_boundary(track, Point(10, 10), 0) == 0


def test_sensor_readings_are_normalized_and_keep_angle_order() -> None:
    readings = sensor_readings(rectangular_track(), Point(10, 1), 0)

    assert len(readings) == 7
    assert all(0 <= reading <= 1 for reading in readings)
    assert readings == pytest.approx(
        (
            3 / MAX_SENSOR_RANGE,
            2 * sqrt(3) / MAX_SENSOR_RANGE,
            6 / MAX_SENSOR_RANGE,
            1,
            2 / MAX_SENSOR_RANGE,
            2 / sqrt(3) / MAX_SENSOR_RANGE,
            1 / MAX_SENSOR_RANGE,
        ),
        abs=1e-4,
    )


def test_heading_rotates_sensor_fan() -> None:
    track = rectangular_track()
    origin = Point(10, 1)

    facing_up = sensor_readings(track, origin, pi / 2)
    facing_down = sensor_readings(track, origin, -pi / 2)

    assert facing_up[3] == pytest.approx(1 / MAX_SENSOR_RANGE, abs=1e-4)
    assert facing_down[3] == pytest.approx(3 / MAX_SENSOR_RANGE, abs=1e-4)


def test_raw_observation_adds_normalized_speed() -> None:
    observation = raw_observation(rectangular_track(), Point(10, 0), 0, speed=6, max_speed=12)

    assert len(observation) == 8
    assert observation[:7] == sensor_readings(rectangular_track(), Point(10, 0), 0)
    assert observation[7] == 0.5


def test_direct_intersections_match_distance_field_reference_on_real_track() -> None:
    points = TRACK.centerline
    for index in (0, 30, 60, 90, 120, 150):
        origin = points[index]
        before = points[(index - 1) % len(points)]
        after = points[(index + 1) % len(points)]
        heading = atan2(after.y - before.y, after.x - before.x)
        for offset in SENSOR_ANGLES_RADIANS:
            direct = raycast_track_boundary(TRACK, origin, heading + offset)
            reference = _reference_raycast(TRACK, origin, heading + offset)
            assert direct == pytest.approx(reference, abs=1e-4)


def _reference_raycast(track: Track, origin: Point, angle: float) -> float:
    """Independent distance-field marcher retained only as a correctness oracle."""
    if not track.is_on_track(origin):
        return 0.0
    direction = Point(cos(angle), sin(angle))
    distance = 0.0
    for _ in range(256):
        point = origin + direction * distance
        clearance = track.half_width - track.distance_from_centerline(point)
        if clearance <= 1e-7:
            return distance
        next_distance = min(MAX_SENSOR_RANGE, distance + clearance)
        if next_distance == MAX_SENSOR_RANGE:
            endpoint = origin + direction * MAX_SENSOR_RANGE
            if track.is_on_track(endpoint):
                return MAX_SENSOR_RANGE
            inside = distance
            outside = MAX_SENSOR_RANGE
            for _ in range(30):
                midpoint = (inside + outside) / 2
                if track.is_on_track(origin + direction * midpoint):
                    inside = midpoint
                else:
                    outside = midpoint
            return (inside + outside) / 2
        distance = next_distance
    raise AssertionError("reference raycast did not converge")
