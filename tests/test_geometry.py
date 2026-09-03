import pytest

from backend.env.geometry import Point, distance_to_segment
from backend.env.simulation import CAR_RADIUS
from backend.env.track import TRACK


def test_distance_to_segment_uses_closest_point() -> None:
    assert distance_to_segment(Point(5, 3), Point(0, 0), Point(10, 0)) == pytest.approx(3)
    assert distance_to_segment(Point(12, 0), Point(0, 0), Point(10, 0)) == pytest.approx(2)


def test_track_classifies_center_and_far_point() -> None:
    assert TRACK.is_on_track(TRACK.centerline[25], CAR_RADIUS)
    assert not TRACK.is_on_track(Point(100, 100), CAR_RADIUS)


def test_track_payload_contains_authoritative_geometry() -> None:
    payload = TRACK.as_dict()
    assert len(payload["centerline"]) >= 100
    assert payload["track_width"] == TRACK.half_width * 2
    assert payload["start_pose"]["heading"] == pytest.approx(TRACK.start_heading)
    assert set(payload) == {"centerline", "track_width", "start_pose", "finish_line", "halfway_gate"}


def test_nonadjacent_track_sections_do_not_overlap() -> None:
    points = TRACK.centerline
    for first_index, point in enumerate(points):
        for second_index in range(len(points)):
            separation = min(
                (first_index - second_index) % len(points),
                (second_index - first_index) % len(points),
            )
            if separation < 12:
                continue
            distance = distance_to_segment(
                point,
                points[second_index],
                points[(second_index + 1) % len(points)],
            )
            assert distance > TRACK.half_width * 2
