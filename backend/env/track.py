from __future__ import annotations

from dataclasses import dataclass
from math import atan2

from .geometry import Point, catmull_rom_closed, distance_to_segment


@dataclass(frozen=True, slots=True)
class Gate:
    center: Point
    tangent: Point
    start: Point
    end: Point

    def signed_progress(self, point: Point) -> float:
        return (point - self.center).dot(self.tangent)

    def contains_crossing(self, previous: Point, current: Point, half_width: float) -> bool:
        crossed_forward = self.signed_progress(previous) < 0 <= self.signed_progress(current)
        normal = Point(-self.tangent.y, self.tangent.x)
        lateral_offset = abs((current - self.center).dot(normal))
        return crossed_forward and lateral_offset <= half_width


@dataclass(frozen=True, slots=True)
class Track:
    centerline: tuple[Point, ...]
    half_width: float
    start_position: Point
    start_heading: float
    finish_gate: Gate
    halfway_gate: Gate

    def distance_from_centerline(self, point: Point) -> float:
        return min(
            distance_to_segment(point, self.centerline[index], self.centerline[(index + 1) % len(self.centerline)])
            for index in range(len(self.centerline))
        )

    def is_on_track(self, point: Point, car_radius: float = 0.0) -> bool:
        return self.distance_from_centerline(point) + car_radius <= self.half_width

    def as_dict(self) -> dict[str, object]:
        return {
            "centerline": [[point.x, point.y] for point in self.centerline],
            "track_width": self.half_width * 2,
            "start_pose": {
                "x": self.start_position.x,
                "y": self.start_position.y,
                "heading": self.start_heading,
            },
            "finish_line": _gate_as_dict(self.finish_gate),
            "halfway_gate": _gate_as_dict(self.halfway_gate),
        }


def _gate_as_dict(gate: Gate) -> dict[str, object]:
    return {
        "start": [gate.start.x, gate.start.y],
        "end": [gate.end.x, gate.end.y],
    }


def _make_gate(centerline: tuple[Point, ...], index: int, half_width: float) -> Gate:
    before = centerline[(index - 1) % len(centerline)]
    after = centerline[(index + 1) % len(centerline)]
    tangent = (after - before).normalized()
    normal = Point(-tangent.y, tangent.x)
    center = centerline[index]
    margin = 0.12
    return Gate(
        center=center,
        tangent=tangent,
        start=center + normal * (half_width + margin),
        end=center - normal * (half_width + margin),
    )


def build_track() -> Track:
    # Clockwise/counter-clockwise variety, a lower straight, eastern sweeper,
    # northern hairpin and a left-right chicane through the middle.
    controls = (
        Point(7, 6),
        Point(20, 6),
        Point(34, 6),
        Point(45, 11),
        Point(45, 21),
        Point(41, 31),
        Point(28, 35),
        Point(21, 31),
        Point(25, 24),
        Point(35, 24),
        Point(36, 17),
        Point(30, 14),
        Point(25, 19),
        Point(20, 14),
        Point(16, 22),
        Point(15, 29),
        Point(8, 31),
        Point(2, 25),
        Point(2, 13),
    )
    centerline = catmull_rom_closed(controls, samples_per_curve=10)
    half_width = 2.0
    finish_index = 0
    halfway_index = len(centerline) // 2
    finish_gate = _make_gate(centerline, finish_index, half_width)
    halfway_gate = _make_gate(centerline, halfway_index, half_width)
    start_position = finish_gate.center + finish_gate.tangent * 0.8
    start_heading = atan2(finish_gate.tangent.y, finish_gate.tangent.x)
    return Track(
        centerline=centerline,
        half_width=half_width,
        start_position=start_position,
        start_heading=start_heading,
        finish_gate=finish_gate,
        halfway_gate=halfway_gate,
    )


TRACK = build_track()
