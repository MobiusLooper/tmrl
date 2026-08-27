from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point:
        return Point(self.x * scalar, self.y * scalar)

    def dot(self, other: Point) -> float:
        return self.x * other.x + self.y * other.y

    @property
    def length(self) -> float:
        return hypot(self.x, self.y)

    def normalized(self) -> Point:
        length = self.length
        if length == 0:
            return Point(0.0, 0.0)
        return Point(self.x / length, self.y / length)


def distance_to_segment(point: Point, start: Point, end: Point) -> float:
    segment = end - start
    length_squared = segment.dot(segment)
    if length_squared == 0:
        return (point - start).length

    projection = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
    closest = start + segment * projection
    return (point - closest).length


def catmull_rom_closed(control_points: tuple[Point, ...], samples_per_curve: int = 10) -> tuple[Point, ...]:
    """Sample a smooth closed curve that passes through every control point."""
    if len(control_points) < 4:
        raise ValueError("A closed Catmull-Rom curve needs at least four points")
    if samples_per_curve < 1:
        raise ValueError("samples_per_curve must be positive")

    points: list[Point] = []
    count = len(control_points)
    for index in range(count):
        p0 = control_points[(index - 1) % count]
        p1 = control_points[index]
        p2 = control_points[(index + 1) % count]
        p3 = control_points[(index + 2) % count]
        for sample in range(samples_per_curve):
            t = sample / samples_per_curve
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                2 * p1.x
                + (-p0.x + p2.x) * t
                + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2
                + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
            )
            y = 0.5 * (
                2 * p1.y
                + (-p0.y + p2.y) * t
                + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2
                + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
            )
            points.append(Point(x, y))
    return tuple(points)
