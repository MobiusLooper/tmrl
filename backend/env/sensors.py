from __future__ import annotations

from math import cos, inf, pi, sin, sqrt

from .geometry import Point
from .track import Track

SENSOR_ANGLES_DEGREES = (-60.0, -30.0, 0.0, 30.0, 60.0)
SENSOR_ANGLES_RADIANS = tuple(angle * pi / 180.0 for angle in SENSOR_ANGLES_DEGREES)
MAX_SENSOR_RANGE = 12.0
BOUNDARY_TOLERANCE = 1e-5


def raycast_track_boundary(
    track: Track,
    origin: Point,
    angle: float,
    max_range: float = MAX_SENSOR_RANGE,
) -> float:
    """Return the distance from an on-track origin to the first track boundary."""
    if max_range <= 0:
        raise ValueError("max_range must be positive")

    direction = Point(cos(angle), sin(angle))
    intervals: list[tuple[float, float]] = []
    points = track.centerline
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        interval = _ray_capsule_interval(origin, direction, start, end, track.half_width)
        if interval is None or interval[1] < 0 or interval[0] > max_range:
            continue
        intervals.append((max(0.0, interval[0]), min(max_range, interval[1])))

    # A track is the union of the capsules around its centreline segments. The
    # end of the connected interval containing t=0 is the first boundary hit.
    reach = 0.0
    for start, end in sorted(intervals):
        if start > reach + BOUNDARY_TOLERANCE:
            break
        reach = max(reach, end)
        if reach >= max_range:
            return max_range
    return reach


def sensor_readings(
    track: Track,
    position: Point,
    heading: float,
    max_range: float = MAX_SENSOR_RANGE,
) -> tuple[float, ...]:
    """Return normalized boundary distances in documented left-to-right order."""
    return tuple(
        min(1.0, max(0.0, raycast_track_boundary(track, position, heading + offset, max_range) / max_range))
        for offset in SENSOR_ANGLES_RADIANS
    )


def raw_observation(
    track: Track,
    position: Point,
    heading: float,
    speed: float,
    max_speed: float,
) -> tuple[float, ...]:
    """Return five normalized sensors followed by normalized speed."""
    if max_speed <= 0:
        raise ValueError("max_speed must be positive")
    normalized_speed = min(1.0, max(0.0, speed / max_speed))
    return (*sensor_readings(track, position, heading), normalized_speed)


def sensor_config() -> dict[str, object]:
    return {
        "angles": list(SENSOR_ANGLES_DEGREES),
        "max_range": MAX_SENSOR_RANGE,
    }


def _ray_capsule_interval(
    origin: Point,
    direction: Point,
    start: Point,
    end: Point,
    radius: float,
) -> tuple[float, float] | None:
    segment = end - start
    length = segment.length
    if length == 0:
        return _ray_circle_interval(origin, direction, start, radius)

    tangent = Point(segment.x / length, segment.y / length)
    normal = Point(-tangent.y, tangent.x)
    relative = origin - start
    local_origin = Point(relative.dot(tangent), relative.dot(normal))
    local_direction = Point(direction.dot(tangent), direction.dot(normal))

    intervals = [
        _ray_rectangle_interval(local_origin, local_direction, 0.0, length, -radius, radius),
        _ray_circle_interval(origin, direction, start, radius),
        _ray_circle_interval(origin, direction, end, radius),
    ]
    present = [interval for interval in intervals if interval is not None]
    if not present:
        return None
    return min(interval[0] for interval in present), max(interval[1] for interval in present)


def _ray_circle_interval(
    origin: Point,
    direction: Point,
    center: Point,
    radius: float,
) -> tuple[float, float] | None:
    offset = origin - center
    projection = offset.dot(direction)
    discriminant = projection * projection - (offset.dot(offset) - radius * radius)
    if discriminant < 0:
        return None
    root = sqrt(max(0.0, discriminant))
    return -projection - root, -projection + root


def _ray_rectangle_interval(
    origin: Point,
    direction: Point,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[float, float] | None:
    entry = -inf
    exit = inf
    for coordinate, velocity, lower, upper in (
        (origin.x, direction.x, min_x, max_x),
        (origin.y, direction.y, min_y, max_y),
    ):
        if abs(velocity) <= BOUNDARY_TOLERANCE:
            if coordinate < lower or coordinate > upper:
                return None
            continue
        first = (lower - coordinate) / velocity
        second = (upper - coordinate) / velocity
        entry = max(entry, min(first, second))
        exit = min(exit, max(first, second))
        if entry > exit:
            return None
    return entry, exit
