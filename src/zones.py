"""Parking-zone interpolation and JSON persistence."""

import json
from pathlib import Path
from typing import Sequence


def split_row(quad: Sequence[Sequence[float]], n: int) -> list[dict]:
    """Divide a four-corner row into slots by linear edge interpolation.

    Each slot's top and bottom edges span adjacent fractions of the source
    quadrilateral; this keeps all generated slot polygons aligned to the row.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(quad) != 4 or any(len(point) != 2 for point in quad):
        raise ValueError("quad must contain four 2D points")
    p0, p1, p2, p3 = [[float(value) for value in point] for point in quad]

    def interpolate(start, end, fraction):
        return [round(start[axis] + (end[axis] - start[axis]) * fraction) for axis in range(2)]

    zones = []
    for index in range(n):
        left, right = index / n, (index + 1) / n
        points = [
            interpolate(p0, p1, left),
            interpolate(p0, p1, right),
            interpolate(p3, p2, right),
            interpolate(p3, p2, left),
        ]
        zones.append({"points": points})
    return zones


def load_zones(path: str | Path) -> list[dict]:
    """Load the repository's list-of-points zone JSON format from disk."""
    with Path(path).open(encoding="utf-8") as stream:
        zones = json.load(stream)
    if not isinstance(zones, list) or any("points" not in zone for zone in zones):
        raise ValueError(f"Invalid zone file: {path}")
    return zones


def save_zones(path: str | Path, zones: list[dict]) -> None:
    """Save point polygons in the existing JSON format, creating the parent folder."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(zones, stream, indent=4)
