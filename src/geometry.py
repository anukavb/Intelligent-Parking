"""Pure parking-zone assignment geometry."""

from collections.abc import Sequence

import cv2
import numpy as np


def _polygon(points: Sequence[Sequence[float]]) -> np.ndarray:
    return cv2.convexHull(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))


def footprint_box(box: Sequence[float], frac: float = 0.4) -> np.ndarray:
    """Return the bottom `frac` of a vehicle box as a four-corner polygon.

    The bottom strip approximates a vehicle's ground-contact footprint and
    reduces false slot overlap from the upper part of a vehicle on angled views.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    x1, y1, x2, y2 = map(float, box)
    top = y2 - (y2 - y1) * frac
    return np.asarray(((x1, top), (x2, top), (x2, y2), (x1, y2)), dtype=np.float32)


def overlap_ratio(footprint: Sequence[Sequence[float]], polygon: Sequence[Sequence[float]]) -> float:
    """Return intersection area divided by footprint area for two convex polygons.

    Convex hulls normalize corner order for OpenCV's convex intersection
    routine; the ratio expresses how much of the vehicle footprint lies in a slot.
    """
    subject = _polygon(footprint)
    subject_area = abs(float(cv2.contourArea(subject)))
    if subject_area <= 0:
        return 0.0
    zone = _polygon(polygon)
    intersection_area, _ = cv2.intersectConvexConvex(subject, zone)
    return min(1.0, max(0.0, float(intersection_area) / subject_area))


def _box_polygon(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)


def centroid_assign(boxes: Sequence[Sequence[float]], polygons: Sequence[Sequence[Sequence[float]]]) -> tuple[list[int], list[int], list[dict]]:
    """Apply the baseline centroid-in-polygon rule for ablation comparisons."""
    occupied = [0] * len(polygons)
    assigned = [-1] * len(boxes)
    for vehicle_index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(float, box)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        for slot_index, polygon in enumerate(polygons):
            contour = _polygon(polygon)
            if cv2.pointPolygonTest(contour, center, False) >= 0:
                occupied[slot_index] = 1
                assigned[vehicle_index] = slot_index
                break
    return occupied, assigned, []


def assign_vehicles(
    boxes: Sequence[Sequence[float]],
    polygons: Sequence[Sequence[Sequence[float]]],
    frac: float = 0.4,
    T: float = 0.30,
    straddle_T: float = 0.25,
) -> tuple[list[int], list[int], list[dict]]:
    """Assign each vehicle to one best-overlap slot and report possible straddlers.

    A bottom vehicle footprint is compared with every slot, assigned only to
    the best slot when overlap reaches `T`, and flagged if its runner-up reaches
    `straddle_T`.
    """
    if not 0 <= T <= 1 or not 0 <= straddle_T <= 1:
        raise ValueError("overlap thresholds must be in [0, 1]")
    occupied = [0] * len(polygons)
    assigned = [-1] * len(boxes)
    straddlers = []
    for vehicle_index, box in enumerate(boxes):
        footprint = footprint_box(box, frac)
        overlaps = [overlap_ratio(footprint, polygon) for polygon in polygons]
        ranked = sorted(enumerate(overlaps), key=lambda item: item[1], reverse=True)
        if ranked and ranked[0][1] >= T:
            slot_index = ranked[0][0]
            occupied[slot_index] = 1
            assigned[vehicle_index] = slot_index
        if len(ranked) > 1 and ranked[1][1] >= straddle_T:
            straddlers.append(
                {
                    "vehicle_index": vehicle_index,
                    "assigned_slot": assigned[vehicle_index],
                    "overlaps": overlaps,
                }
            )
    return occupied, assigned, straddlers
