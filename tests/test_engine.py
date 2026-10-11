import cv2
import numpy as np
import pytest

from src.analytics import summarize
from src.engine import Detections, ParkingEngine
from src.geometry import assign_vehicles, centroid_assign, footprint_box, overlap_ratio
from src.smoothing import SlotStateMachine


def raster_overlap(footprint, polygon):
    canvas = np.zeros((120, 120), dtype=np.uint8)
    vehicle = np.zeros_like(canvas)
    cv2.fillPoly(canvas, [np.asarray(polygon, np.int32)], 1)
    cv2.fillPoly(vehicle, [np.asarray(footprint, np.int32)], 1)
    total = int(vehicle.sum())
    return int((canvas & vehicle).sum()) / total if total else 0.0


def test_overlap_ratio_matches_rasterized_masks():
    rng = np.random.default_rng(17)
    for _ in range(24):
        x, y = rng.integers(10, 70, size=2)
        footprint = np.array([[x, y], [x + 18, y], [x + 18, y + 12], [x, y + 12]], np.float32)
        px, py = rng.integers(5, 90, size=2)
        polygon = np.array([[px, py], [px + 22, py + 2], [px + 20, py + 24], [px - 2, py + 20]], np.float32)
        assert overlap_ratio(footprint, polygon) == pytest.approx(raster_overlap(footprint, polygon), abs=0.08)


def test_footprint_and_vehicle_assignment_only_select_one_slot():
    assert np.allclose(footprint_box((0, 0, 10, 20), 0.4), [[0, 12], [10, 12], [10, 20], [0, 20]])
    slots = [[[0, 0], [15, 0], [15, 20], [0, 20]], [[15, 0], [30, 0], [30, 20], [15, 20]]]
    occupied, assigned, straddlers = assign_vehicles([(8, 0, 22, 20)], slots, frac=1, T=0.3, straddle_T=0.25)
    assert sum(occupied) == 1
    assert assigned[0] in (0, 1)
    assert len(straddlers) == 1
    raw, baseline, _ = centroid_assign([(8, 0, 22, 20)], slots)
    assert sum(raw) <= 2
    assert baseline[0] == 0


def test_state_machine_flicker_and_sustained_observations():
    machine = SlotStateMachine(1, alpha=0.3, on=0.65, off=0.35)
    assert machine.update([0]) == [0]
    assert machine.update([1]) == [0]
    assert machine.update([1]) == [0]
    assert machine.update([1]) == [1]
    assert machine.update([0]) == [1]
    assert machine.update([0]) == [0]


def test_engine_stub_detector_and_roi_translation():
    zones = [{"points": [[10, 10], [80, 10], [80, 80], [10, 80]]}]
    engine = ParkingEngine(zones, detector=lambda frame: ([(5, 5, 25, 35)], [2], [0.9]), roi=True, roi_margin=5)
    dets, decision = engine.step(np.zeros((100, 100, 3), dtype=np.uint8))
    assert dets.boxes.tolist() == [[10.0, 10.0, 30.0, 40.0]]
    assert decision.raw == [1]
    assert decision.states == [1]
    assert decision.counts == {"occupied": 1, "available": 0}


def test_analytics_flip_count_and_sessions(tmp_path):
    history = [
        {"frame": 0, "raw": [0], "states": [0], "inference_ms": 10},
        {"frame": 2, "raw": [1], "states": [1], "inference_ms": 20},
        {"frame": 4, "raw": [0], "states": [0], "inference_ms": 10},
    ]
    report = summarize(history, 1, 2, {}, display_fps=2)
    assert report["flips_raw"] == 2
    assert report["flips_smoothed"] == 2
    assert report["per_slot_occupied_seconds"] == [1.0]
    assert report["per_slot_longest_stay_s"] == [1.0]
