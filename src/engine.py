"""YOLO-backed detection and parking-slot decision engine."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import cv2
import numpy as np

try:
    from .config import DEFAULTS
    from .geometry import assign_vehicles, centroid_assign
    from .smoothing import SlotStateMachine
except ImportError:
    from config import DEFAULTS
    from geometry import assign_vehicles, centroid_assign
    from smoothing import SlotStateMachine


@dataclass
class Detections:
    boxes: np.ndarray
    cls: np.ndarray
    conf: np.ndarray
    inference_ms: float
    names: dict[int, str]


@dataclass
class Decision:
    raw: list[int]
    states: list[int]
    assigned: list[int]
    straddlers: list[dict]
    counts: dict[str, int]


class ParkingEngine:
    """Detect vehicles, assign their footprints to slots, and draw decisions."""

    def __init__(
        self,
        zones,
        model_path="yolo11s.pt",
        classes=DEFAULTS["classes"],
        conf=DEFAULTS["conf"],
        imgsz=DEFAULTS["imgsz"],
        device=None,
        mode=DEFAULTS["mode"],
        smoothing=DEFAULTS["smoothing"],
        footprint_frac=DEFAULTS["footprint_frac"],
        overlap_threshold=DEFAULTS["overlap_threshold"],
        straddle_threshold=DEFAULTS["straddle_threshold"],
        alpha=DEFAULTS["alpha"],
        on=DEFAULTS["on"],
        off=DEFAULTS["off"],
        roi=DEFAULTS["roi"],
        roi_margin=DEFAULTS["roi_margin"],
        detector: Callable | None = None,
        per_class_conf: dict[int, float] | None = None,
        size_filter: bool = DEFAULTS["size_filter"],
        size_ratio_min: float = DEFAULTS["size_ratio_min"],
        size_ratio_max: float = DEFAULTS["size_ratio_max"],
        aspect_ratio_min: float = DEFAULTS["aspect_ratio_min"],
        aspect_ratio_max: float = DEFAULTS["aspect_ratio_max"],
        ignore_areas: list | None = None,
        disabled_slots: list[int] | None = None,
        show_only_touching: bool = DEFAULTS["show_only_touching"],
    ):
        if mode not in {"footprint", "centroid"}:
            raise ValueError("mode must be 'footprint' or 'centroid'")
        self.zones = [np.asarray(zone["points"] if isinstance(zone, dict) else zone, dtype=np.float32) for zone in zones]
        self.classes = tuple(classes) if classes is not None else None
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.mode = mode
        self.smoothing = smoothing
        self.footprint_frac = footprint_frac
        self.overlap_threshold = overlap_threshold
        self.straddle_threshold = straddle_threshold
        self.roi = roi
        self.roi_margin = roi_margin
        self.detector = detector
        self.size_filter = bool(size_filter)
        self.size_ratio_min = float(size_ratio_min)
        self.size_ratio_max = float(size_ratio_max)
        self.aspect_ratio_min = float(aspect_ratio_min)
        self.aspect_ratio_max = float(aspect_ratio_max)
        self.show_only_touching = bool(show_only_touching)
        self.disabled_slots = set(int(slot) for slot in (disabled_slots or []))
        default_per_class_conf = {int(k): float(v) for k, v in DEFAULTS["per_class_conf"].items()}
        self.per_class_conf = default_per_class_conf if per_class_conf is None else {int(k): float(v) for k, v in per_class_conf.items()}
        self.ignore_areas = [
            np.asarray(area, dtype=np.float32).reshape(-1, 2)
            for area in (ignore_areas or [])
        ]
        self.model = None
        self.names: dict[int, str] = {}
        if detector is None:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
            self.names = self.model.names
            self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8), imgsz=self.imgsz, device=self.device, verbose=False)
        self.state_machine = SlotStateMachine(len(self.zones), alpha, on, off)
        self.states = [0] * len(self.zones)
        self.last_detections = Detections(np.empty((0, 4)), np.empty(0), np.empty(0), 0.0, self.names)
        self.last_decision = self.decide(self.last_detections, update=False)

    def _class_confidence_threshold(self, class_id: int) -> float:
        """Return the per-class confidence floor for a detected object class."""
        return float(self.per_class_conf.get(int(class_id), self.conf))

    def _point_in_ignore_area(self, point: tuple[float, float]) -> bool:
        """Check whether a box centre falls within any configured ignore polygon."""
        if not self.ignore_areas:
            return False
        point = np.asarray(point, dtype=np.float32)
        for polygon in self.ignore_areas:
            if cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), tuple(point), False) >= 0:
                return True
        return False

    def _size_is_valid(self, box: np.ndarray) -> bool:
        """Reject obviously wrong detections using slot-relative footprint limits."""
        if not self.size_filter or not self.zones:
            return True
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if width <= 0 or height <= 0:
            return False
        median_area = float(np.median([cv2.contourArea(np.asarray(zone, dtype=np.float32).reshape(-1, 1, 2)) for zone in self.zones]))
        area = width * height
        if median_area <= 0:
            return True
        if area < self.size_ratio_min * median_area or area > self.size_ratio_max * median_area:
            return False
        aspect = width / height if height > 0 else 1.0
        if aspect < self.aspect_ratio_min or aspect > self.aspect_ratio_max:
            return False
        return True

    def _roi_bounds(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        points = np.concatenate(self.zones, axis=0)
        margin = self.roi_margin
        left = max(0, int(np.floor(points[:, 0].min())) - margin)
        top = max(0, int(np.floor(points[:, 1].min())) - margin)
        right = min(width, int(np.ceil(points[:, 0].max())) + margin)
        bottom = min(height, int(np.ceil(points[:, 1].max())) + margin)
        return left, top, max(left + 1, right), max(top + 1, bottom)

    def detect(self, frame: np.ndarray) -> Detections:
        """Run inference and return detections translated to full-frame pixels."""
        left = top = 0
        image = frame
        if self.roi and self.zones:
            left, top, right, bottom = self._roi_bounds(frame)
            image = frame[top:bottom, left:right]
        start = perf_counter()
        if self.detector is not None:
            result = self.detector(image.copy())
            elapsed = (perf_counter() - start) * 1000
            if isinstance(result, Detections):
                boxes, classes, confidences = result.boxes, result.cls, result.conf
                names = result.names
                elapsed = result.inference_ms or elapsed
            else:
                boxes, classes, confidences = result[:3]
                names = self.names
        else:
            kwargs = {"imgsz": self.imgsz, "conf": self.conf, "verbose": False}
            if self.classes is not None:
                kwargs["classes"] = list(self.classes)
            if self.device is not None:
                kwargs["device"] = self.device
            result = self.model.predict(image, **kwargs)[0]
            elapsed = (perf_counter() - start) * 1000
            names = self.model.names
            if result.boxes is None:
                boxes, classes, confidences = np.empty((0, 4)), np.empty(0), np.empty(0)
            else:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        classes = np.asarray(classes, dtype=np.float32).reshape(-1)
        confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
        if len(boxes) != len(classes) or len(boxes) != len(confidences):
            raise ValueError("detector must return equal counts of boxes, classes, and confidences")
        if left or top:
            boxes[:, (0, 2)] += left
            boxes[:, (1, 3)] += top
        keep = np.ones(len(boxes), dtype=bool)
        if self.classes is not None:
            keep &= np.isin(classes.astype(int), self.classes)
        for index, (class_id, confidence) in enumerate(zip(classes, confidences)):
            if confidence < self._class_confidence_threshold(int(class_id)):
                keep[index] = False
        if self.size_filter:
            keep &= np.array([self._size_is_valid(box) for box in boxes], dtype=bool)
        if self.ignore_areas:
            keep &= np.array([
                not self._point_in_ignore_area(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
                for box in boxes
            ], dtype=bool)
        boxes = boxes[keep]
        classes = classes[keep]
        confidences = confidences[keep]
        return Detections(boxes, classes, confidences, elapsed, names)

    def decide(self, dets: Detections, update: bool = True) -> Decision:
        """Convert vehicle boxes to raw and smoothed slot states and counts."""
        if self.classes is None:
            keep = np.ones(len(dets.boxes), dtype=bool)
        else:
            keep = np.isin(dets.cls.astype(int), self.classes)
        boxes = dets.boxes[keep]
        if self.mode == "centroid":
            raw, assigned, straddlers = centroid_assign(boxes, self.zones)
        else:
            raw, assigned, straddlers = assign_vehicles(
                boxes, self.zones, self.footprint_frac, self.overlap_threshold, self.straddle_threshold
            )
        if self.disabled_slots:
            raw = list(raw)
            for slot_index in sorted(self.disabled_slots):
                if slot_index < len(raw):
                    raw[slot_index] = 0
        states = self.state_machine.update(raw) if self.smoothing and update else (
            self.state_machine.states.copy() if self.smoothing else raw.copy()
        )
        if self.disabled_slots:
            states = list(states)
            for slot_index in sorted(self.disabled_slots):
                if slot_index < len(states):
                    states[slot_index] = 0
        if self.smoothing and update:
            self.states = states.copy()
        elif not self.smoothing:
            self.states = states.copy()
        enabled_slots = [slot for slot in range(len(states)) if slot not in self.disabled_slots]
        occupied = sum(states[slot] for slot in enabled_slots)
        return Decision(raw, states, assigned, straddlers, {
            "occupied": occupied,
            "available": len(enabled_slots) - occupied,
        })

    def step(self, frame: np.ndarray) -> tuple[Detections, Decision]:
        """Run detection then update the parking decision once."""
        dets = self.detect(frame)
        decision = self.decide(dets)
        self.last_detections, self.last_decision = dets, decision
        return dets, decision

    def draw(self, frame: np.ndarray, dets: Detections, decision: Decision, hud_extra: dict | None = None) -> np.ndarray:
        """Draw slot states, detected vehicles, and a compact occupancy HUD."""
        output = frame.copy()
        for index, zone in enumerate(self.zones):
            points = np.round(zone).astype(np.int32).reshape(-1, 1, 2)
            color = (0, 0, 255) if decision.states[index] else (0, 210, 0)
            cv2.polylines(output, [points], True, color, 2)
        assigned = {index for index, slot in enumerate(decision.assigned) if slot >= 0}
        straddlers = {int(item["vehicle_index"]) for item in decision.straddlers}
        for index, box in enumerate(dets.boxes):
            if self.show_only_touching and index not in assigned and index not in straddlers:
                continue
            x1, y1, x2, y2 = np.round(box).astype(int)
            straddling = any(item["vehicle_index"] == index for item in decision.straddlers)
            color = (0, 165, 255) if straddling else (255, 255, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            class_id = int(dets.cls[index])
            label = dets.names.get(class_id, str(class_id))
            cv2.putText(output, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        height, width = output.shape[:2]
        cv2.rectangle(output, (8, 8), (min(width - 8, 300), 55), (20, 20, 20), -1)
        cv2.putText(output, f"Occupancy: {decision.counts['occupied']} / Available: {decision.counts['available']}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        if decision.counts["available"] == 0:
            cv2.putText(output, "LOT FULL", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
        if hud_extra:
            y = 72
            for key, value in hud_extra.items():
                cv2.putText(output, f"{key}: {value}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y += 16
        return output
