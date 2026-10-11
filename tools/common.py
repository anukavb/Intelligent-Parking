"""Shared cached-detection and ground-truth evaluation helpers."""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cache_path(name: str, model: str = "yolo11s", imgsz: int = 640, roi: bool = False) -> Path:
    """Build the stable cache filename from its video and inference inputs."""
    model_name = Path(model).stem
    return ROOT / "outputs" / "cache" / f"{name}__{model_name}__{imgsz}__roi-{int(roi)}.npz"


def load_cache(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a non-pickled cache of flat detections and per-frame offsets."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        boxes = data["boxes"].astype(np.float32, copy=False)
        classes = data["classes"].astype(np.int32, copy=False)
        confidences = data["confidences"].astype(np.float32, copy=False)
        offsets = data["offsets"].astype(np.int64, copy=False)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Detection cache boxes must have shape (detections, 4)")
    if len(classes) != len(boxes) or len(confidences) != len(boxes):
        raise ValueError("Detection cache arrays have inconsistent detection counts")
    if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0:
        raise ValueError("Detection cache offsets must be a one-dimensional array starting at zero")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(boxes):
        raise ValueError("Detection cache offsets must be monotonic and end at the detection count")
    return metadata, boxes, classes, confidences, offsets


def read_ground_truth(path: str | Path) -> dict[tuple[int, int], int]:
    """Read CSV labels whose occupied column is 0 or 1 for a frame and slot."""
    labels = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame, slot, occupied = int(row["frame"]), int(row["slot"]), int(row["occupied"])
            if frame < 0 or slot < 0 or occupied not in (0, 1):
                raise ValueError(f"Invalid ground-truth row: {row}")
            labels[(frame, slot)] = occupied
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def score_states(predictions: list[list[int]], labels: dict[tuple[int, int], int]) -> dict:
    """Compute occupied-positive precision, recall, F1, accuracy, and confusion counts."""
    tp = fp = fn = tn = 0
    for (frame, slot), actual in labels.items():
        if frame >= len(predictions) or slot >= len(predictions[frame]):
            raise ValueError(f"Ground-truth coordinate is outside cached video: {(frame, slot)}")
        predicted = int(predictions[frame][slot])
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "label_count": len(labels),
    }


def evaluate_cache(name: str, config: dict, cache_file: str | Path, ground_truth_file: str | Path) -> dict:
    """Replay cached boxes through slot logic and score smoothed states at labels."""
    try:
        from src.config import DEFAULTS
        from src.engine import Detections, ParkingEngine
        from src.zones import load_zones
    except ImportError:
        from config import DEFAULTS
        from engine import Detections, ParkingEngine
        from zones import load_zones

    metadata, boxes, classes, confidences, offsets = load_cache(cache_file)
    cached_conf = float(metadata["conf"])
    configured_conf = float(config.get("conf", DEFAULTS["conf"]))
    if configured_conf < cached_conf:
        raise ValueError(
            f"Requested conf={configured_conf} is below the cache threshold "
            f"{cached_conf}; rebuild the detection cache at a lower threshold"
        )
    if len(offsets) != int(metadata["frames"]) + 1 or offsets[-1] != len(boxes):
        raise ValueError("Detection cache offsets do not match its frame/detection counts")
    for key in ("imgsz", "roi"):
        if key in config and config[key] != metadata[key]:
            raise ValueError(
                f"Cannot change {key} during cached evaluation; rebuild detections with {key}={config[key]}"
            )
    zones_path = ROOT / "data" / name / "bounding_boxes.json"
    zones = load_zones(zones_path)
    params = dict(DEFAULTS)
    params["imgsz"] = metadata["imgsz"]
    params["roi"] = metadata["roi"]
    params.update(config)
    params["classes"] = tuple(params.get("classes", DEFAULTS["classes"]))
    engine = ParkingEngine(
        zones,
        classes=params["classes"],
        conf=params.get("conf", DEFAULTS["conf"]),
        imgsz=params.get("imgsz", DEFAULTS["imgsz"]),
        mode=params.get("mode", "footprint"),
        smoothing=params.get("smoothing", True),
        footprint_frac=params.get("footprint_frac", 0.4),
        overlap_threshold=params.get("overlap_threshold", 0.3),
        straddle_threshold=params.get("straddle_threshold", 0.25),
        alpha=params.get("alpha", 0.3),
        on=params.get("on", 0.65),
        off=params.get("off", 0.35),
        roi=params.get("roi", False),
        roi_margin=params.get("roi_margin", 40),
        detector=lambda _frame: ([], [], []),
    )
    predictions, raw_predictions = [], []
    for frame_index in range(len(offsets) - 1):
        start, end = offsets[frame_index:frame_index + 2]
        frame_conf = confidences[start:end]
        keep = frame_conf >= float(params.get("conf", 0.35))
        dets = Detections(
            boxes[start:end][keep],
            classes[start:end][keep],
            frame_conf[keep],
            0.0,
            {},
        )
        decision = engine.decide(dets)
        raw_predictions.append(decision.raw)
        predictions.append(decision.states)
    labels = read_ground_truth(ground_truth_file)
    metrics = score_states(predictions, labels)
    previous_raw = [0] * len(zones)
    previous_smoothed = [0] * len(zones)
    flips_raw = flips_smoothed = 0
    for raw, smoothed in zip(raw_predictions, predictions):
        flips_raw += sum(a != b for a, b in zip(previous_raw, raw))
        flips_smoothed += sum(a != b for a, b in zip(previous_smoothed, smoothed))
        previous_raw, previous_smoothed = raw, smoothed
    metrics.update(
        name=name,
        frames=metadata["frames"],
        flips_raw=flips_raw,
        flips_smoothed=flips_smoothed,
        configuration=params,
        states=predictions,
        raw_states=raw_predictions,
    )
    return metrics
