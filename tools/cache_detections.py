"""Cache one complete YOLO detection pass for later parameter sweeps."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zones import load_zones
from tools.common import cache_path


def _roi_bounds(zones, frame_shape, margin=40):
    """Return a clipped rectangle around every slot polygon plus an ROI margin."""
    height, width = frame_shape[:2]
    points = np.concatenate([zone["points"] for zone in zones], axis=0)
    left = max(0, int(np.floor(points[:, 0].min())) - margin)
    top = max(0, int(np.floor(points[:, 1].min())) - margin)
    right = min(width, int(np.ceil(points[:, 0].max())) + margin)
    bottom = min(height, int(np.ceil(points[:, 1].max())) + margin)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def main(argv=None):
    """Detect all vehicle classes at confidence 0.25 and write a compressed NPZ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="parking1")
    parser.add_argument("--video")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--max-frames", type=int, help="limit frames for a quick cache smoke test")
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    video = Path(args.video) if args.video else ROOT / "data" / args.name / f"{args.name}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    zones_path = ROOT / "data" / args.name / "bounding_boxes.json"
    zones = load_zones(zones_path)
    data_dir = ROOT / "data" / args.name
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"width": 1080, "height": 600}
    width, height = int(meta["width"]), int(meta["height"])

    from ultralytics import YOLO

    model = YOLO(str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model)
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, conf=0.25, classes=[2, 3, 5, 7], verbose=False)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Unable to open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    boxes, classes, confidences = [], [], []
    offsets = [0]
    frame_count = 0
    while args.max_frames is None or frame_count < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height))
        left = top = 0
        image = frame
        if args.roi and zones:
            left, top, right, bottom = _roi_bounds(zones, frame.shape)
            image = frame[top:bottom, left:right]
        result = model.predict(
            image, imgsz=args.imgsz, conf=0.25, classes=[2, 3, 5, 7], verbose=False
        )[0]
        if result.boxes is not None:
            frame_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            frame_boxes[:, (0, 2)] += left
            frame_boxes[:, (1, 3)] += top
            frame_classes = result.boxes.cls.cpu().numpy().astype(np.int32)
            frame_confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
            boxes.extend(frame_boxes)
            classes.extend(frame_classes)
            confidences.extend(frame_confidences)
        frame_count += 1
        offsets.append(len(boxes))
    capture.release()
    metadata = {
        "name": args.name,
        "video": str(video),
        "model": Path(args.model).stem,
        "imgsz": args.imgsz,
        "roi": args.roi,
        "conf": 0.25,
        "classes": [2, 3, 5, 7],
        "fps": fps,
        "frames": frame_count,
    }
    output = cache_path(args.name, args.model, args.imgsz, args.roi)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        classes=np.asarray(classes, dtype=np.int32),
        confidences=np.asarray(confidences, dtype=np.float32),
        offsets=np.asarray(offsets, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"Cached {frame_count} frames and {len(boxes)} detections to {output}")


if __name__ == "__main__":
    main()
