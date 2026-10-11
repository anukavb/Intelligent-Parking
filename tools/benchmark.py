"""Measure YOLO model inference throughput on a sample video."""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "data" / "parking1" / "parking1.mp4"))
    parser.add_argument("--model", default=str(ROOT / "yolo11s.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.warmup < 0:
        raise ValueError("--frames must be positive and --warmup cannot be negative")
    video = Path(args.video)
    model_path = Path(args.model)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"Unable to open video: {video}")
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if args.device:
        model.to(args.device)
    latencies = []
    processed = 0
    while processed < args.warmup + args.frames:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            capture.release()
            raise OSError(f"Unable to read frames from {video}")
        started = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, verbose=False)
        elapsed = (time.perf_counter() - started) * 1000
        if processed >= args.warmup:
            latencies.append(elapsed)
        processed += 1
    capture.release()
    measured_seconds = sum(latencies) / 1000
    report = {
        "video": str(video),
        "model": str(model_path),
        "imgsz": args.imgsz,
        "device": str(next(model.model.parameters()).device),
        "frames": len(latencies),
        "mean_inference_ms": statistics.mean(latencies),
        "p95_inference_ms": float(np.percentile(latencies, 95)),
        "inference_hz": len(latencies) / measured_seconds,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
