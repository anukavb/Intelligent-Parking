"""Command-line live monitor and deterministic offline detector."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULTS
from engine import ParkingEngine
from live import LiveRunner, run_offline

VIDEO_INPUT = ROOT / "data" / "parking1" / "parking1.mp4"
VIDEO_OUTPUT = ROOT / "outputs" / "parking1" / "output.mp4"
MODEL_WEIGHTS = ROOT / "yolo11s.pt"
DETECT_CLASSES = list(DEFAULTS["classes"])
ZONES_JSON = ROOT / "data" / "parking1" / "bounding_boxes.json"


def _name_from_source(source: str) -> str:
    """Choose a filesystem-safe run name derived from a video path or camera URL."""
    parsed = urlparse(source)
    stem = Path(parsed.path).stem if parsed.scheme else Path(source).stem
    candidate = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem).strip("_")
    return candidate or "camera"


def _parse_source(value: str):
    """Convert an integer source string to a camera index, otherwise keep it as a path/URL."""
    return int(value) if value.isdigit() else value


def _parser():
    """Create the detector options shared by command-line and orchestration callers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(VIDEO_INPUT), help="video path, camera index, or URL")
    parser.add_argument("--name", help="data set/run name; defaults to the source file stem")
    parser.add_argument("--tag", default="", help="optional output suffix for ablation runs")
    parser.add_argument("--mode", choices=("live", "offline"), default="live")
    parser.add_argument("--model", default=str(MODEL_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    parser.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--cars-only", action="store_true")
    parser.add_argument("--centroid", action="store_true")
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="inference stride for offline mode")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--use-tuned", action="store_true")
    return parser


def _print_summary(summary: dict):
    """Print the key run metrics in a compact table."""
    rows = (
        ("Frames", summary.get("frames", 0)),
        ("Duration", f"{summary.get('duration_s', 0):.2f} s"),
        ("Peak occupancy", summary.get("peak_occupancy", 0)),
        ("Mean occupancy", f"{summary.get('mean_occupancy_pct', 0):.1f}%"),
        ("Raw/smoothed flips", f"{summary.get('flips_raw', 0)} / {summary.get('flips_smoothed', 0)}"),
        ("Mean inference", f"{summary.get('mean_inference_ms', 0):.1f} ms"),
        ("Inference/display", f"{summary.get('effective_inference_hz', 0):.2f} / {summary.get('display_fps', 0):.2f} Hz"),
        ("Mean/p95 lag", f"{summary.get('mean_lag_ms', 0):.1f} / {summary.get('p95_lag_ms', 0):.1f} ms"),
        ("Runtime", f"{summary.get('runtime_s', 0):.2f} s"),
    )
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}} | {value}")


def main(argv=None):
    """Run the selected live or sequential detector mode and publish its artifacts."""
    args = _parser().parse_args(argv)
    source = _parse_source(args.source)
    name = args.name or _name_from_source(args.source)
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
        raise ValueError("--name may contain only letters, digits, hyphens, and underscores")
    data_dir = ROOT / "data" / name
    zones_path = data_dir / "bounding_boxes.json"
    if not zones_path.is_file():
        raise FileNotFoundError(f"Parking zones not found: {zones_path}")
    zones = json.loads(zones_path.read_text(encoding="utf-8"))
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        processing_size = (int(meta["width"]), int(meta["height"]))
    else:
        processing_size = (1080, 600)
    classes = [2] if args.cars_only else list(DEFAULTS["classes"])
    mode = "centroid" if args.centroid else DEFAULTS["mode"]
    smoothing = not args.no_smoothing
    tuned = {}
    if args.use_tuned:
        tuned_path = ROOT / "outputs" / "tuning" / "best_params.json"
        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
        args.conf = tuned.get("conf", args.conf)
    engine = ParkingEngine(
        zones,
        model_path=args.model,
        classes=classes,
        conf=args.conf,
        imgsz=args.imgsz,
        mode=tuned.get("mode", mode),
        smoothing=tuned.get("smoothing", smoothing),
        footprint_frac=tuned.get("footprint_frac", DEFAULTS["footprint_frac"]),
        overlap_threshold=tuned.get("overlap_threshold", DEFAULTS["overlap_threshold"]),
        straddle_threshold=tuned.get("straddle_threshold", DEFAULTS["straddle_threshold"]),
        alpha=tuned.get("alpha", DEFAULTS["alpha"]),
        on=tuned.get("on", DEFAULTS["on"]),
        off=tuned.get("off", DEFAULTS["off"]),
        roi=tuned.get("roi", args.roi),
    )
    run_name = name + (f"__{args.tag}" if args.tag else "")
    output_dir = ROOT / "outputs" / run_name
    record_path = output_dir / "output.mp4"
    if args.mode == "offline":
        summary = run_offline(
            engine, source, args.stride, output_dir, record_path,
            processing_size, args.max_frames,
        )
    else:
        runner = LiveRunner(
            engine, source, processing_size, paced=True, record_path=record_path,
            output_dir=output_dir, max_frames=args.max_frames,
        ).start()
        paused = False
        try:
            while not runner.done:
                frame, _meta = runner.latest()
                if frame is not None and not args.no_show and not paused:
                    cv2.imshow("Intelligent Parking", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        runner.stop()
                        break
                    if key == ord(" "):
                        paused = True
                elif frame is not None and not args.no_show and paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(" "):
                        paused = False
                    elif key == 27:
                        runner.stop()
                        break
                else:
                    time.sleep(0.02)
            if not runner.done:
                runner.stop()
            runner.raise_if_failed()
        finally:
            cv2.destroyAllWindows()
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    _print_summary(summary)
    print(f"Output directory: {output_dir}")
    if args.publish:
        occupancy_source = output_dir / "occupancy.json"
        if not occupancy_source.is_file():
            raise FileNotFoundError(f"Run occupancy output missing: {occupancy_source}")
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(occupancy_source, data_dir / "occupancy.json")
        print(f"Published occupancy: {data_dir / 'occupancy.json'}")
    return summary


if __name__ == "__main__":
    main()
