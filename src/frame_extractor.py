"""Extract a processing-size reference frame and dimensions for one video."""

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "parking1" / "parking1.mp4"
OUTPUT_PATH = ROOT / "data" / "parking1" / "img_0.jpg"
PROCESSING_WIDTH = 1080


def processing_dimensions(source_width: int, source_height: int) -> tuple[int, int]:
    """Scale a source to width 1080 and round its aspect-ratio height up to even pixels."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    height = round(PROCESSING_WIDTH * source_height / source_width)
    height += height % 2
    return PROCESSING_WIDTH, height


def extract_frame(video: str | Path, name: str, frame_index: int = 0) -> Path:
    """Extract the selected video frame and persist its processing dimensions."""
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Unable to open video: {video}")
    try:
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width, height = processing_dimensions(source_width, source_height)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Unable to read frame {frame_index} from {video}")
        resized = cv2.resize(frame, (width, height))
    finally:
        capture.release()
    data_dir = ROOT / "data" / name
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "img_0.jpg"
    if not cv2.imwrite(str(output), resized):
        raise OSError(f"Unable to write reference frame: {output}")
    meta = {"width": width, "height": height}
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {output} at {width}x{height}; metadata: {data_dir / 'meta.json'}")
    return output


def main(argv=None):
    """Parse the reference-frame extraction command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--name")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args(argv)
    name = args.name or (Path(args.video).stem if args.video != str(VIDEO_PATH) else "parking1")
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in name):
        raise ValueError("--name may contain only letters, digits, hyphens, and underscores")
    extract_frame(args.video, name, args.frame)


if __name__ == "__main__":
    main()
