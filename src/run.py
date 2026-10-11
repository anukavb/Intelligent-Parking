"""Prepare a new video's zones and start the live parking monitor."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: str) -> str:
    """Restrict dataset folder names so generated paths cannot escape data/."""
    if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
        raise ValueError("name may contain only letters, digits, hyphens, and underscores")
    return value


def main(argv=None):
    """Extract a reference image if needed, collect zones, then invoke detector CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="input video file")
    parser.add_argument("--name", help="data set name; defaults to the video stem")
    args, detector_args = parser.parse_known_args(argv)
    video = Path(args.video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    name = _safe_name(args.name or video.stem)
    data_dir = ROOT / "data" / name
    frame_path = data_dir / "img_0.jpg"
    zones_path = data_dir / "bounding_boxes.json"
    if not frame_path.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "frame_extractor.py"), "--video", str(video), "--name", name],
            cwd=ROOT,
            check=True,
        )
    if not zones_path.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "zone_selector.py"), "--name", name],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "detector.py"), "--source", str(video), "--name", name, *detector_args],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
