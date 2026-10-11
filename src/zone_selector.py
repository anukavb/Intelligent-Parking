"""Launch the interactive single-slot or row-based zone selector."""

import argparse
from pathlib import Path

from parking_core import ParkingPtsSelection

ROOT = Path(__file__).resolve().parents[1]
ZONES_JSON = ROOT / "data" / "parking1" / "bounding_boxes.json"


def main(argv=None):
    """Open the parking1 file chooser or load the selected dataset's reference image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="data set name; opens data/<name>/img_0.jpg")
    args = parser.parse_args(argv)
    if args.name:
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in args.name):
            raise ValueError("--name may contain only letters, digits, hyphens, and underscores")
        data_dir = ROOT / "data" / args.name
        image_path = data_dir / "img_0.jpg"
        zones_path = data_dir / "bounding_boxes.json"
        if not image_path.is_file():
            raise FileNotFoundError(f"Reference frame not found: {image_path}")
    else:
        image_path = None
        zones_path = ZONES_JSON
    ParkingPtsSelection(zones_json=zones_path, image_path=image_path)


if __name__ == "__main__":
    main()
