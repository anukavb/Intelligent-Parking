"""Manually label parking-slot occupancy in sampled video frames."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.zones import load_zones


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="parking1")
    parser.add_argument("--video")
    parser.add_argument("--labels")
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args(argv)
    video = Path(args.video) if args.video else ROOT / "data" / args.name / f"{args.name}.mp4"
    labels_path = Path(args.labels) if args.labels else ROOT / "outputs" / f"{args.name}_ground_truth.csv"
    zones = load_zones(ROOT / "data" / args.name / "bounding_boxes.json")
    meta_path = ROOT / "data" / args.name / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        processing_size = (int(meta["width"]), int(meta["height"]))
    else:
        processing_size = (1080, 600)
    labels = {}
    if labels_path.exists():
        with labels_path.open(newline="", encoding="utf-8") as stream:
            labels = {(int(row["frame"]), int(row["slot"])): int(row["occupied"]) for row in csv.DictReader(stream)}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"Unable to open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(args.start, max(0, frame_count - 1))
    selected_slot = 0

    def save():
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        with labels_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "slot", "occupied"])
            for (frame, slot), occupied in sorted(labels.items()):
                writer.writerow([frame, slot, occupied])

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, processing_size)
        for slot, zone in enumerate(zones):
            polygon = np.asarray(zone["points"], dtype=np.int32)
            color = (0, 255, 0) if slot != selected_slot else (255, 255, 0)
            cv2.polylines(frame, [polygon], True, color, 2)
            center = tuple(np.mean(polygon, axis=0).astype(int))
            occupied = labels.get((frame_index, slot))
            text = f"{slot}: {'?' if occupied is None else occupied}"
            cv2.putText(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(
            frame,
            f"frame {frame_index} | slot {selected_slot} | 0 free, 1 occupied, n/p frame, arrows slot, s save, q quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Parking ground truth labels", frame)
        key = cv2.waitKeyEx(0)
        if key == ord("q"):
            break
        if key == ord("s"):
            save()
        elif key in (ord("0"), ord("1")):
            labels[(frame_index, selected_slot)] = int(chr(key))
        elif key == ord("n"):
            frame_index = min(max(0, frame_count - 1), frame_index + args.step)
        elif key == ord("p"):
            frame_index = max(0, frame_index - args.step)
        elif key in (81, 2424832, 65361, ord("[")):
            selected_slot = max(0, selected_slot - 1)
        elif key in (83, 2555904, 65363, ord("]")):
            selected_slot = min(len(zones) - 1, selected_slot + 1)
    save()
    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labels to {labels_path}")


if __name__ == "__main__":
    main()
