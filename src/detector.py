# =============================================================================
# detector.py
# -----------------------------------------------------------------------------
# Purpose : Main entry point for real-time parking occupancy detection.
#           Reads a parking lot video frame-by-frame, runs YOLO inference via
#           ParkingManagement, and writes an annotated output video.
#
# Run AFTER zone_selector.py has produced bounding_boxes.json.
# Usage   : python detector.py
#
# Output  : output.mp4  (annotated video saved in the same directory)
# =============================================================================

import cv2
from parking_core import ParkingManagement

# ── Configuration ─────────────────────────────────────────────────────────────

# Input video: pre-recorded CCTV footage of the parking lot
VIDEO_INPUT  = r"parking1.mp4"

# Output video: annotated version with zone overlays and occupancy stats
VIDEO_OUTPUT = r"output.mp4"

# YOLO model weights (downloaded automatically by Ultralytics if not present)
MODEL_WEIGHTS = "yolo11s.pt"

# COCO class IDs to detect — 2 = car
DETECT_CLASSES = [2]

# Zone definitions produced by zone_selector.py
ZONES_JSON = r"bounding_boxes.json"

# Standard frame size (must match what zone_selector.py used when you drew the boxes)
FRAME_WIDTH  = 1080
FRAME_HEIGHT = 600

# ── Video I/O setup ───────────────────────────────────────────────────────────

cap = cv2.VideoCapture(VIDEO_INPUT)

# Preserve the original video frame rate in the output
fps = cap.get(cv2.CAP_PROP_FPS)

# VideoWriter encodes the annotated frames into an MP4 file
out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

# ── Detector initialisation ───────────────────────────────────────────────────

parking_manager = ParkingManagement(
    model=MODEL_WEIGHTS,
    classes=DETECT_CLASSES,
    json_file=ZONES_JSON,
)

# ── Frame processing loop ─────────────────────────────────────────────────────

print("Processing video — press ESC in the preview window to stop early.")

while cap.isOpened():
    ret, im0 = cap.read()

    # End of video or read error
    if not ret:
        break

    # Resize frame to the standard resolution expected by the zone coordinates
    im0 = cv2.resize(im0, (FRAME_WIDTH, FRAME_HEIGHT))

    # Run YOLO detection + zone occupancy analysis; returns annotated frame
    im0 = parking_manager.process_data(im0)

    # Save annotated frame to output video
    out.write(im0)

    # Live preview window (close with ESC)
    cv2.imshow("Parking Occupancy Detection", im0)
    if cv2.waitKey(1) & 0xFF == 27:
        print("Stopped early by user.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────

cap.release()
out.release()           # finalise and flush the output MP4
cv2.destroyAllWindows()

print(f"Done. Annotated video saved to: {VIDEO_OUTPUT}")
