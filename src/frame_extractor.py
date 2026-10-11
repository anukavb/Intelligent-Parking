# =============================================================================
# frame_extractor.py
# -----------------------------------------------------------------------------
# Purpose : Extracts a sample frame from the input parking lot video.
#           Run this FIRST to get a reference image before defining parking
#           zones in zone_selector.py.
#
# Output  : img_0.jpg  (saved in the same directory as the script)
# Usage   : python frame_extractor.py
# =============================================================================

import cv2
import time
import os

# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the input parking lot video
VIDEO_PATH = r"parking1.mp4"

# Path template for saving extracted frames  (%d is replaced by frame index)
OUTPUT_PATH = r"img_%d.jpg"

# Path template for the website's copy of the frame
SITE_OUTPUT_PATH = r"docs/source/data/img_%d.jpg"

# How many frames to extract (1 = just a single reference image)
MAX_FRAMES = 1

# Only process every Nth frame (3 = skip 2, process 1) to avoid near-duplicates
FRAME_SKIP = 3

# ── Main extraction loop ───────────────────────────────────────────────────────

cpt = 0        # counts how many frames have been saved
count = 0      # counts total frames read from the video

cap = cv2.VideoCapture(VIDEO_PATH)

while cpt < MAX_FRAMES:
    ret, frame = cap.read()

    # Stop if the video ends before we reach MAX_FRAMES
    if not ret:
        break

    count += 1

    # Skip frames that aren't multiples of FRAME_SKIP
    if count % FRAME_SKIP != 0:
        continue

    # Resize to a standard display resolution
    frame = cv2.resize(frame, (1080, 600))

    # Preview the frame on screen
    cv2.imshow("Extracted Frame", frame)

    # Save the frame as a JPEG image (and a copy for the website)
    cv2.imwrite(OUTPUT_PATH % cpt, frame)
    os.makedirs(os.path.dirname(SITE_OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(SITE_OUTPUT_PATH % cpt, frame)

    time.sleep(0.01)  # small pause so the window can render
    cpt += 1

    # Press ESC to abort early
    if cv2.waitKey(5) & 0xFF == 27:
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print(f"Done. {cpt} frame(s) saved.")
