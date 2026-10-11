# =============================================================================
# zone_selector.py
# -----------------------------------------------------------------------------
# Purpose : Launches an interactive Tkinter GUI that lets you draw parking
#           zone polygons on a reference image.  Each zone is defined by
#           exactly 4 mouse-click points.  The coordinates are saved to
#           bounding_boxes.json, which is consumed by detector.py at runtime.
#
# Run AFTER frame_extractor.py has produced a reference image.
# Usage   : python zone_selector.py
#
# Steps inside the GUI
#   1. Click "Upload Image" → open the reference frame (e.g. img_0.jpg)
#   2. Click 4 corners of a parking slot → a blue box appears
#   3. Repeat for every slot in the lot
#   4. Click "Remove Last BBox" to undo the most recent box if needed
#   5. Click "Save" → writes bounding_boxes.json
# =============================================================================

from parking_core import ParkingPtsSelection

# Launch the GUI.  Execution blocks here until the window is closed.
ParkingPtsSelection()
