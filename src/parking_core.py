# =============================================================================
# parking_core.py
# -----------------------------------------------------------------------------
# Purpose : Core module containing two classes:
#
#   ParkingPtsSelection  – Tkinter GUI for manually defining parking zone
#                          polygons on a reference image and saving them to
#                          bounding_boxes.json.
#
#   ParkingManagement    – Real-time inference engine that loads the saved
#                          zones, runs YOLO detection on each video frame,
#                          checks which zones are occupied via point-in-polygon
#                          logic, and annotates the frame with colour-coded
#                          overlays and occupancy statistics.
#
# Dependencies : opencv-python, numpy, ultralytics
# =============================================================================

import json

import cv2
import numpy as np
import os

from ultralytics.solutions.solutions import LOGGER, BaseSolution
from ultralytics.utils.plotting import Annotator


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1 : ParkingPtsSelection
# ─────────────────────────────────────────────────────────────────────────────

class ParkingPtsSelection:
    """
    Interactive Tkinter tool for selecting parking zone polygons on an image.

    Workflow
    --------
    1. Upload a reference frame from the parking lot video.
    2. Click 4 points to outline each parking slot.
    3. Save → coordinates are written to bounding_boxes.json (scaled back to
       original image resolution so they remain accurate regardless of canvas
       display size).
    """

    def __init__(self):
        """Bootstrap: verify tkinter is available, build the UI, start the event loop."""
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
        except ImportError:
            raise ImportError(
                "Tkinter is not available in your Python installation. "
                "On Windows, re-run the Python installer and check 'tcl/tk and IDLE'."
            )

        # Store references to tkinter submodules for use in other methods
        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox

        self.setup_ui()
        self.initialize_properties()
        self.master.mainloop()  # blocks until the window is closed

    # ── UI construction ───────────────────────────────────────────────────────

    def setup_ui(self):
        """Create the main window, canvas, and control buttons."""
        self.master = self.tk.Tk()
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)

        # Canvas where the image and drawn boxes are displayed
        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        # Button row at the top
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

    def initialize_properties(self):
        """Set default values for image, canvas, and bounding-box state."""
        self.image = self.canvas_image = None   # PIL Image and its Tk-compatible version
        self.rg_data = []                        # list of completed 4-point boxes
        self.current_box = []                    # points collected for the box in progress
        self.imgw = self.imgh = 0                # original image dimensions (for coordinate scaling)
        self.canvas_max_width = 1020             # canvas display limits
        self.canvas_max_height = 500

    # ── Image handling ────────────────────────────────────────────────────────

    def upload_image(self):
        """Open a file dialog, load the chosen image, resize to fit the canvas, and display it."""
        from PIL import Image, ImageTk  # imported here because ImageTk requires an active Tk root

        self.image = Image.open(
            self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        )
        if not self.image:
            return

        self.imgw, self.imgh = self.image.size

        # Compute a canvas size that respects both the image aspect ratio and max dimensions
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw)
            if aspect_ratio > 1
            else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh)
            if aspect_ratio <= 1
            else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(
            self.image.resize((canvas_width, canvas_height), Image.LANCZOS)
        )
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

        # Bind left-click to point collection
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Reset any previously drawn boxes when a new image is loaded
        self.rg_data.clear()
        self.current_box.clear()

    # ── Point and box interaction ─────────────────────────────────────────────

    def on_canvas_click(self, event):
        """
        Record a clicked point.  Once 4 points are collected, commit the box
        to rg_data and draw it on the canvas.
        """
        self.current_box.append((event.x, event.y))

        # Visual feedback: small red dot at each click
        self.canvas.create_oval(
            event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red"
        )

        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box):
        """Draw a closed quadrilateral (blue lines) connecting the 4 points of a box."""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Delete the most recently added box and refresh the canvas."""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self):
        """Clear and redraw the canvas image plus all currently stored boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_json(self):
        """
        Scale canvas-space coordinates back to original image resolution and
        write all boxes to bounding_boxes.json.

        Scaling is necessary because the image was downsized to fit the canvas;
        the JSON must store true pixel coordinates so they align with the full-
        resolution video frames processed during detection.
        """
        scale_w = self.imgw / self.canvas.winfo_width()
        scale_h = self.imgh / self.canvas.winfo_height()

        data = [
            {"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]}
            for box in self.rg_data
        ]

        with open("bounding_boxes.json", "w") as f:
            json.dump(data, f, indent=4)

        os.makedirs("docs/source/data", exist_ok=True)
        with open("docs/source/data/bounding_boxes.json", "w") as f:
            json.dump(data, f, indent=4)

        self.messagebox.showinfo(
            "Success",
            "Bounding boxes saved to bounding_boxes.json\n"
            "and docs/source/data/bounding_boxes.json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2 : ParkingManagement
# ─────────────────────────────────────────────────────────────────────────────

class ParkingManagement(BaseSolution):
    """
    Real-time parking occupancy monitor using YOLO and pre-defined zone polygons.

    For each video frame:
      1. Run YOLO detection (class 2 = car) via the parent BaseSolution.
      2. For every detected car, compute its bounding-box centroid.
      3. Test the centroid against every zone polygon using cv2.pointPolygonTest.
      4. Colour zones green (available) or red (occupied) and overlay stats.
    """

    def __init__(self, **kwargs):
        """
        Load the zone definitions from JSON and set up colour constants.

        Expected kwargs (passed through to BaseSolution + stored in self.CFG):
            model     – path to YOLO weights, e.g. "yolo11s.pt"
            classes   – list of COCO class IDs to detect, e.g. [2] for cars
            json_file – path to bounding_boxes.json produced by zone_selector.py
        """
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]
        if self.json_file is None:
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path cannot be empty")

        # Load zone polygon definitions
        with open(self.json_file) as f:
            self.json = json.load(f)

        # Running counters updated each frame
        self.pr_info = {"Occupancy": 0, "Available": 0}

        # Annotation colours (BGR)
        self.arc = (0, 255, 0)    # available zone border  → green
        self.occ = (0, 0, 255)    # occupied zone border   → red
        self.dc  = (255, 0, 189)  # vehicle centroid dot   → pink/magenta

    # ── Per-frame processing ──────────────────────────────────────────────────

    def process_data(self, im0):
        """
        Run detection and occupancy analysis on a single frame.

        Parameters
        ----------
        im0 : np.ndarray
            BGR frame read from the video (already resized to 1080×600).

        Returns
        -------
        np.ndarray
            Annotated frame with zone overlays and occupancy text.
        """
        # Run YOLO tracking / detection; populates self.boxes and self.clss
        self.extract_tracks(im0)

        # Start assuming all slots are empty
        es = len(self.json)   # empty slot count (decremented as cars are found)
        fs = 0                # filled slot count

        annotator = Annotator(im0, self.line_width)

        for region in self.json:
            # Convert the 4-point polygon to the shape OpenCV expects: (N, 1, 2)
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))

            rg_occupied = False  # flag: has a car been found in this zone?

            for box, cls in zip(self.boxes, self.clss):
                # Centroid of the detected bounding box
                xc = int((box[0] + box[2]) / 2)
                yc = int((box[1] + box[3]) / 2)

                # pointPolygonTest returns >= 0 if the point is inside or on the polygon
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)

                if dist >= 0:
                    # Draw a filled circle at the vehicle centroid
                    cv2.circle(
                        im0, (xc, yc),
                        radius=self.line_width * 4,
                        color=self.dc,
                        thickness=-1
                    )
                    # Label the detected class (e.g. "car") near the centroid
                    cv2.putText(
                        im0,
                        self.model.names[int(cls)],
                        (xc, yc + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    rg_occupied = True
                    break  # one car is enough to mark the zone occupied; move on

            # Update slot counters
            if rg_occupied:
                fs += 1
                es -= 1

            # Draw the zone polygon: red if occupied, green if available
            cv2.polylines(
                im0, [pts_array],
                isClosed=True,
                color=self.occ if rg_occupied else self.arc,
                thickness=2
            )

        # Store updated counts
        self.pr_info["Occupancy"] = fs
        self.pr_info["Available"] = es

        # Overlay occupancy statistics in the top-left corner
        y_offset = 30
        for key, value in self.pr_info.items():
            label = f"{key}: {value}"
            cv2.putText(
                im0, label,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            y_offset += 30

        # Call parent display hook (handles env-specific output, e.g. stream mode)
        self.display_output(im0)

        return im0
