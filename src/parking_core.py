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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZONES_JSON = ROOT / "data" / "parking1" / "bounding_boxes.json"
try:
    from .zones import split_row
except ImportError:
    from zones import split_row


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

    def __init__(self, zones_json=ZONES_JSON, image_path=None):
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
        self.zones_json = zones_json
        self.initial_image_path = image_path

        self.setup_ui()
        self.initialize_properties()
        if self.initial_image_path:
            self.load_image(self.initial_image_path)
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
            ("Mode: Single slot", self.toggle_mode),
            ("Remove Last", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            button = self.tk.Button(button_frame, text=text, command=cmd)
            button.pack(side=self.tk.LEFT)
            if text.startswith("Mode:"):
                self.mode_button = button

    def initialize_properties(self):
        """Set default values for image, canvas, and bounding-box state."""
        self.image = self.canvas_image = None   # PIL Image and its Tk-compatible version
        self.rg_data = []                        # list of completed 4-point boxes
        self.groups = []                         # each undo group is one slot or one row
        self.current_box = []                    # points collected for the box in progress
        self.row_mode = False
        self.imgw = self.imgh = 0                # original image dimensions (for coordinate scaling)
        self.canvas_max_width = 1020             # canvas display limits
        self.canvas_max_height = 500

    # ── Image handling ────────────────────────────────────────────────────────

    def upload_image(self):
        """Open a file dialog, load the chosen image, resize to fit the canvas, and display it."""
        path = self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.load_image(path)

    def load_image(self, path):
        """Load and fit a reference image while retaining its original pixel dimensions."""
        from PIL import Image, ImageTk  # imported here because ImageTk requires an active Tk root

        self.image = Image.open(path)

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
        self.groups.clear()
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
            points = self.current_box.copy()
            if self.row_mode:
                from tkinter import simpledialog

                count = simpledialog.askinteger(
                    "Slots in row", "Number of slots (1-40):",
                    minvalue=1, maxvalue=40, parent=self.master,
                )
                if count is None:
                    self.current_box.clear()
                    self.redraw_canvas()
                    return
                group = split_row(points, count)
            else:
                group = [{"points": points}]
            self.groups.append(group)
            self.rg_data.extend(group)
            for zone in group:
                self.draw_box(zone["points"])
            self.current_box.clear()

    def toggle_mode(self):
        """Switch between drawing one slot and subdividing a whole row."""
        self.row_mode = not self.row_mode
        self.mode_button.config(text=f"Mode: {'Row' if self.row_mode else 'Single slot'}")

    def draw_box(self, box):
        """Draw a closed quadrilateral (blue lines) connecting the 4 points of a box."""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Delete the most recently added box and refresh the canvas."""
        if not self.groups:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.groups.pop()
        self.rg_data = [zone for group in self.groups for zone in group]
        self.redraw_canvas()

    def redraw_canvas(self):
        """Clear and redraw the canvas image plus all currently stored boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for zone in self.rg_data:
            self.draw_box(zone["points"])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_json(self):
        """
        Scale canvas-space coordinates back to original image resolution and
        write all boxes to data/parking1/bounding_boxes.json.

        Scaling is necessary because the image was downsized to fit the canvas;
        the JSON must store true pixel coordinates so they align with the full-
        resolution video frames processed during detection.
        """
        scale_w = self.imgw / self.canvas.winfo_width()
        scale_h = self.imgh / self.canvas.winfo_height()

        data = [{
            "points": [(int(x * scale_w), int(y * scale_h)) for x, y in zone["points"]]
        } for zone in self.rg_data]

        Path(self.zones_json).parent.mkdir(parents=True, exist_ok=True)
        with open(self.zones_json, "w") as f:
            json.dump(data, f, indent=4)

        self.messagebox.showinfo(
            "Success",
            f"Bounding boxes saved to {self.zones_json}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2 : ParkingManagement
# ─────────────────────────────────────────────────────────────────────────────

class ParkingManagement:
    """Compatibility wrapper that retains the faculty pipeline's public API."""

    def __init__(self, model="yolo11s.pt", classes=(2,), json_file=ZONES_JSON, **kwargs):
        """Load saved zones and configure the shared engine for legacy callers."""
        if json_file is None:
            raise ValueError("json_file path cannot be empty")
        try:
            from .engine import ParkingEngine
            from .zones import load_zones
        except ImportError:
            from engine import ParkingEngine
            from zones import load_zones
        self.json_file = Path(json_file)
        self.json = load_zones(self.json_file)
        self.engine = ParkingEngine(
            self.json, model_path=model, classes=classes, **kwargs
        )
        self.model = self.engine.model
        self.pr_info = {"Occupancy": 0, "Available": len(self.json)}

    def process_data(self, im0):
        """Return the annotated frame while preserving the legacy method name."""
        detections, decision = self.engine.step(im0)
        self.pr_info["Occupancy"] = decision.counts["occupied"]
        self.pr_info["Available"] = decision.counts["available"]
        return self.engine.draw(im0, detections, decision)
