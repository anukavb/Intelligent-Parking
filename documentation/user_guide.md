# User guide

## Setup

From the repository root, create and activate a virtual environment, then install the pinned packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

The first detector run may download `yolo11s.pt` into the repository root. Python 3.9 or later is required; Windows, macOS, and Linux are supported where the Python dependencies are available.

## Run the project

Run these commands from the repository root:

```powershell
py src/frame_extractor.py
py src/zone_selector.py
py src/detector.py
py web/extract_occupancy.py
py web/build.py
```

The first two steps are needed only when configuring a **new video/camera**. The included parking1 sample already has its reference frame and zone definitions. When selecting zones, open `data/parking1/img_0.jpg`, draw four points for each slot, and save the polygons to `data/parking1/bounding_boxes.json`.

The detector displays a live preview; press Escape to stop early. It writes the annotated video to `outputs/output.mp4`. Occupancy extraction reads that video's red and green zone colours and writes `data/parking1/occupancy.json`. The build step regenerates the self-contained `docs/index.html` page.

The occupancy extractor also accepts an optional data folder and video path:

```powershell
py web/extract_occupancy.py data/parking1 outputs/output.mp4
```

The route-finder page is specific to parking1's road layout. For another lot, update the `DRIVABLE` polygon in `web/routing.js` before rebuilding the page.
