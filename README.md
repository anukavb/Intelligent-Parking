# Intelligent Parking

Intelligent Parking detects vehicles in CCTV footage, marks each parking slot occupied or free, and guides a driver to the nearest free slot.

## Folder layout

```text
intelligent-parking/
├── README.md
├── requirements.txt
├── src/                     Detection pipeline
├── web/                     Route-finder page source and build scripts
├── data/
│   └── parking1/            Sample video, reference frame, zones, occupancy
├── outputs/                 Generated videos (ignored by Git)
├── docs/
│   └── index.html           Generated GitHub Pages page
└── documentation/           Project overview and user guide
```

## Setup

From the repository root, create and activate a virtual environment, then install the pinned dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Ultralytics downloads the YOLO11 model weights when needed. Model weights are ignored by Git.

## Run

Run each command from the repository root:

```powershell
py src/frame_extractor.py
py src/zone_selector.py
py src/detector.py
py web/extract_occupancy.py
py web/build.py
```

Frame extraction and zone selection are needed only for a **new video/camera**. The included `parking1` sample already has its reference frame and zone definitions. When selecting zones, open `data/parking1/img_0.jpg` and save the polygons to `data/parking1/bounding_boxes.json`.

The detector writes its annotated video to `outputs/output.mp4`. Occupancy extraction reads the zone colours in that video and writes `data/parking1/occupancy.json`; the final command rebuilds the self-contained `docs/index.html` page.

## Acknowledgements

The base detection pipeline (Ultralytics YOLO11 plus the zone-polygon test) was provided by our faculty and extended by our team.

## Known limitations

- Zones are specific to each camera and must be drawn manually.
- The route page is specific to parking1's road layout.
- Web-page occupancy is read from the output video's zone colours.
