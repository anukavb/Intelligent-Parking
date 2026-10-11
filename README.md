# Intelligent Parking

Intelligent Parking detects vehicles in CCTV footage, marks each parking slot occupied or free, and guides a driver to the nearest free slot. It combines a live/offline YOLO-based parking monitor, tools for preparing and evaluating camera zones, and a self-contained route-finder page.

## Folder layout

```text
intelligent-parking/
├── README.md
├── requirements.txt
├── src/                     Detection pipeline
├── web/                     Route-finder page source and build scripts
├── app/                     Local live-monitor dashboard
├── tools/                   Labeling, detection cache, evaluation, tuning, benchmark
├── tests/                   Unit and integration tests
├── data/
│   └── parking1/            Sample video, reference frame, zones, occupancy
├── videos/                  New source videos (Git-ignored)
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

Run commands from the repository root. For the included sample, the core offline-to-page flow is:

```powershell
py src/detector.py --mode offline --no-show
py web/extract_occupancy.py data/parking1 outputs/parking1/output.mp4
py web/build.py
```

The detector also supports live display (the default); press Escape to stop. Use `--publish` to copy its per-frame occupancy output into the dataset directory. For a **new video**, place it under `videos/` and let `src/run.py` prepare a reference frame and zones before starting live monitoring:

```powershell
py src/run.py videos\campus_lot.mp4 --name campus_lot
```

The orchestration command opens the zone-selection UI only when zones do not exist yet. You can also do the preparation steps individually:

```powershell
py src/frame_extractor.py --video videos\campus_lot.mp4 --name campus_lot
py src/zone_selector.py --name campus_lot
py src/detector.py --source videos\campus_lot.mp4 --name campus_lot --mode live
```

The detector writes run artifacts under `outputs/<name>/`. `web/extract_occupancy.py` reads the red/green zone markings in an annotated video and writes occupancy data to the selected dataset folder. The page build embeds `parking1`'s image, zones, and occupancy in the generated `docs/index.html`.

### Local dashboard

Start the local-only dashboard, then open `http://127.0.0.1:5000`:

```powershell
py app/server.py
```

The page can monitor the sample video, a camera, or an uploaded video. For a new source, save or draw its zones in the dashboard before starting the session. Outputs and uploads remain local.

### Labeling, evaluation, and tuning

Create manually verified frame/slot labels with the OpenCV labeling window. Keys `0` and `1` label the selected slot free/occupied; `n`/`p` step through frames and the arrow keys (or `[`/`]`) change slots. Labels are saved under `outputs/`, never the checked-in sample data:

```powershell
py tools/label_frames.py --name parking1 --step 15
```

Cache detections once, then evaluate configurations repeatedly without rerunning YOLO:

```powershell
py tools/cache_detections.py --name parking1 --model yolo11s.pt
py tools/evaluate.py --name parking1 --labels outputs\parking1_ground_truth.csv
py tools/benchmark.py --video data\parking1\parking1.mp4 --model yolo11s.pt
```

The cache uses a 0.25 detection-confidence floor; evaluation can raise the confidence threshold but cannot lower it without rebuilding the cache. Inference settings such as ROI and image size are fixed by the cache and cannot be swept against it. To search decision settings, create a JSON grid whose values are arrays (for example, `{"alpha":[0.2,0.3],"on":[0.6,0.7]}`) and run:

```powershell
py tools/sweep.py --name parking1 --labels outputs\parking1_ground_truth.csv --grid outputs\tuning\grid.json --output outputs\tuning\sweep.json
py src/detector.py --use-tuned
```

The sweep writes its best engine configuration to `outputs/tuning/best_params.json`. Tuning quality depends on representative ground-truth labels; keep labels separate from the shipped sample data.

### Tests

Run the automated suite with:

```powershell
py -m pytest tests -q
```

The Tkinter zone editor and browser dashboard should also be exercised manually on the target machine, because automated tests cover their file/API behavior rather than human interaction.

## Acknowledgements

The base detection pipeline (Ultralytics YOLO11 plus the zone-polygon test) was provided by our faculty and extended by our team.

## Known limitations

- Zones are specific to each camera and must be drawn manually.
- The route page is specific to parking1's road layout.
- Web-page occupancy is read from the output video's zone colours.
- Live mode prioritizes responsive display and may skip source frames when inference cannot keep up; use offline mode for sequential reproducible processing.
- Ground-truth evaluation measures only the manually labeled frame/slot pairs and is not a substitute for representative labeling across cameras and conditions.
