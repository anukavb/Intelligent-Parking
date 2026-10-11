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

Run these commands from the repository root for the supplied parking1 sample:

```powershell
py src/detector.py --mode offline --no-show
py web/extract_occupancy.py data/parking1 outputs/parking1/output.mp4
py web/build.py
```

The detector defaults to live display; use `--mode offline --no-show` for sequential, reproducible processing. Frame extraction and zone selection are needed only for a **new video/camera**. The included parking1 sample already has its reference frame and zone definitions.

For a new video, place it under `videos/` and run the orchestration command:

```powershell
py src/run.py videos\campus_lot.mp4 --name campus_lot
```

This extracts a reference frame and opens zone selection if needed, then starts live monitoring. To prepare those steps individually, run `src/frame_extractor.py --video ... --name ...` and `src/zone_selector.py --name ...`. The zone selector saves four-point parking-slot polygons to `data/<name>/bounding_boxes.json`.

The detector writes recordings and analytics under `outputs/<name>/`. The local browser dashboard is available by running `py app/server.py` and opening `http://127.0.0.1:5000`.

The occupancy extractor accepts an optional data folder and annotated video path:

```powershell
py web/extract_occupancy.py data/parking1 outputs/parking1/output.mp4
```

It reads the annotated zone colours and writes `occupancy.json` into the selected data folder. The build step regenerates the self-contained `docs/index.html` page. The route-finder is specific to parking1's road layout; for another lot, update the `DRIVABLE` polygon in `web/routing.js` before rebuilding.

## Ground-truth tools

Create a CSV of manually labeled frame/slot states under ignored `outputs/` with:

```powershell
py tools/label_frames.py --name parking1 --step 15
```

Use `0` for free, `1` for occupied, `n`/`p` to change frames, arrow keys or `[`/`]` to change slots, `s` to save, and `q` to quit. Cache YOLO detections once and compare decision settings against the labeled observations:

```powershell
py tools/cache_detections.py --name parking1 --model yolo11s.pt
py tools/evaluate.py --name parking1 --labels outputs\parking1_ground_truth.csv
py tools/sweep.py --name parking1 --labels outputs\parking1_ground_truth.csv --grid outputs\tuning\grid.json
py tools/benchmark.py --video data\parking1\parking1.mp4 --model yolo11s.pt
```

The cache stores detections at confidence 0.25; evaluation may use higher confidence thresholds. Sweeps rank decision settings by F1, accuracy, then fewer smoothed state changes, and save the selected configuration under `outputs/tuning/best_params.json`. Inference settings such as ROI and image size are fixed by each cache. `py src/detector.py --use-tuned` loads the selected engine configuration.
