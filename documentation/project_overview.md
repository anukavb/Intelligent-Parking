# Project overview

Intelligent Parking detects vehicles in CCTV footage, marks manually defined parking zones as occupied or available, and provides a web page that routes a driver to the nearest available slot. The repository includes a paced live monitor, a sequential offline mode, a local dashboard, and tools for labeling and evaluating detector decisions.

## Repository layout

- `src/` contains the frame extractor, zone-selection interface, YOLO detector, and shared parking logic.
- `web/` contains the route-finder page source, its build script, and the occupancy extractor.
- `app/` provides a local Flask dashboard for video upload/source selection, zone editing, live display, events, and downloads.
- `tools/` contains manual frame labeling, cached YOLO detection, evaluation, parameter sweeps, and throughput benchmarking.
- `tests/` covers geometry, smoothing, runners, zone helpers, and dashboard API behavior.
- `data/parking1/` contains the sample video, reference frame, zone definitions, and web occupancy data.
- `outputs/` receives generated detector videos and is excluded from Git.
- `docs/index.html` is the generated, self-contained GitHub Pages page.
- `documentation/` contains this overview and the [user guide](./user_guide.md).

## Detection pipeline

The frame extractor saves a reference image from the input video. The zone selector lets a user draw four-point polygons for parking slots and saves them in `data/parking1/bounding_boxes.json`. The detector runs Ultralytics YOLO11 on each frame, checks detected car centroids against those polygons, annotates occupied and available zones, and writes `outputs/output.mp4`.

The web occupancy extractor reads the annotated video's red and green zone colours and saves per-frame occupancy in `data/parking1/occupancy.json`. The web build script embeds the frame, zones, occupancy, and route-finding source into `docs/index.html`, which can be opened directly without a server.

## Using a different video

For a new camera/video, extract a reference frame and redraw the slot polygons for that camera. After generating an annotated video, extract occupancy from it and rebuild the page. The route-finder's `DRIVABLE` polygon in `web/routing.js` must also be adjusted to match the new lot layout.

The sample `data/parking1/img_0.jpg`, `data/parking1/bounding_boxes.json`, and `data/parking1/occupancy.json` are already included, so those setup steps are not needed for the provided parking1 sample.

## Detection details

The default engine uses YOLO11 small and COCO vehicle classes 2, 3, 5, and 7. It approximates a vehicle's ground footprint with the bottom 40% of its detection box, then assigns it to the slot with the greatest footprint overlap if that ratio is at least 0.30. A runner-up overlap of at least 0.25 is reported as a possible straddler. `--centroid` provides the original centroid-in-polygon behavior for ablation comparisons.

Each slot maintains a smoothed occupancy evidence value updated once per inference as `e <- e + alpha * (observation - e)`. The defaults are `alpha=0.3`, occupied threshold `on=0.65`, and free threshold `off=0.35`; hysteresis retains the current state between those thresholds. `--no-smoothing` bypasses this state machine. Optional ROI inference crops to the zone bounds plus a margin; live mode separates source-paced display from inference and can drop frames to stay responsive, while offline mode processes sequentially and is intended for repeatable comparisons.

## Labeling and evaluation

`tools/label_frames.py` writes a CSV of manually labeled `(frame, slot, occupied)` observations under `outputs/`. `tools/cache_detections.py` stores one compressed set of YOLO boxes per frame so geometry, confidence, and smoothing configurations can be replayed without repeated model inference. The cache stores detections down to confidence 0.25, so evaluation cannot test lower thresholds unless a new cache is produced with a lower floor.

`tools/evaluate.py` reports occupied-positive precision, recall, F1, accuracy, and a TP/FP/FN/TN confusion matrix on labeled slots, along with raw/smoothed state flips. `tools/sweep.py` runs a Cartesian grid of decision settings, ranks by F1 then accuracy and fewer smoothed flips, and writes the best configuration to `outputs/tuning/best_params.json`. Inference settings such as model, ROI, and image size are fixed when caching; the confidence threshold can only be raised above the cache floor. `src/detector.py --use-tuned` loads the selected configuration. `tools/benchmark.py` measures model inference latency and throughput for a selected video/model/image size.

Evaluation is only as representative as the manual labels. Labels and tuning results are generated under ignored `outputs/` and should not be confused with the included occupancy sample. The Tkinter zone selector and browser dashboard still need manual interaction checks on a desktop environment; automated tests exercise core logic and dashboard endpoints, not full usability or accessibility.
