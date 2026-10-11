# Project overview

Intelligent Parking detects cars in recorded CCTV footage, marks manually defined parking zones as occupied or available, and provides a web page that routes a driver to the nearest available slot.

## Repository layout

- `src/` contains the frame extractor, zone-selection interface, YOLO detector, and shared parking logic.
- `web/` contains the route-finder page source, its build script, and the occupancy extractor.
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

The detector uses the YOLO11 small model with COCO class 2 (car). It computes each detected car's bounding-box centroid and uses OpenCV's point-in-polygon test to determine whether a slot is occupied. It then draws green available zones, red occupied zones, detected centroids, and occupancy counts on the output frames.
