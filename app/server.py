"""Local Flask dashboard for uploading videos and monitoring parking occupancy."""

import json
import math
import re
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine import ParkingEngine
from src.frame_extractor import processing_dimensions
from src.live import LiveRunner
from src.zones import load_zones, save_zones

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 2 * 1024**3
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def create_app(repo_root: str | Path = ROOT, engine_factory=ParkingEngine, runner_factory=LiveRunner):
    """Create an isolated dashboard app whose source registry is process-local."""
    root = Path(repo_root).resolve()
    app = Flask(__name__, static_folder="static", template_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    sources: dict[str, str | int] = {"parking1": str(root / "data" / "parking1" / "parking1.mp4")}
    state_lock = threading.RLock()
    active: dict[str, object] = {"runner": None, "name": None, "starting": False}

    def checked_name(name: str) -> str:
        """Reject dataset names that could escape the repository data directory."""
        if not NAME_PATTERN.fullmatch(name or ""):
            raise ValueError("name may contain only letters, digits, hyphens, and underscores")
        return name

    def data_dir(name: str) -> Path:
        """Resolve a validated dataset name beneath the data root."""
        return root / "data" / checked_name(name)

    def current_runner():
        with state_lock:
            return active["runner"], active["name"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(error="Video upload exceeds the 2 GB limit."), 413

    @app.post("/api/source")
    def select_source():
        payload = request.get_json(silent=False)
        kind = payload.get("kind")
        if kind == "sample":
            name = "parking1"
            source = root / "data" / "parking1" / "parking1.mp4"
        elif kind == "camera":
            index = payload.get("index")
            if not isinstance(index, int) or index < 0:
                return jsonify(error="Camera index must be a non-negative integer."), 400
            name, source = f"camera_{index}", index
        else:
            return jsonify(error="kind must be 'sample' or 'camera'."), 400
        if isinstance(source, Path) and not source.is_file():
            return jsonify(error=f"Sample video not found: {source}"), 404
        with state_lock:
            sources[name] = str(source) if isinstance(source, Path) else source
        return jsonify(name=name, saved_zones=(data_dir(name) / "bounding_boxes.json").is_file())

    @app.post("/api/upload")
    def upload_video():
        uploaded = request.files.get("video")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Select a video file to upload."), 400
        filename = secure_filename(uploaded.filename)
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in ALLOWED_EXTENSIONS:
            return jsonify(error="Supported video formats are MP4, AVI, MOV, and MKV."), 400
        videos_dir = root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        destination = videos_dir / filename
        if destination.exists():
            return jsonify(error="A video with that safe filename already exists."), 409
        uploaded.save(destination)
        name = checked_name(Path(filename).stem)
        if name in sources and name != "parking1":
            destination.unlink(missing_ok=True)
            return jsonify(error="A source with that video name is already registered."), 409
        with state_lock:
            sources[name] = str(destination)
        return jsonify(name=name, filename=filename, saved_zones=(data_dir(name) / "bounding_boxes.json").is_file())

    @app.get("/api/source/<name>/frame")
    def source_frame(name):
        name = checked_name(name)
        with state_lock:
            source = sources.get(name)
        if source is None:
            return jsonify(error="Select or upload this source first."), 404
        folder = data_dir(name)
        saved_image = folder / "img_0.jpg"
        if name == "parking1" and saved_image.is_file():
            return send_file(saved_image, mimetype="image/jpeg")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            return jsonify(error=f"Unable to open video source for {name}."), 400
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            return jsonify(error="Could not read a reference frame from the selected source."), 400
        height, width = frame.shape[:2]
        process_width, process_height = processing_dimensions(width, height)
        resized = cv2.resize(frame, (process_width, process_height))
        folder.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(saved_image), resized):
            return jsonify(error=f"Unable to save reference frame: {saved_image}"), 500
        (folder / "meta.json").write_text(
            json.dumps({"width": process_width, "height": process_height}, indent=2), encoding="utf-8"
        )
        return send_file(saved_image, mimetype="image/jpeg")

    @app.get("/api/zones/<name>")
    def get_zones(name):
        name = checked_name(name)
        path = data_dir(name) / "bounding_boxes.json"
        if not path.is_file():
            return jsonify(saved=False, zones=[]), 200
        return jsonify(saved=True, zones=load_zones(path))

    @app.post("/api/zones/<name>")
    def post_zones(name):
        name = checked_name(name)
        payload = request.get_json(silent=False)
        zones = payload.get("zones") if isinstance(payload, dict) else None
        if not isinstance(zones, list) or not zones or len(zones) > 1000:
            return jsonify(error="zones must be a non-empty list of at most 1000 polygons."), 400
        for zone in zones:
            points = zone.get("points") if isinstance(zone, dict) else None
            if not isinstance(points, list) or len(points) != 4 or any(
                not isinstance(point, list) or len(point) != 2 or
            any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
                for point in points
            ):
                return jsonify(error="Each zone must contain exactly four 2D points."), 400
        path = data_dir(name) / "bounding_boxes.json"
        save_zones(path, zones)
        return jsonify(saved=True, count=len(zones))

    @app.post("/api/start")
    def start_monitor():
        payload = request.get_json(silent=False)
        name = checked_name(payload.get("name", ""))
        zones_path = data_dir(name) / "bounding_boxes.json"
        if not zones_path.is_file():
            return jsonify(error="Save or select parking zones before starting."), 400
        with state_lock:
            if active["starting"]:
                return jsonify(error="A parking session is already starting."), 409
            old_runner = active["runner"]
            if old_runner is not None and not old_runner.done:
                return jsonify(error="A parking session is already active."), 409
            source = sources.get(name)
        if source is None:
            return jsonify(error="The video source is no longer registered."), 404
        settings = payload.get("settings", {})
        model_size = settings.get("model", "s")
        if model_size not in {"n", "s"}:
            return jsonify(error="model must be 'n' or 's'."), 400
        with state_lock:
            if active["starting"]:
                return jsonify(error="A parking session is already starting."), 409
            active["starting"] = True
        try:
            meta_path = data_dir(name) / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"width": 1080, "height": 600}
            classes = [2] if settings.get("cars_only", False) else [2, 3, 5, 7]
            mode = "centroid" if settings.get("rule") == "centroid" else "footprint"
            engine = engine_factory(
                load_zones(zones_path),
                model_path=root / f"yolo11{model_size}.pt",
                classes=classes,
                conf=float(settings.get("conf", 0.35)),
                imgsz=int(settings.get("imgsz", 640)),
                mode=mode,
                smoothing=bool(settings.get("smoothing", True)),
                roi=bool(settings.get("roi", False)),
            )
            output_dir = root / "outputs" / name
            runner = runner_factory(
                engine, source, (int(meta["width"]), int(meta["height"])),
                paced=True, record_path=output_dir / "output.mp4", output_dir=output_dir,
            ).start()
            with state_lock:
                active["runner"], active["name"] = runner, name
        finally:
            with state_lock:
                active["starting"] = False
        return jsonify(started=True, name=name, output_dir=f"outputs/{name}")

    @app.post("/api/stop")
    def stop_monitor():
        runner, _name = current_runner()
        if runner is None:
            return jsonify(error="No parking session is active."), 409
        runner.stop()
        return jsonify(stopped=True)

    @app.get("/api/state")
    def state_snapshot():
        runner, name = current_runner()
        if runner is None:
            return jsonify(running=False, name=None)
        _frame, meta = runner.latest()
        return jsonify(running=not runner.done, done=runner.done, name=name, **meta)

    @app.get("/events")
    def events():
        runner, name = current_runner()
        if runner is None:
            return jsonify(error="No parking session is active."), 409

        @stream_with_context
        def generate():
            previous_frame = -1
            while True:
                _frame, meta = runner.latest()
                if meta and meta.get("frame_idx") != previous_frame:
                    previous_frame = meta["frame_idx"]
                    states = meta.get("states", [])
                    event = {
                        "name": name,
                        "occupied": sum(states),
                        "available": len(states) - sum(states),
                        "lot_full": bool(states) and all(states),
                        "states": states,
                        "inference_ms": meta.get("inference_ms", 0),
                        "inference_hz": meta.get("inference_hz", 0),
                        "display_fps": meta.get("display_fps", 0),
                        "lag_ms": meta.get("lag_ms", 0),
                        "video_time_s": meta.get("video_time_s", 0),
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                if runner.done:
                    yield f"data: {json.dumps({'name': name, 'done': True})}\n\n"
                    return
                time.sleep(0.1)

        return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/stream")
    def stream():
        runner, _name = current_runner()
        if runner is None:
            return jsonify(error="No parking session is active."), 409

        @stream_with_context
        def generate():
            last_frame = -1
            while True:
                frame, meta = runner.latest()
                if frame is None:
                    if runner.done:
                        return
                    time.sleep(0.03)
                    continue
                if meta.get("frame_idx") == last_frame:
                    if runner.done:
                        return
                    time.sleep(0.03)
                    continue
                last_frame = meta["frame_idx"]
                height, width = frame.shape[:2]
                if width > 960:
                    frame = cv2.resize(frame, (960, round(height * 960 / width)))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("Unable to encode the live dashboard frame.")
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                if runner.done:
                    return

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/download/<name>/<filename>")
    def download(name, filename):
        name = checked_name(name)
        allowed = {"output.mp4", "events.csv", "sessions.csv", "summary.json"}
        if filename not in allowed:
            return jsonify(error="Unknown output file."), 404
        path = root / "outputs" / name / filename
        if not path.is_file():
            return jsonify(error="Output is not ready yet."), 404
        return send_file(path, as_attachment=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
