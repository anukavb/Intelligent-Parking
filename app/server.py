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


def json_error(message: str, hint: str | None = None, status: int = 400):
    """Return a single error payload shape that the UI can render verbosely."""
    payload = {"error": message}
    if hint:
        payload["hint"] = hint
    return jsonify(payload), status


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
        return json_error("Video upload exceeds the 2 GB limit.", "Try a smaller file or reduce the uploaded size.", 413)

    @app.errorhandler(404)
    def not_found(_error):
        return json_error("The requested page or API route was not found.", "Check the URL and try again.", 404)

    @app.errorhandler(500)
    def server_error(_error):
        return json_error("The server ran into an unexpected problem.", "Retry the action and check the server log if it keeps failing.", 500)

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
                return json_error("Camera index must be a non-negative integer.", "Choose a valid camera number to continue.", 400)
            name, source = f"camera_{index}", index
        else:
            return json_error("kind must be 'sample' or 'camera'.", "Pick a sample or a camera index from the dashboard.", 400)
        if isinstance(source, Path) and not source.is_file():
            return json_error(f"Sample video not found: {source}", "The sample file is missing from the project data folder.", 404)
        with state_lock:
            sources[name] = str(source) if isinstance(source, Path) else source
        return jsonify(name=name, saved_zones=(data_dir(name) / "bounding_boxes.json").is_file())

    @app.post("/api/upload")
    def upload_video():
        uploaded = request.files.get("video")
        if uploaded is None or not uploaded.filename:
            return json_error("Select a video file to upload.", "Choose a valid MP4, AVI, MOV, or MKV file.", 400)
        filename = secure_filename(uploaded.filename)
        suffix = Path(filename).suffix.lower()
        if not filename or suffix not in ALLOWED_EXTENSIONS:
            return json_error("Supported video formats are MP4, AVI, MOV, and MKV.", "The file type is not accepted by the dashboard.", 400)
        name = Path(filename).stem
        if not NAME_PATTERN.fullmatch(name):
            return json_error(
                "The video filename must have a simple name before its extension.",
                "Use only letters, digits, hyphens, or underscores in the filename.",
                400,
            )
        with state_lock:
            if name in sources:
                return json_error("A source with that video name is already registered.", "Choose another filename to avoid replacing its zones or reference frame.", 409)
        videos_dir = root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        destination = videos_dir / filename
        if destination.exists():
            return json_error("A video with that safe filename already exists.", "Use a different file name or remove the old upload.", 409)
        uploaded.save(destination)
        with state_lock:
            sources[name] = str(destination)
        return jsonify(name=name, filename=filename, saved_zones=(data_dir(name) / "bounding_boxes.json").is_file())

    @app.get("/api/source/<name>/frame")
    def source_frame(name):
        name = checked_name(name)
        with state_lock:
            source = sources.get(name)
        if source is None:
            return json_error("Select or upload this source first.", "Choose a valid source from the dashboard before selecting zones.", 404)
        folder = data_dir(name)
        saved_image = folder / "img_0.jpg"
        if name == "parking1" and saved_image.is_file():
            return send_file(saved_image, mimetype="image/jpeg")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            return json_error(f"Unable to open video source for {name}.", "The chosen file may be unreadable or corrupted.", 400)
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            return json_error("Could not read a reference frame from the selected source.", "The video may be empty or not readable by OpenCV.", 400)
        height, width = frame.shape[:2]
        process_width, process_height = processing_dimensions(width, height)
        resized = cv2.resize(frame, (process_width, process_height))
        folder.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(saved_image), resized):
            return json_error(f"Unable to save reference frame: {saved_image}", "The reference frame could not be written to disk.", 500)
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
            return json_error("zones must be a non-empty list of at most 1000 polygons.", "Draw a valid parking-slot polygon for each slot.", 400)
        for zone in zones:
            points = zone.get("points") if isinstance(zone, dict) else None
            if not isinstance(points, list) or len(points) != 4 or any(
                not isinstance(point, list) or len(point) != 2 or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
                for point in points
            ):
                return json_error("Each zone must contain exactly four 2D points.", "Use a quadrilateral outline for each parking bay.", 400)
        path = data_dir(name) / "bounding_boxes.json"
        save_zones(path, zones)
        return jsonify(saved=True, count=len(zones))

    @app.post("/api/start")
    def start_monitor():
        payload = request.get_json(silent=False)
        name = checked_name(payload.get("name", ""))
        zones_path = data_dir(name) / "bounding_boxes.json"
        if not zones_path.is_file():
            return json_error("Save or select parking zones before starting.", "Mark the parking bays first, then start the session.", 400)
        with state_lock:
            if active["starting"]:
                return json_error("A parking session is already starting.", "Wait for the current run to finish before starting another.", 409)
            old_runner = active["runner"]
            if old_runner is not None and not old_runner.done:
                return json_error("A parking session is already active.", "Stop the active video before starting a new one.", 409)
            source = sources.get(name)
        if source is None:
            return json_error("The video source is no longer registered.", "Re-select the source and try again.", 404)
        settings = payload.get("settings", {})
        model_size = settings.get("model", "s")
        if model_size not in {"n", "s"}:
            return json_error("model must be 'n' or 's'.", "Choose a YOLO11n or YOLO11s model.", 400)
        with state_lock:
            if active["starting"]:
                return json_error("A parking session is already starting.", "Wait for the current run to finish before starting another.", 409)
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
            return json_error("No parking session is active.", "Start a live monitor before trying to stop it.", 409)
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
            return json_error("No parking session is active.", "Start a live monitor before streaming occupancy updates.", 409)

        @stream_with_context
        def generate():
            previous_frame = -1
            while True:
                _frame, meta = runner.latest()
                if meta and meta.get("frame_idx") != previous_frame:
                    previous_frame = meta["frame_idx"]
                    states = meta.get("states", [])
                    counts = meta.get("counts", {})
                    occupied = counts.get("occupied", sum(states))
                    available = counts.get("available", len(states) - sum(states))
                    event = {
                        "name": name,
                        "occupied": occupied,
                        "available": available,
                        "lot_full": occupied > 0 and available == 0,
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
            return json_error("No parking session is active.", "Start a live monitor before streaming frames.", 409)

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
            return json_error("Unknown output file.", "Choose one of the generated dashboard artifacts.", 404)
        path = root / "outputs" / name / filename
        if not path.is_file():
            return json_error("Output is not ready yet.", "Wait until the analysis has completed before downloading it.", 404)
        return send_file(path, as_attachment=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
