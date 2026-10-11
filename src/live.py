"""Paced live and deterministic offline runners for the parking engine."""

import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from .analytics import summarize, write_analytics
except ImportError:
    from analytics import summarize, write_analytics


def _open_source(source):
    """Open a video, camera, or URL and raise immediately when it is unavailable."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Unable to open video source: {source}")
    return capture


def _transcode_h264(path: Path) -> None:
    """Replace an MP4V recording with browser-compatible H.264 when ffmpeg exists."""
    if not path.exists():
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        print("H.264 conversion skipped: imageio-ffmpeg is not installed; leaving MP4V output.")
        return
    temporary = path.with_name(path.stem + ".h264" + path.suffix)
    try:
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        temporary.replace(path)
    except (OSError, subprocess.CalledProcessError) as error:
        temporary.unlink(missing_ok=True)
        print(f"H.264 conversion failed; leaving MP4V output: {error}")


def _write_outputs(output_dir, record_path, history, frame_updates, frame_count, fps, n_slots, parameters, display_fps):
    """Write analytics and full-rate forward-filled occupancy after a run."""
    count = frame_count or max(frame_updates, default=-1) + 1
    state = [0] * n_slots
    raw_state = [0] * n_slots
    states_by_frame, raw_by_frame = {}, {}
    for frame_idx, update in sorted(frame_updates.items()):
        state = list(update["states"])
        raw_state = list(update["raw"])
        states_by_frame[frame_idx] = state
        raw_by_frame[frame_idx] = raw_state
    states, raw = [], []
    next_states, next_raw = [0] * n_slots, [0] * n_slots
    update_items = sorted(frame_updates.items())
    pointer = 0
    for frame_idx in range(count):
        while pointer < len(update_items) and update_items[pointer][0] <= frame_idx:
            next_states = list(update_items[pointer][1]["states"])
            next_raw = list(update_items[pointer][1]["raw"])
            pointer += 1
        states.append(next_states.copy())
        raw.append(next_raw.copy())
    complete_history = []
    for item in history:
        entry = dict(item)
        complete_history.append(entry)
    report = summarize(complete_history, n_slots, fps, parameters, display_fps, count)
    occupancy = {"fps": round(fps, 2), "frames": states, "raw_frames": raw}
    if output_dir is not None:
        write_analytics(output_dir, history, report, occupancy)
    elif record_path is not None:
        write_analytics(Path(record_path).parent, history, report, occupancy)
    return report, occupancy


class LiveRunner:
    """Display at source cadence while a worker infers on the newest frame."""

    def __init__(
        self,
        engine,
        source,
        processing_size=(1080, 600),
        paced=True,
        record_path=None,
        output_dir=None,
        max_frames=None,
    ):
        self.engine = engine
        self.source = source
        self.processing_size = tuple(processing_size)
        self.paced = paced
        self.record_path = Path(record_path) if record_path else None
        self.output_dir = Path(output_dir) if output_dir else (self.record_path.parent if self.record_path else None)
        self.max_frames = max_frames
        self.capture = _open_source(source)
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.source_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._lock = threading.RLock()
        self._reader_done = False
        self._done = False
        self._error = None
        self._raw = None
        self._annotated = None
        self._decision = engine.last_decision
        self._detections = engine.last_detections
        self._reader_thread = None
        self._inference_thread = None
        self._completion_thread = None
        self._writer = None
        self._history = []
        self._frame_updates = {}
        self._inference_ticks = deque(maxlen=30)
        self._display_ticks = deque(maxlen=30)
        self._lag_samples = deque(maxlen=300)
        self._start_time = None
        self._display_count = 0

    @property
    def done(self):
        return self._done

    def start(self):
        """Open recording output and start the independent reader/inference workers."""
        if self._reader_thread is not None:
            raise RuntimeError("LiveRunner can only be started once")
        self._start_time = time.perf_counter()
        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self.record_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.source_fps,
                self.processing_size,
            )
            if not self._writer.isOpened():
                raise OSError(f"Unable to create recording: {self.record_path}")
        self._reader_thread = threading.Thread(target=self._reader_loop, name="parking-reader", daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, name="parking-inference", daemon=True)
        self._inference_thread.start()
        self._reader_thread.start()
        self._completion_thread = threading.Thread(target=self._completion_loop, name="parking-completion", daemon=True)
        self._completion_thread.start()
        return self

    def _reader_loop(self):
        """Read and publish source-paced display frames without waiting for inference."""
        frame_idx = 0
        interval = 1 / self.source_fps
        start = time.perf_counter()
        try:
            while not self._stop.is_set() and (self.max_frames is None or frame_idx < self.max_frames):
                if self.paced:
                    target = start + frame_idx * interval
                    delay = target - time.perf_counter()
                    if delay > 0 and self._stop.wait(delay):
                        break
                    behind = time.perf_counter() - target
                    if behind > interval:
                        drop = int(behind / interval)
                        for _ in range(drop):
                            if not self.capture.grab():
                                self._reader_done = True
                                return
                        frame_idx += drop
                ok, frame = self.capture.read()
                if not ok:
                    break
                t_capture = time.perf_counter()
                frame = cv2.resize(frame, self.processing_size)
                video_time_s = frame_idx / self.source_fps
                with self._lock:
                    raw = {
                        "frame": frame.copy(),
                        "frame_idx": frame_idx,
                        "video_time_s": video_time_s,
                        "t_capture": t_capture,
                    }
                    self._raw = raw
                    dets, decision = self._detections, self._decision
                now = time.perf_counter()
                self._display_ticks.append(now)
                elapsed = now - self._start_time
                display_fps = (len(self._display_ticks) - 1) / max(
                    self._display_ticks[-1] - self._display_ticks[0], 1e-6
                ) if len(self._display_ticks) > 1 else 0.0
                inference_hz = self._rate(self._inference_ticks)
                meta = {
                    "frame_idx": frame_idx,
                    "video_time_s": video_time_s,
                    "t_capture": t_capture,
                    "inference_ms": dets.inference_ms,
                    "inference_hz": inference_hz,
                    "display_fps": display_fps,
                    "lag_ms": max(0.0, (now - t_capture) * 1000),
                    "states": decision.states.copy(),
                    "raw": decision.raw.copy(),
                    "counts": dict(decision.counts),
                }
                annotated = self.engine.draw(
                    frame, dets, decision,
                    {"Inference": f"{meta['inference_ms']:.0f} ms", "Display": f"{display_fps:.1f} fps", "Lag": f"{meta['lag_ms']:.0f} ms"},
                )
                meta["lag_ms"] = max(0.0, (time.perf_counter() - t_capture) * 1000)
                self._lag_samples.append(meta["lag_ms"])
                with self._lock:
                    self._annotated = (annotated, meta)
                if self._writer:
                    self._writer.write(annotated)
                with self._condition:
                    self._condition.notify_all()
                self._display_count += 1
                frame_idx += 1
            self._reader_done = True
        except Exception as error:
            self._set_error(error)
            self._reader_done = True
        finally:
            self.capture.release()
            if self._writer:
                self._writer.release()

    def _inference_loop(self):
        """Infer only the newest raw frame, publishing states atomically."""
        last_idx = -1
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop.is_set() or self._reader_done or (
                            self._raw is not None and self._raw["frame_idx"] > last_idx
                        ),
                        timeout=0.1,
                    )
                    with self._lock:
                        raw = None if self._raw is None else dict(self._raw)
                    reader_done = self._reader_done
                if raw is not None and raw["frame_idx"] > last_idx:
                    start = time.perf_counter()
                    detections, decision = self.engine.step(raw["frame"])
                    total_ms = (time.perf_counter() - start) * 1000
                    completed = time.perf_counter()
                    self._inference_ticks.append(completed)
                    item = {
                        "frame": raw["frame_idx"],
                        "video_time_s": raw["video_time_s"],
                        "raw": decision.raw.copy(),
                        "states": decision.states.copy(),
                        "inference_ms": detections.inference_ms,
                        "total_ms": total_ms,
                        "straddle_count": len(decision.straddlers),
                    }
                    with self._lock:
                        self._detections = detections
                        self._decision = decision
                        self._history.append(item)
                        self._frame_updates[raw["frame_idx"]] = item
                    last_idx = raw["frame_idx"]
                    continue
                if reader_done or self._stop.is_set():
                    break
            self._reader_done = True
        except Exception as error:
            self._set_error(error)
            self._stop.set()

    @staticmethod
    def _rate(ticks):
        """Calculate event rate from the observed monotonic interval."""
        return (len(ticks) - 1) / (ticks[-1] - ticks[0]) if len(ticks) > 1 and ticks[-1] > ticks[0] else 0.0

    def _set_error(self, error):
        with self._lock:
            if self._error is None:
                self._error = error

    def _completion_loop(self):
        """Finalize outputs after both processing workers have fully stopped."""
        try:
            self._reader_thread.join()
            self._inference_thread.join()
            with self._lock:
                updates = dict(self._frame_updates)
                history = list(self._history)
                display_fps = self._rate(self._display_ticks)
            if self.output_dir is not None or self.record_path is not None:
                report, _ = _write_outputs(
                    self.output_dir, self.record_path, history, updates,
                    self.source_frame_count, self.source_fps, len(self.engine.zones),
                    self.parameters, display_fps,
                )
                lag_samples = sorted(self._lag_samples)
                report["runtime_s"] = time.perf_counter() - self._start_time
                report["effective_inference_hz"] = self._rate(self._inference_ticks)
                report["mean_lag_ms"] = sum(lag_samples) / len(lag_samples) if lag_samples else 0.0
                report["p95_lag_ms"] = lag_samples[min(len(lag_samples) - 1, int(0.95 * len(lag_samples)))] if lag_samples else 0.0
                (self.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if self.record_path:
                _transcode_h264(self.record_path)
        except Exception as error:
            self._set_error(error)
        finally:
            self._done = True
            with self._condition:
                self._condition.notify_all()

    @property
    def parameters(self):
        return {
            "mode": self.engine.mode,
            "classes": self.engine.classes,
            "conf": self.engine.conf,
            "imgsz": self.engine.imgsz,
            "smoothing": self.engine.smoothing,
            "footprint_frac": self.engine.footprint_frac,
            "overlap_threshold": self.engine.overlap_threshold,
            "straddle_threshold": self.engine.straddle_threshold,
            "roi": self.engine.roi,
        }

    def latest(self):
        """Return the latest annotated frame and a live metadata snapshot."""
        with self._lock:
            if self._annotated is None:
                return None, {}
            image, meta = self._annotated
            snapshot = dict(meta)
            snapshot["inference_hz"] = self._rate(self._inference_ticks)
            snapshot["lag_ms"] = max(0.0, (time.perf_counter() - snapshot["t_capture"]) * 1000)
            return image.copy(), snapshot

    def latest_annotated(self):
        """Expose the newest frame through the web server's polling interface."""
        return self.latest()

    def stop(self):
        """Request shutdown and join both workers, surfacing worker exceptions."""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._reader_thread:
            self._reader_thread.join()
        if self._inference_thread:
            self._inference_thread.join()
        if self._completion_thread:
            self._completion_thread.join()
        self.raise_if_failed()

    def raise_if_failed(self):
        """Reraise the first captured worker exception in the caller thread."""
        if self._error is not None:
            raise RuntimeError("LiveRunner worker failed") from self._error


def run_offline(engine, source, stride=1, output_dir=None, record_path=None, processing_size=(1080, 600), max_frames=None):
    """Process frames sequentially as fast as possible for reproducible evaluation.

    This mode is deterministic and intentionally separate from paced live mode;
    it supports ablations and benchmarks without thread scheduling variability.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    capture = _open_source(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target = Path(record_path) if record_path else None
    writer = None
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, processing_size)
        if not writer.isOpened():
            capture.release()
            raise OSError(f"Unable to create recording: {target}")
    history, updates = [], {}
    last_dets, last_decision = engine.last_detections, engine.last_decision
    total_start = time.perf_counter()
    index = 0
    try:
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, processing_size)
            if index % stride == 0:
                step_start = time.perf_counter()
                last_dets, last_decision = engine.step(frame)
                total_ms = (time.perf_counter() - step_start) * 1000
                update = {
                    "frame": index,
                    "video_time_s": index / fps,
                    "raw": last_decision.raw.copy(),
                    "states": last_decision.states.copy(),
                    "inference_ms": last_dets.inference_ms,
                    "total_ms": total_ms,
                    "straddle_count": len(last_decision.straddlers),
                }
                history.append(update)
                updates[index] = update
            display = engine.draw(frame, last_dets, last_decision)
            if writer:
                writer.write(display)
            index += 1
    finally:
        capture.release()
        if writer:
            writer.release()
    elapsed = time.perf_counter() - total_start
    report, _ = _write_outputs(
        output_dir, target, history, updates, index, fps, len(engine.zones),
        {"mode": engine.mode, "classes": engine.classes, "conf": engine.conf, "imgsz": engine.imgsz, "smoothing": engine.smoothing, "stride": stride},
        index / elapsed if elapsed else 0.0,
    )
    report["runtime_s"] = elapsed
    report["effective_inference_hz"] = len(history) / elapsed if elapsed else 0.0
    if output_dir is not None:
        (Path(output_dir) / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if target:
        _transcode_h264(target)
    return report
