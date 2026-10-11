import time

import cv2
import numpy as np

from src.engine import ParkingEngine
from src.live import LiveRunner


def test_live_pacing_is_independent_of_slow_inference(tmp_path):
    source = tmp_path / "slow-inference.mp4"
    fps, frame_count, size = 10.0, 24, (100, 100)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for _ in range(frame_count):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()

    calls = 0

    def slow_detector(_frame):
        nonlocal calls
        calls += 1
        time.sleep(0.3)
        if calls < 2:
            return [], [], []
        return [(20, 20, 60, 65)], [2], [0.9]

    zones = [{"points": [[10, 10], [80, 10], [80, 80], [10, 80]]}]
    engine = ParkingEngine(zones, detector=slow_detector)
    runner = LiveRunner(engine, str(source), processing_size=size, paced=True).start()
    state_frame = None
    displayed_red = False
    try:
        deadline = time.perf_counter() + 6
        while not runner.done and time.perf_counter() < deadline:
            frame, meta = runner.latest()
            if frame is not None:
                if meta.get("states") == [1]:
                    state_frame = meta["frame_idx"] if state_frame is None else state_frame
                    if meta["frame_idx"] > state_frame:
                        displayed_red = tuple(frame[80, 45]) == (0, 0, 255)
            time.sleep(0.005)
    finally:
        runner.stop()

    assert runner.done
    assert calls >= 3
    assert state_frame is not None
    assert displayed_red
    _frame, meta = runner.latest()
    assert 0.9 * fps <= meta["display_fps"] <= 1.1 * fps
