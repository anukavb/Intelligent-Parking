import io

import cv2
import numpy as np

from app.server import create_app
from src.engine import ParkingEngine


def test_dashboard_upload_zones_stream_events_stop_and_outputs(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 8, (80, 60))
    assert writer.isOpened()
    for index in range(16):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        cv2.putText(frame, str(index), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()

    def engine_factory(zones, **kwargs):
        return ParkingEngine(zones, detector=lambda _frame: ([], [], []), **kwargs)

    app = create_app(tmp_path, engine_factory=engine_factory)
    app.config.update(TESTING=True)
    client = app.test_client()

    with source.open("rb") as video:
        response = client.post(
            "/api/upload",
            data={"video": (io.BytesIO(video.read()), "demo.mp4")},
            content_type="multipart/form-data",
        )
    assert response.status_code == 200
    assert response.json["name"] == "demo"
    frame_response = client.get("/api/source/demo/frame")
    assert frame_response.status_code == 200
    assert client.get("/api/zones/demo").json["saved"] is False

    zones = [{"points": [[10, 10], [70, 10], [70, 50], [10, 50]]}]
    assert client.post("/api/zones/demo", json={"zones": zones}).status_code == 200
    started = client.post("/api/start", json={"name": "demo", "settings": {"model": "s"}})
    assert started.status_code == 200

    stream = client.get("/stream", buffered=False)
    chunks = [next(stream.response) for _ in range(5)]
    stream.close()
    assert all(b"Content-Type: image/jpeg" in chunk for chunk in chunks)
    assert b"--frame" in chunks[0]

    events = client.get("/events", buffered=False)
    event_chunk = next(events.response)
    events.close()
    assert event_chunk.startswith(b"data: ")
    client.post("/api/stop")

    output_dir = tmp_path / "outputs" / "demo"
    for filename in ("output.mp4", "events.csv", "sessions.csv", "summary.json", "occupancy.json"):
        assert (output_dir / filename).is_file(), filename
