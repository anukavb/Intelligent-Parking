import cv2
import numpy as np

from src.frame_extractor import extract_frame, processing_dimensions


def test_processing_dimensions_scale_width_and_make_height_even():
    assert processing_dimensions(1920, 1080) == (1080, 608)
    assert processing_dimensions(1280, 720) == (1080, 608)


def test_extract_frame_writes_image_and_meta(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    assert writer.isOpened()
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.write(np.full((48, 64, 3), 255, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr("src.frame_extractor.ROOT", tmp_path)

    output = extract_frame(source, "sample", 1)
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (1080, 810)
    capture.release()
    assert (tmp_path / "data/sample/meta.json").read_text(encoding="utf-8") == '{\n  "width": 1080,\n  "height": 810\n}'
