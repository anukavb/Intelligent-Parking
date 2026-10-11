import csv

import numpy as np
import pytest

from tools.common import evaluate_cache, score_states


def test_score_states_reports_occupied_positive_metrics():
    report = score_states([[0, 1], [1, 0]], {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1})
    assert report["confusion_matrix"] == {"tp": 2, "fp": 0, "fn": 1, "tn": 1}
    assert report["precision"] == 1.0
    assert report["recall"] == pytest.approx(2 / 3)
    assert report["f1"] == pytest.approx(0.8)
    assert report["accuracy"] == 0.75


def test_cached_empty_frame_evaluation(tmp_path):
    cache = tmp_path / "fixture.npz"
    np.savez_compressed(
        cache,
        boxes=np.empty((0, 4), np.float32),
        classes=np.empty(0, np.int32),
        confidences=np.empty(0, np.float32),
        offsets=np.asarray([0, 0], np.int64),
        metadata=np.asarray('{"conf": 0.25, "frames": 1, "imgsz": 640, "roi": false}'),
    )
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "slot", "occupied"])
        writer.writerow([0, 0, 0])
    report = evaluate_cache("parking1", {}, cache, labels)
    assert report["label_count"] == 1
    assert report["accuracy"] == 1.0
    assert len(report["states"]) == 1


def test_evaluation_rejects_confidence_below_cache_threshold(tmp_path):
    cache = tmp_path / "fixture.npz"
    np.savez_compressed(
        cache,
        boxes=np.empty((0, 4), np.float32),
        classes=np.empty(0, np.int32),
        confidences=np.empty(0, np.float32),
        offsets=np.asarray([0, 0], np.int64),
        metadata=np.asarray('{"conf": 0.25, "frames": 1, "imgsz": 640, "roi": false}'),
    )
    with pytest.raises(ValueError, match="below the cache threshold"):
        evaluate_cache("parking1", {"conf": 0.1}, cache, tmp_path / "unused.csv")


def test_evaluation_rejects_cache_inference_setting_changes(tmp_path):
    cache = tmp_path / "fixture.npz"
    np.savez_compressed(
        cache,
        boxes=np.empty((0, 4), np.float32),
        classes=np.empty(0, np.int32),
        confidences=np.empty(0, np.float32),
        offsets=np.asarray([0, 0], np.int64),
        metadata=np.asarray('{"conf": 0.25, "frames": 1, "imgsz": 640, "roi": false}'),
    )
    with pytest.raises(ValueError, match="Cannot change roi"):
        evaluate_cache("parking1", {"roi": True}, cache, tmp_path / "unused.csv")
