"""Video-time parking event and session analytics."""

import csv
import json
from pathlib import Path


def summarize(history: list[dict], n_slots: int, fps: float, parameters: dict, display_fps: float = 0.0) -> dict:
    """Summarize occupancy histories using frame-derived video time.

    State durations use the next update timestamp (or video duration) as the
    interval end, avoiding wall-clock timing in reproducible results.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = max((item["frame"] for item in history), default=-1) + 1
    duration = frames / fps
    ordered = sorted(history, key=lambda item: item["frame"])
    events, sessions = [], []
    per_slot_seconds = [0.0] * n_slots
    longest = [0.0] * n_slots
    opens: list[float | None] = [None] * n_slots
    previous_raw = [0] * n_slots
    previous_smooth = [0] * n_slots
    flips_raw = flips_smooth = 0
    peak, peak_time, full_seconds = 0, 0.0, 0.0
    weighted_occupied = 0.0
    inference_times, total_times = [], []
    for position, item in enumerate(ordered):
        time_s = item["frame"] / fps
        end_time = ordered[position + 1]["frame"] / fps if position + 1 < len(ordered) else duration
        delta = max(0.0, end_time - time_s)
        states = item["states"]
        raw = item["raw"]
        occupied = sum(states)
        if occupied > peak:
            peak, peak_time = occupied, time_s
        if occupied == n_slots:
            full_seconds += delta
        weighted_occupied += occupied * delta
        flips_raw += sum(a != b for a, b in zip(previous_raw, raw))
        flips_smooth += sum(a != b for a, b in zip(previous_smooth, states))
        for slot, (raw_state, state) in enumerate(zip(raw, states)):
            if state != previous_smooth[slot]:
                events.append((slot, state, item["frame"], time_s))
                if state:
                    opens[slot] = time_s
                elif opens[slot] is not None:
                    start = opens[slot]
                    stay = time_s - start
                    sessions.append((slot, start, time_s, stay))
                    per_slot_seconds[slot] += stay
                    longest[slot] = max(longest[slot], stay)
                    opens[slot] = None
        previous_raw, previous_smooth = list(raw), list(states)
        inference_times.append(float(item.get("inference_ms", 0)))
        total_times.append(float(item.get("total_ms", item.get("inference_ms", 0))))
    for slot, start in enumerate(opens):
        if start is not None:
            stay = max(0.0, duration - start)
            sessions.append((slot, start, duration, stay))
            per_slot_seconds[slot] += stay
            longest[slot] = max(longest[slot], stay)
    mean_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
    sorted_inference = sorted(inference_times)
    p95 = sorted_inference[min(len(sorted_inference) - 1, int(0.95 * len(sorted_inference)))] if sorted_inference else 0.0
    total_inference_s = sum(inference_times) / 1000
    return {
        "n_slots": n_slots,
        "frames": frames,
        "duration_s": duration,
        "peak_occupancy": peak,
        "peak_occupancy_time_s": peak_time,
        "mean_occupancy_pct": 100 * weighted_occupied / (duration * n_slots) if duration and n_slots else 0.0,
        "lot_full_seconds": full_seconds,
        "flips_raw": flips_raw,
        "flips_smoothed": flips_smooth,
        "per_slot_occupied_seconds": per_slot_seconds,
        "per_slot_longest_stay_s": longest,
        "turnover": sum(len([event for event in events if event[0] == slot and event[1] == 1]) for slot in range(n_slots)),
        "straddle_event_count": sum(int(item.get("straddle_count", 0)) for item in ordered),
        "parameters": parameters,
        "mean_inference_ms": mean_inference,
        "p95_inference_ms": p95,
        "mean_total_ms": sum(total_times) / len(total_times) if total_times else 0.0,
        "effective_inference_hz": 1000 / mean_inference if mean_inference else 0.0,
        "display_fps": display_fps,
    }


def write_analytics(output_dir: str | Path, history: list[dict], summary: dict, occupancy: dict | None = None) -> None:
    """Write event, session, summary, and optional frame occupancy products."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    fps = summary["frames"] / summary["duration_s"] if summary["duration_s"] else 1.0
    with (target / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("slot", "state", "frame", "time_s"))
        previous = [0] * summary["n_slots"]
        for item in history:
            for slot, state in enumerate(item["states"]):
                if state != previous[slot]:
                    writer.writerow((slot, state, item["frame"], item["frame"] / fps))
                previous[slot] = state
    with (target / "sessions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("slot", "start_s", "end_s", "duration_s"))
        opened: list[float | None] = [None] * summary["n_slots"]
        previous = [0] * summary["n_slots"]
        for item in history:
            time_s = item["frame"] / fps
            for slot, state in enumerate(item["states"]):
                if state and not previous[slot]:
                    opened[slot] = time_s
                elif not state and previous[slot] and opened[slot] is not None:
                    writer.writerow((slot, opened[slot], time_s, time_s - opened[slot]))
                    opened[slot] = None
            previous = list(item["states"])
        for slot, start in enumerate(opened):
            if start is not None:
                writer.writerow((slot, start, summary["duration_s"], summary["duration_s"] - start))
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if occupancy is not None:
        (target / "occupancy.json").write_text(json.dumps(occupancy, separators=(",", ":")), encoding="utf-8")
