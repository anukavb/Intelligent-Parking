"""Shared defaults for the parking decision engine."""

DEFAULTS = {
    "classes": (2, 3, 5, 7),
    "conf": 0.35,
    "imgsz": 640,
    "mode": "footprint",
    "smoothing": True,
    "footprint_frac": 0.4,
    "overlap_threshold": 0.30,
    "straddle_threshold": 0.25,
    "alpha": 0.3,
    "on": 0.65,
    "off": 0.35,
    "roi": False,
    "roi_margin": 40,
}
