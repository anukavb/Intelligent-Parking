"""Grid-search decision-engine settings against labeled frames and cached boxes."""

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.common import cache_path, evaluate_cache
from src.config import DEFAULTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="parking1")
    parser.add_argument("--labels")
    parser.add_argument("--cache")
    parser.add_argument("--grid", required=True, help="JSON file mapping configuration names to lists of values")
    parser.add_argument("--base-config", help="JSON file containing settings shared by all trials")
    parser.add_argument("--output")
    parser.add_argument("--best-params", default=str(ROOT / "outputs" / "tuning" / "best_params.json"))
    args = parser.parse_args(argv)
    labels = Path(args.labels) if args.labels else ROOT / "outputs" / f"{args.name}_ground_truth.csv"
    cache = Path(args.cache) if args.cache else cache_path(args.name)
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Grid must contain non-empty lists of candidate values")
    base = json.loads(Path(args.base_config).read_text(encoding="utf-8")) if args.base_config else {}
    unknown = (set(grid) | set(base)) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown engine configuration setting(s): {', '.join(sorted(unknown))}")
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[key] for key in keys)):
        config = {**base, **dict(zip(keys, values))}
        result = evaluate_cache(args.name, config, cache, labels)
        result.pop("states")
        result.pop("raw_states")
        results.append(result)
    results.sort(key=lambda result: (result["f1"], result["accuracy"], -result["flips_smoothed"]), reverse=True)
    report = {"trials": len(results), "best": results[0], "results": results}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    best_params = Path(args.best_params)
    best_params.parent.mkdir(parents=True, exist_ok=True)
    best_params.write_text(json.dumps(results[0]["configuration"], indent=2) + "\n", encoding="utf-8")
    report["best_params_file"] = str(best_params)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
