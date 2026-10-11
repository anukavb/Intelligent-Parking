"""Evaluate cached detections against manually labeled slot states."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.common import cache_path, evaluate_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="parking1")
    parser.add_argument("--labels")
    parser.add_argument("--cache")
    parser.add_argument("--config", help="JSON file containing engine configuration overrides")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    labels = Path(args.labels) if args.labels else ROOT / "outputs" / f"{args.name}_ground_truth.csv"
    cache = Path(args.cache) if args.cache else cache_path(args.name)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    result = evaluate_cache(args.name, config, cache, labels)
    result.pop("states")
    result.pop("raw_states")
    rendered = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
