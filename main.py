"""Command-line interface for the deterministic evaluator."""

from __future__ import annotations

import argparse
import json
import sys

from evaluator import EvaluationError, evaluate_items, load_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score model answers against reference answers using deterministic rules."
    )
    parser.add_argument("input_file", help="Path to a JSON file containing evaluation items.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        items = load_items(args.input_file)
        output = evaluate_items(items)
    except EvaluationError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
