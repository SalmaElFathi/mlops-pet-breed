"""CI quality gate: fail if a metric regressed against the base branch (higher is better)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_baseline(ref: str, path: str) -> dict[str, Any] | None:
    """The metrics file as committed on `ref`, or None if it does not exist there."""
    try:
        out = subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return json.loads(out)


def is_acceptable(candidate: float, baseline: float, max_drop: float) -> bool:
    return candidate >= baseline - max_drop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail if a metric dropped too much")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--metric", default="top1")
    parser.add_argument("--max-drop", type=float, default=0.005)
    args = parser.parse_args(argv)

    candidate = float(json.loads(Path(args.metrics).read_text())[args.metric])
    baseline = load_baseline(args.baseline_ref, args.metrics)
    if baseline is None or args.metric not in baseline:
        print(f"No {args.metric} on {args.baseline_ref}: quality gate skipped.")
        return 0
    base = float(baseline[args.metric])
    print(f"{args.metric}: candidate {candidate:.5f}, base {base:.5f}, max drop {args.max_drop}")
    if not is_acceptable(candidate, base, args.max_drop):
        print(f"FAIL: {args.metric} dropped by {base - candidate:.5f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
