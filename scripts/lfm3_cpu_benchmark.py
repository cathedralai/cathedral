#!/usr/bin/env python3
"""Run an LFM3 suite CPU benchmark on this machine.

Reports load time, time to first token, and tokens per second. Label the
numbers as a cloud VM measurement, not a sealed-host result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sandbox_models.catalog import LFM3_1_7B_ID, get_model
from sandbox_models.cpu_benchmark import (
    default_cache_dir,
    evidence_path,
    run_benchmark,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--model-id", default=LFM3_1_7B_ID)
    parser.add_argument("--prompt", default="Reply with the single word pong.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    model = get_model(args.model_id)
    report_path = args.report or evidence_path(model)
    report = run_benchmark(
        cache_dir=args.cache_dir,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        model_id=args.model_id,
    )
    write_report(report, report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
