#!/usr/bin/env python3
"""Run the LFM3-1.7B CPU benchmark on this machine.

Reports load time, time to first token, and tokens per second. Label the
numbers as a cloud VM measurement, not a sealed-host result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sandbox_models.cpu_benchmark import default_cache_dir, run_benchmark, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--prompt", default="Reply with the single word pong.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/evidence/lfm3-1.7b-cpu-benchmark.json"),
    )
    args = parser.parse_args()
    report = run_benchmark(
        cache_dir=args.cache_dir,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    write_report(report, args.report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
