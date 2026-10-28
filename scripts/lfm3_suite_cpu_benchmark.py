#!/usr/bin/env python3
"""Run CPU smoke tests for every published LFM3 suite model that can run here.

Writes one evidence JSON per model plus docs/evidence/lfm3-suite-cpu-benchmark.json.
Every number is from this cloud VM, not a sealed host.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from sandbox_models.catalog import get_model, list_models
from sandbox_models.cpu_benchmark import (
    default_cache_dir,
    evidence_path,
    host_info,
    run_benchmark,
    skipped_report,
    write_report,
)

try:
    from sandbox_models.transformers_benchmark import run_transformers_benchmark
except ImportError:
    run_transformers_benchmark = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--suite-report",
        type=Path,
        default=Path("docs/evidence/lfm3-suite-cpu-benchmark.json"),
    )
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    models = [m for m in list_models() if not args.only or m.id in args.only]
    suite: dict[str, object] = {
        **host_info(),
        "models": {},
    }
    model_reports: dict[str, object] = {}
    for model in models:
        path = evidence_path(model)
        try:
            if model.runtime == "llama.cpp" and model.gguf_filename:
                report = run_benchmark(cache_dir=args.cache_dir, model_id=model.id)
            else:
                report = _run_transformers(model.id, args.cache_dir)
        except (
            RuntimeError,
            OSError,
            ValueError,
            FileNotFoundError,
            ImportError,
        ) as exc:
            report = skipped_report(model, f"{type(exc).__name__}: {exc}")
            report["traceback"] = traceback.format_exc()[-4000:]
        write_report(report, path)
        model_reports[model.id] = report
        print(json.dumps({"model_id": model.id, "path": str(path)}, indent=2))
    suite["models"] = model_reports
    write_report(suite, args.suite_report)
    print(json.dumps({"suite": str(args.suite_report)}, indent=2))
    return 0


def _run_transformers(model_id: str, cache_dir: Path) -> dict[str, object]:
    if run_transformers_benchmark is None:
        return skipped_report(
            get_model(model_id),
            "transformers stack not installed on this cloud VM",
        )
    return run_transformers_benchmark(model_id=model_id, cache_dir=cache_dir)


if __name__ == "__main__":
    raise SystemExit(main())
