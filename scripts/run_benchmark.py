from __future__ import annotations

import argparse
from pathlib import Path

from tscan.evaluation.benchmark import run_benchmark
from tscan.evaluation.report import generate_html_report
from tscan.evaluation.runner import evaluate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSON extraction benchmark")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    evaluation_manifest = run_benchmark(args.manifest, output_dir=args.output_dir)
    summary = evaluate_manifest(evaluation_manifest)
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    generate_html_report(evaluation_manifest, reports / "baseline.html")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
