from __future__ import annotations

import argparse
from pathlib import Path

from tscan.evaluation.report import generate_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TScan benchmark HTML report")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/baseline.html"))
    args = parser.parse_args()
    generate_html_report(args.manifest, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
