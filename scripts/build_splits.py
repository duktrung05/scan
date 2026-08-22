from __future__ import annotations

import argparse
from pathlib import Path

from tscan.data import build_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic leakage-safe dataset splits")
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("artifacts/dataset_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by", default="source_id")
    args = parser.parse_args()
    result = build_split_manifest(
        args.manifest, output_dir=args.output_dir, seed=args.seed, group_by=args.group_by
    )
    print(f"Wrote {len(result['records'])} split records")


if __name__ == "__main__":
    main()
