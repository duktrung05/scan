from __future__ import annotations

import argparse
from pathlib import Path

from tscan.config import get_settings
from tscan.data import audit_registry, load_dataset_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit datasets registered in configs/datasets.yaml")
    parser.add_argument("registry", type=Path, nargs="?", default=Path("configs/datasets.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    report = audit_registry(load_dataset_registry(args.registry), schemas_dir=get_settings().schemas_dir, output_dir=args.output_dir.resolve())
    print(f"Accepted {report['summary']['accepted']}; rejected {report['summary']['rejected']}")


if __name__ == "__main__":
    main()
