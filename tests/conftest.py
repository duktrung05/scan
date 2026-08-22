from __future__ import annotations

from pathlib import Path

import pytest

from tscan.config import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_provider="mock",
        runs_dir=tmp_path / "runs",
        schemas_dir=tmp_path / "schemas",
        max_file_mb=5,
        max_pdf_pages=4,
        max_image_megapixels=2,
    )
