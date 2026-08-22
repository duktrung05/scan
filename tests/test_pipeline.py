from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tscan.pipeline import ExtractionPipeline


def test_mock_pipeline_creates_auditable_artifacts(settings, tmp_path: Path) -> None:
    source = tmp_path / "receipt.png"
    Image.new("RGB", (240, 120), "white").save(source)
    pipeline = ExtractionPipeline(settings=settings)
    result = pipeline.extract(source, document_type="receipt", language="vi")
    assert result.valid is True
    assert result.inference.provider == "mock"
    run_dir = settings.runs_dir / result.run_id
    assert (run_dir / "raw.txt").is_file()
    assert (run_dir / "raw.json").is_file()
    assert (run_dir / "normalized.json").is_file()
    assert (run_dir / "audit.json").is_file()
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "result.csv").is_file()
    assert (run_dir / "result.xlsx").is_file()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["document_sha256"] == result.document_sha256
    assert metadata["raw_value"] == result.raw_value
    assert metadata["schema_valid"] is True
