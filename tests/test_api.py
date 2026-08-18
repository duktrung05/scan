from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from triscan.api.app import create_app
from triscan.pipeline import ExtractionPipeline


def test_extract_endpoint(settings) -> None:
    pipeline = ExtractionPipeline(settings=settings)
    client = TestClient(create_app(settings=settings, pipeline=pipeline))
    buffer = io.BytesIO()
    Image.new("RGB", (160, 80), "white").save(buffer, format="PNG")
    response = client.post(
        "/api/extract",
        files={"file": ("invoice.png", buffer.getvalue(), "image/png")},
        data={"mode": "json", "document_type": "invoice", "language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "json"
    assert body["original_filename"] == "invoice.png"
    assert body["valid"] is True
    assert "json" in body["artifacts"]


def test_rejects_oversized_upload(settings) -> None:
    settings.max_file_mb = 1
    client = TestClient(create_app(settings=settings))
    response = client.post(
        "/api/extract",
        files={"file": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
        data={"mode": "json", "document_type": "invoice", "language": "en"},
    )
    assert response.status_code == 413
