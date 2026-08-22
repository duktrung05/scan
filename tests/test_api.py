from __future__ import annotations

import asyncio
import io

import httpx
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from tscan.api.app import _save_upload, create_app
from tscan.pipeline import ExtractionPipeline


def request(app, method: str, path: str, **kwargs):
    async def execute():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(execute())


def test_extract_endpoint(settings) -> None:
    pipeline = ExtractionPipeline(settings=settings)
    app = create_app(settings=settings, pipeline=pipeline)
    buffer = io.BytesIO()
    Image.new("RGB", (160, 80), "white").save(buffer, format="PNG")
    response = request(
        app,
        "POST",
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
    upload = UploadFile(file=io.BytesIO(b"x" * (1024 * 1024 + 1)), filename="large.png")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_save_upload(upload, settings))
    assert exc.value.status_code == 413


def test_config_exposes_japanese(settings) -> None:
    app = create_app(settings=settings)
    response = request(app, "GET", "/api/config")
    assert response.status_code == 200
    assert "ja" in response.json()["languages"]
