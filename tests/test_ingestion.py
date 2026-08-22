from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tscan.exceptions import DocumentError
from tscan.ingestion import load_document


def test_load_image(settings, tmp_path: Path) -> None:
    source = tmp_path / "invoice.png"
    Image.new("RGB", (320, 180), "white").save(source)
    document = load_document(source, settings)
    assert document.media_type == "image/png"
    assert document.byte_size > 0
    assert len(document.sha256) == 64
    assert len(document.pages) == 1
    assert document.pages[0].width == 320


def test_rejects_fake_extension(settings, tmp_path: Path) -> None:
    source = tmp_path / "fake.png"
    source.write_text("not an image", encoding="utf-8")
    with pytest.raises(DocumentError):
        load_document(source, settings)


def test_load_multi_page_pdf(settings, tmp_path: Path) -> None:
    source = tmp_path / "two-pages.pdf"
    pages = [Image.new("RGB", (180, 120), color) for color in ("white", "lightgray")]
    pages[0].save(source, format="PDF", save_all=True, append_images=pages[1:])
    document = load_document(source, settings)
    assert document.media_type == "application/pdf"
    assert [page.page_number for page in document.pages] == [1, 2]
