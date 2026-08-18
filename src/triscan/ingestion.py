from __future__ import annotations

import hashlib
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

from triscan.config import Settings
from triscan.exceptions import DocumentError
from triscan.types import DocumentPage, LoadedDocument

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sniff_media_type(path: Path) -> str:
    with path.open("rb") as handle:
        signature = handle.read(16)
    if signature.startswith(b"%PDF-"):
        return "application/pdf"
    try:
        with Image.open(path) as image:
            image.verify()
            return Image.MIME.get(image.format, "application/octet-stream")
    except (UnidentifiedImageError, OSError) as exc:
        raise DocumentError("Unsupported or corrupted document") from exc


def _resize_if_needed(image: Image.Image, max_megapixels: int) -> Image.Image:
    max_pixels = max_megapixels * 1_000_000
    current_pixels = image.width * image.height
    if current_pixels <= max_pixels:
        return image
    ratio = (max_pixels / current_pixels) ** 0.5
    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _load_image(path: Path, settings: Settings) -> list[DocumentPage]:
    try:
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DocumentError(f"Unable to decode image: {path.name}") from exc
    image = _resize_if_needed(image, settings.max_image_megapixels)
    return [DocumentPage(1, image, image.width, image.height)]


def _load_pdf(path: Path, settings: Settings) -> list[DocumentPage]:
    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise DocumentError(f"Unable to open PDF: {path.name}") from exc

    pages: list[DocumentPage] = []
    try:
        page_count = len(pdf)
        if page_count == 0:
            raise DocumentError("PDF has no pages")
        if page_count > settings.max_pdf_pages:
            raise DocumentError(f"PDF has {page_count} pages; limit is {settings.max_pdf_pages}")
        scale = settings.pdf_dpi / 72.0
        for index in range(page_count):
            page = pdf[index]
            bitmap = None
            try:
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("RGB").copy()
                image = _resize_if_needed(image, settings.max_image_megapixels)
                pages.append(DocumentPage(index + 1, image, image.width, image.height))
            finally:
                if bitmap is not None:
                    bitmap.close()
                page.close()
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Unable to render PDF: {path.name}") from exc
    finally:
        pdf.close()
    return pages


def load_document(path: str | Path, settings: Settings) -> LoadedDocument:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DocumentError(f"Document not found: {source}")
    byte_size = source.stat().st_size
    if byte_size == 0:
        raise DocumentError("Document is empty")
    if byte_size > settings.max_file_bytes:
        raise DocumentError(
            f"Document is {byte_size / 1024 / 1024:.1f} MB; limit is {settings.max_file_mb} MB"
        )

    media_type = _sniff_media_type(source)
    if media_type == "application/pdf":
        pages = _load_pdf(source, settings)
    elif source.suffix.lower() in IMAGE_SUFFIXES or media_type.startswith("image/"):
        pages = _load_image(source, settings)
    else:
        raise DocumentError(f"Unsupported media type: {media_type}")

    return LoadedDocument(
        source_path=source,
        original_filename=source.name,
        media_type=media_type,
        sha256=_sha256(source),
        byte_size=byte_size,
        pages=pages,
    )
