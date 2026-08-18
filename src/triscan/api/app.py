from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from triscan import __version__
from triscan.config import Settings, get_settings
from triscan.exceptions import ModelError, TriScanError
from triscan.pipeline import ExtractionPipeline
from triscan.schema_registry import SchemaRegistry
from triscan.types import DocumentLanguage, ExtractionMode

LOGGER = logging.getLogger("triscan.api")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "document.bin").suffix.lower()
    return (
        suffix if suffix in {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"} else ".bin"
    )


async def _save_upload(upload: UploadFile, settings: Settings) -> Path:
    path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix="triscan-", suffix=_safe_suffix(upload.filename), delete=False
        ) as handle:
            path = Path(handle.name)
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_file_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_file_mb} MB limit",
                    )
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        assert path is not None
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def create_app(
    settings: Settings | None = None,
    pipeline: ExtractionPipeline | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    app = FastAPI(
        title="TriScan API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.pipeline = pipeline or ExtractionPipeline(settings=settings)
    app.state.registry = app.state.pipeline.registry

    @app.exception_handler(TriScanError)
    async def handle_triscan_error(_request: Request, exc: TriScanError) -> JSONResponse:
        status = 502 if isinstance(exc, ModelError) else 400
        LOGGER.warning("Request failed: %s", exc)
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/assets/{asset_name}", include_in_schema=False)
    async def asset(asset_name: str) -> FileResponse:
        if asset_name not in {"app.css", "app.js"}:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(STATIC_DIR / asset_name)

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "provider": settings.model_provider,
            "model": settings.model_name if settings.model_provider != "mock" else "triscan-mock",
        }

    @app.get("/api/config")
    async def config() -> dict[str, object]:
        return {
            "provider": settings.model_provider,
            "model": settings.model_name,
            "max_file_mb": settings.max_file_mb,
            "max_pdf_pages": settings.max_pdf_pages,
            "max_batch_files": settings.max_batch_files,
            "languages": [item.value for item in DocumentLanguage],
            "modes": [item.value for item in ExtractionMode],
        }

    @app.get("/api/schemas")
    async def schemas() -> dict[str, list[str]]:
        return {"schemas": app.state.registry.list()}

    @app.get("/api/schemas/{name}")
    async def schema(name: str) -> dict:
        return app.state.registry.load(name)

    @app.post("/api/extract")
    async def extract(
        file: Annotated[UploadFile, File(description="PDF or image")],
        mode: Annotated[str, Form()] = "json",
        document_type: Annotated[str, Form()] = "invoice",
        language: Annotated[str, Form()] = "auto",
        schema_text: Annotated[str | None, Form()] = None,
    ) -> dict:
        original_filename = Path(file.filename or "document").name
        temp_path = await _save_upload(file, settings)
        try:
            custom_schema = (
                SchemaRegistry.parse_custom(schema_text)
                if schema_text and schema_text.strip()
                else None
            )
            result = await run_in_threadpool(
                app.state.pipeline.extract,
                temp_path,
                mode=mode,
                document_type=document_type,
                language=language,
                custom_schema=custom_schema,
                original_filename=original_filename,
            )
            return result.model_dump(mode="json")
        finally:
            temp_path.unlink(missing_ok=True)

    @app.get("/api/runs/{run_id}/artifacts/{artifact_name}")
    async def artifact(run_id: str, artifact_name: str) -> FileResponse:
        try:
            run_dir = app.state.pipeline.run_directory(run_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        allowed = set(metadata.get("artifacts", {}).values())
        if artifact_name not in allowed:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(run_dir / artifact_name, filename=f"{run_id}-{artifact_name}")

    @app.post("/api/batch")
    async def batch(
        files: Annotated[list[UploadFile], File(description="Multiple PDFs or images")],
        mode: Annotated[str, Form()] = "json",
        document_type: Annotated[str, Form()] = "invoice",
        language: Annotated[str, Form()] = "auto",
        schema_text: Annotated[str | None, Form()] = None,
    ) -> FileResponse:
        if len(files) > settings.max_batch_files:
            raise HTTPException(
                status_code=400,
                detail=f"Batch limit is {settings.max_batch_files} files",
            )
        custom_schema = (
            SchemaRegistry.parse_custom(schema_text)
            if schema_text and schema_text.strip()
            else None
        )
        temp_uploads: list[tuple[Path, str]] = []
        try:
            for upload in files:
                original_filename = Path(upload.filename or "document").name
                temp_uploads.append((await _save_upload(upload, settings), original_filename))
            results = []
            for path, original_filename in temp_uploads:
                results.append(
                    await run_in_threadpool(
                        app.state.pipeline.extract,
                        path,
                        mode=mode,
                        document_type=document_type,
                        language=language,
                        custom_schema=custom_schema,
                        original_filename=original_filename,
                    )
                )
            with tempfile.NamedTemporaryFile(
                prefix="triscan-batch-", suffix=".zip", delete=False
            ) as archive_handle:
                archive_path = Path(archive_handle.name)
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                summary = [result.model_dump(mode="json") for result in results]
                archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
                for result in results:
                    run_dir = app.state.pipeline.run_directory(result.run_id)
                    for artifact_path in run_dir.iterdir():
                        if artifact_path.name == result.artifacts.get("source"):
                            continue
                        archive.write(artifact_path, f"{result.run_id}/{artifact_path.name}")
            return FileResponse(
                archive_path,
                media_type="application/zip",
                filename="triscan-batch-results.zip",
                background=BackgroundTask(archive_path.unlink, missing_ok=True),
            )
        finally:
            for path, _ in temp_uploads:
                path.unlink(missing_ok=True)

    return app


app = create_app()
