from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triscan.config import Settings, get_settings
from triscan.exporters import export_csv, export_xlsx
from triscan.ingestion import load_document
from triscan.models import create_model
from triscan.models.base import ModelRequest, VisionModel
from triscan.prompts import PROMPT_VERSION, build_prompt
from triscan.schema_registry import SchemaRegistry
from triscan.types import (
    DocumentLanguage,
    ExtractionMode,
    ExtractionResult,
    InferenceMetadata,
)
from triscan.validation import validate_and_normalize


class ExtractionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        model: VisionModel | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or create_model(self.settings)
        self.registry = registry or SchemaRegistry(self.settings.schemas_dir)
        self._model_lock = threading.Lock()
        self.settings.runs_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        source: str | Path,
        *,
        mode: ExtractionMode | str = ExtractionMode.JSON,
        document_type: str = "invoice",
        language: DocumentLanguage | str = DocumentLanguage.AUTO,
        custom_schema: dict[str, Any] | None = None,
        original_filename: str | None = None,
    ) -> ExtractionResult:
        mode = ExtractionMode(mode)
        language = DocumentLanguage(language)
        document_type = document_type.strip().lower().replace(" ", "_")
        schema = None
        if mode == ExtractionMode.JSON:
            schema = custom_schema or self.registry.load(document_type)

        document = load_document(source, self.settings)
        if original_filename:
            document.original_filename = Path(original_filename).name
        prompt = build_prompt(
            mode=mode,
            language=language,
            document_type=document_type,
            schema=schema,
        )
        request = ModelRequest(
            pages=document.pages,
            prompt=prompt,
            mode=mode,
            document_type=document_type,
            language=language,
            schema=schema,
        )
        started = time.perf_counter()
        with self._model_lock:
            response = self.model.generate(request)
        latency_ms = round((time.perf_counter() - started) * 1000)

        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
        result = ExtractionResult(
            run_id=run_id,
            created_at=datetime.now(UTC),
            document_id=document.sha256[:16],
            original_filename=document.original_filename,
            document_sha256=document.sha256,
            document_type=document_type,
            language=language,
            mode=mode,
            raw_output=response.text,
            inference=InferenceMetadata(
                provider=response.provider,
                model=response.model,
                latency_ms=latency_ms,
                prompt_version=PROMPT_VERSION,
                page_count=len(document.pages),
                extra=response.metadata,
            ),
        )

        if mode == ExtractionMode.JSON:
            validation = validate_and_normalize(response.text, schema or {})
            result.parsed_output = validation.parsed
            result.normalized_output = validation.normalized
            result.valid = validation.valid
            result.repaired = validation.repaired
            result.issues = validation.issues
            result.inference.extra["repair_actions"] = validation.repair_actions
        else:
            result.valid = None

        self._persist(result, document.source_path)
        return result

    def _persist(self, result: ExtractionResult, source_path: Path) -> None:
        run_dir = self.settings.runs_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        source_name = f"source{source_path.suffix.lower()}"
        shutil.copy2(source_path, run_dir / source_name)
        (run_dir / "raw.txt").write_text(result.raw_output, encoding="utf-8")

        artifacts = {"source": source_name, "raw": "raw.txt"}
        if result.mode == ExtractionMode.MARKDOWN:
            (run_dir / "result.md").write_text(result.raw_output, encoding="utf-8")
            artifacts["markdown"] = "result.md"
        else:
            value = result.normalized_output
            (run_dir / "result.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "validation.json").write_text(
                json.dumps(
                    {
                        "valid": result.valid,
                        "repaired": result.repaired,
                        "issues": [issue.model_dump() for issue in result.issues],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            export_csv(value, run_dir / "result.csv")
            export_xlsx(value, run_dir / "result.xlsx")
            artifacts.update(
                {
                    "json": "result.json",
                    "validation": "validation.json",
                    "csv": "result.csv",
                    "xlsx": "result.xlsx",
                }
            )
        result.artifacts = artifacts
        (run_dir / "metadata.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def run_directory(self, run_id: str) -> Path:
        if not run_id or any(token in run_id for token in ("/", "\\", "..")):
            raise ValueError("Invalid run id")
        path = (self.settings.runs_dir / run_id).resolve()
        base = self.settings.runs_dir.resolve()
        if path.parent != base or not path.is_dir():
            raise FileNotFoundError(run_id)
        return path
