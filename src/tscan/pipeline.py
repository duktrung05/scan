from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tscan.config import Settings, get_settings
from tscan.exporters import export_csv, export_xlsx
from tscan.ingestion import load_document
from tscan.models import create_model
from tscan.models.base import ModelRequest, VisionModel
from tscan.prompts import PROMPT_VERSION, build_prompt
from tscan.schema_registry import SchemaRegistry
from tscan.types import (
    DocumentLanguage,
    ExtractionMode,
    ExtractionResult,
    InferenceMetadata,
)
from tscan.validation import validate_and_normalize


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
            raw_response=response.text,
            inference=InferenceMetadata(
                provider=response.provider,
                model=response.model,
                latency_ms=latency_ms,
                prompt_version=PROMPT_VERSION,
                page_count=len(document.pages),
                extra={
                    **response.metadata,
                    "model_revision": response.metadata.get(
                        "model_revision", self.settings.model_revision
                    ),
                    "schema_sha256": hashlib.sha256(
                        json.dumps(schema, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest()
                    if schema is not None
                    else None,
                },
            ),
        )

        if mode == ExtractionMode.JSON:
            validation = validate_and_normalize(
                response.text,
                schema or {},
                language=language.value,
                page_count=len(document.pages),
            )
            result.parsed_output = validation.parsed
            result.raw_value = validation.parsed
            result.normalized_output = validation.normalized
            result.normalized_value = validation.normalized
            result.syntax_valid = validation.syntax_valid
            result.raw_schema_valid = validation.raw_schema_valid
            result.schema_valid = validation.schema_valid
            result.valid = validation.valid
            result.repaired = validation.repaired
            result.normalization_actions = validation.repair_actions
            result.raw_schema_issues = validation.raw_schema_issues
            result.issues = validation.issues
            result.warnings = [issue for issue in validation.issues if issue.severity == "warning"]
            result.inference.extra["repair_actions"] = validation.repair_actions
        else:
            result.valid = None

        self._persist(result, document.source_path, document.preprocessing_log)
        return result

    def _persist(
        self,
        result: ExtractionResult,
        source_path: Path,
        preprocessing_log: list[dict[str, Any]],
    ) -> None:
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
            value = result.normalized_value
            (run_dir / "raw.json").write_text(
                json.dumps(result.raw_value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "normalized.json").write_text(
                json.dumps(result.normalized_value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "result.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "validation.json").write_text(
                json.dumps(
                    {
                        "valid": result.valid,
                        "syntax_valid": result.syntax_valid,
                        "raw_schema_valid": result.raw_schema_valid,
                        "schema_valid": result.schema_valid,
                        "repaired": result.repaired,
                        "normalization_actions": result.normalization_actions,
                        "raw_schema_issues": [
                            issue.model_dump() for issue in result.raw_schema_issues
                        ],
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
                    "raw_json": "raw.json",
                    "normalized_json": "normalized.json",
                    "json": "result.json",
                    "validation": "validation.json",
                    "csv": "result.csv",
                    "xlsx": "result.xlsx",
                }
            )
            (run_dir / "audit.json").write_text(
                json.dumps(
                    {
                        "document_sha256": result.document_sha256,
                        "preprocessing": preprocessing_log,
                        "model": result.inference.model,
                        "provider": result.inference.provider,
                        "prompt_version": result.inference.prompt_version,
                        "raw_response_saved": True,
                        "syntax_valid": result.syntax_valid,
                        "raw_schema_valid": result.raw_schema_valid,
                        "schema_valid": result.schema_valid,
                        "raw_schema_issues": [
                            issue.model_dump() for issue in result.raw_schema_issues
                        ],
                        "normalization_actions": result.normalization_actions,
                        "warnings": [warning.model_dump() for warning in result.warnings],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            artifacts["audit"] = "audit.json"
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
