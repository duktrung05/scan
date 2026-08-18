from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from triscan.config import get_settings
from triscan.evaluation import evaluate_manifest
from triscan.exceptions import TriScanError
from triscan.pipeline import ExtractionPipeline
from triscan.schema_registry import SchemaRegistry
from triscan.types import DocumentLanguage, ExtractionMode
from triscan.validation import validate_and_normalize

app = typer.Typer(
    name="triscan",
    help="Multilingual document understanding for Markdown and schema-guided JSON.",
    no_args_is_help=True,
)


def _pipeline() -> ExtractionPipeline:
    return ExtractionPipeline(settings=get_settings())


@app.command()
def extract(
    document: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    mode: Annotated[ExtractionMode, typer.Option()] = ExtractionMode.JSON,
    document_type: Annotated[str, typer.Option("--type")] = "invoice",
    language: Annotated[DocumentLanguage, typer.Option()] = DocumentLanguage.AUTO,
    schema_path: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Extract one PDF or image and persist a complete run directory."""
    try:
        custom_schema = None
        if schema_path:
            custom_schema = SchemaRegistry.parse_custom(schema_path.read_text(encoding="utf-8"))
        result = _pipeline().extract(
            document,
            mode=mode,
            document_type=document_type,
            language=language,
            custom_schema=custom_schema,
        )
    except (TriScanError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def batch(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    pattern: Annotated[str, typer.Option()] = "*",
    mode: Annotated[ExtractionMode, typer.Option()] = ExtractionMode.JSON,
    document_type: Annotated[str, typer.Option("--type")] = "invoice",
    language: Annotated[DocumentLanguage, typer.Option()] = DocumentLanguage.AUTO,
) -> None:
    """Process a local directory sequentially."""
    pipeline = _pipeline()
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    files = sorted(path for path in directory.glob(pattern) if path.suffix.lower() in supported)
    if not files:
        typer.echo("No supported documents found", err=True)
        raise typer.Exit(1)
    failed = 0
    for path in files:
        try:
            result = pipeline.extract(
                path,
                mode=mode,
                document_type=document_type,
                language=language,
            )
            typer.echo(f"OK  {path.name} -> {result.run_id}")
        except (TriScanError, ValueError) as exc:
            failed += 1
            typer.echo(f"ERR {path.name}: {exc}", err=True)
    if failed:
        raise typer.Exit(1)


@app.command("validate-json")
def validate_json(
    model_output: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    schema: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Parse, normalize and validate a saved model response."""
    schema_value = SchemaRegistry.parse_custom(schema.read_text(encoding="utf-8"))
    result = validate_and_normalize(model_output.read_text(encoding="utf-8"), schema_value)
    typer.echo(
        json.dumps(
            {
                "valid": result.valid,
                "repaired": result.repaired,
                "repair_actions": result.repair_actions,
                "normalized": result.normalized,
                "issues": [issue.model_dump() for issue in result.issues],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not result.valid:
        raise typer.Exit(2)


@app.command()
def evaluate(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Evaluate JSON or Markdown predictions described by a JSONL manifest."""
    try:
        summary = evaluate_manifest(manifest)
    except TriScanError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    text = summary.model_dump_json(indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
    typer.echo(text)


@app.command()
def schemas() -> None:
    """List available built-in and custom schemas."""
    registry = SchemaRegistry(get_settings().schemas_dir)
    for name in registry.list():
        typer.echo(name)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8080,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    """Start the API and browser workspace."""
    import uvicorn

    uvicorn.run("triscan.api.app:app", host=host, port=port, reload=reload)
