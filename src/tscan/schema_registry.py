from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tscan.exceptions import ConfigurationError


class SchemaRegistry:
    def __init__(self, custom_directory: Path | None = None) -> None:
        self.custom_directory = custom_directory

    def _builtin_path(self, name: str):
        return files("tscan.builtin_schemas").joinpath(f"{name}.schema.json")

    def list(self) -> list[str]:
        names = {
            item.name.removesuffix(".schema.json")
            for item in files("tscan.builtin_schemas").iterdir()
            if item.name.endswith(".schema.json")
        }
        if self.custom_directory and self.custom_directory.is_dir():
            names.update(
                path.name.removesuffix(".schema.json")
                for path in self.custom_directory.glob("*.schema.json")
            )
        return sorted(names)

    def load(self, name: str) -> dict[str, Any]:
        safe_name = name.strip().lower().replace(" ", "_")
        if not safe_name or any(part in safe_name for part in ("/", "\\", "..")):
            raise ConfigurationError("Invalid schema name")
        custom_path = (
            self.custom_directory / f"{safe_name}.schema.json" if self.custom_directory else None
        )
        try:
            if custom_path and custom_path.is_file():
                schema = json.loads(custom_path.read_text(encoding="utf-8"))
            else:
                schema = json.loads(self._builtin_path(safe_name).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError(f"Unknown schema: {safe_name}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON schema file: {safe_name}") from exc
        Draft202012Validator.check_schema(schema)
        return schema

    @staticmethod
    def parse_custom(schema_text: str) -> dict[str, Any]:
        try:
            schema = json.loads(schema_text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Custom schema is not valid JSON: {exc.msg}") from exc
        if not isinstance(schema, dict):
            raise ConfigurationError("Custom schema must be a JSON object")
        Draft202012Validator.check_schema(schema)
        return schema
