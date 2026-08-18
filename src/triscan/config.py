from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRISCAN_",
        extra="ignore",
        case_sensitive=False,
    )

    model_provider: Literal["mock", "openai_compatible", "transformers"] = "mock"
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    api_base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"

    max_new_tokens: int = Field(default=4096, ge=64, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout_seconds: float = Field(default=180.0, ge=1.0, le=1800.0)

    max_file_mb: int = Field(default=25, ge=1, le=500)
    max_pdf_pages: int = Field(default=20, ge=1, le=500)
    pdf_dpi: int = Field(default=160, ge=72, le=400)
    max_image_megapixels: int = Field(default=16, ge=1, le=100)
    max_batch_files: int = Field(default=10, ge=1, le=100)

    runs_dir: Path = Path("runs")
    schemas_dir: Path = Path("schemas")
    log_level: str = "INFO"

    @field_validator("runs_dir", "schemas_dir", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @field_validator("api_base_url")
    @classmethod
    def trim_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
