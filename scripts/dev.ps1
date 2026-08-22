$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

uv sync --extra dev
uv run uvicorn tscan.api.app:app --reload --host 127.0.0.1 --port 8080

