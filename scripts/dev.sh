#!/usr/bin/env sh
set -eu

if [ ! -f .env ]; then
  cp .env.example .env
fi

uv sync --extra dev
uv run uvicorn tscan.api.app:app --reload --host 127.0.0.1 --port 8080

