.PHONY: install dev test lint run clean

install:
	uv sync --extra dev

dev:
	uv run uvicorn tscan.api.app:app --reload --host 127.0.0.1 --port 8080

test:
	uv run pytest

lint:
	uv run ruff check .

run:
	uv run tscan serve

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov dist build

