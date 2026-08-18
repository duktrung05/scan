FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY schemas ./schemas

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 triscan && \
    mkdir -p /app/runs && chown -R triscan:triscan /app

USER triscan
EXPOSE 8080

CMD ["uvicorn", "triscan.api.app:app", "--host", "0.0.0.0", "--port", "8080"]

