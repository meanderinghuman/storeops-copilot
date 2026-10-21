# One image for the Python services (gateway + web UI, forecast stub). The model runs natively on the
# host (llama-server with Metal); containers reach it at host.docker.internal:8080.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY store ./store
COPY datagen ./datagen
COPY serve ./serve
