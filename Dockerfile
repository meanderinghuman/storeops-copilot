# One image for the Python services (gateway, forecast stub, UI). The model runs natively on the
# host (llama-server with Metal); containers reach it at host.docker.internal:8080.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --group ui --no-install-project
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

COPY store ./store
COPY datagen ./datagen
COPY serve ./serve
