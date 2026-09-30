# Build stage: uv resolves dependencies from uv.lock into a venv.
FROM python:3.12-slim AS builder

# Pinned version, not `latest`: this tool materialises the dependency graph,
# so a floating tag would defeat `--locked` three lines below.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# --locked: fails when uv.lock has drifted from pyproject.toml, the same
# guarantee CI enforces.
# --no-default-groups: no dev, no training (mlflow, kagglehub); the image
# serves the model, it does not train it.
# --no-editable: uv installs the project as editable by default, pointing at
# /build/src, which the runtime stage does not have: "No module named api".
RUN uv sync --locked --no-default-groups --no-editable

# Runtime stage: the venv and the code only. No uv; it cost 55 MB.
FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# The artifact is gitignored, so it exists on disk but not in the repository.
# This image therefore builds only locally, after training.
COPY models/pipeline.joblib ./models/pipeline.joblib

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

ENV CS_MODEL_PATH=/app/models/pipeline.joblib

# `exec` is critical. Without it uvicorn is a child of /bin/sh, SIGTERM hits
# the shell, and the container dies from SIGKILL after the full timeout:
# 10.9 s and exit 137, against 0.67 s and exit 0.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
