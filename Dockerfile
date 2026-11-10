# Build stage: uv resolves dependencies from uv.lock into a venv.
FROM python:3.12-slim AS builder

# Pinned version, not `latest`: this tool materialises the dependency graph,
# so a floating tag would defeat `--locked` three lines below.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

# Empty by default, so every build verifies certificates. Set it only on a
# machine whose TLS is intercepted by a proxy uv's rustls rejects:
#   docker build --build-arg UV_INSECURE_HOST="pypi.org files.pythonhosted.org"
ARG UV_INSECURE_HOST=""

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INSECURE_HOST=${UV_INSECURE_HOST}

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# --locked: fails when uv.lock has drifted from pyproject.toml, the same
# guarantee CI enforces.
# --no-default-groups: no dev, no training (mlflow, kagglehub); the image
# serves the model, it does not train it.
# --no-editable: uv installs the project as editable by default, pointing at
# /build/src, which the runtime stage does not have: "No module named api".
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

# Runtime stage: the venv and the code only. No uv; it cost 55 MB.
FROM python:3.12-slim AS serve

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# The artifact is not in the image. Training writes it to a shared volume and
# the service reads it from there, so the same image runs anywhere.
ENV CS_MODEL_PATH=/models/pipeline.joblib

# `exec` is critical. Without it uvicorn is a child of /bin/sh, SIGTERM hits
# the shell, and the container dies from SIGKILL after the full timeout:
# 10.9 s and exit 137, against 0.67 s and exit 0.
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]

# Training image: the groups the serving image omits. mlflow and kagglehub
# took the serving image from ~520 MB to 1.28 GB, which is why they are split.
FROM python:3.12-slim AS trainer-builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ARG UV_INSECURE_HOST=""

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INSECURE_HOST=${UV_INSECURE_HOST}

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group training --no-editable

FROM python:3.12-slim AS trainer

COPY --from=trainer-builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CS_MODEL_PATH=/models/pipeline.joblib \
    CS_DATA_PATH=/data/raw/application_train.csv

WORKDIR /app

RUN useradd --create-home --uid 1000 app

CMD ["cs-train"]
