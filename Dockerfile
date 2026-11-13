# Build stage: uv resolves dependencies from uv.lock into a venv.
FROM python:3.12-slim AS builder

# Pinned version, not `latest`: this tool materialises the dependency graph,
# so a floating tag would defeat `--locked` three lines below.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

# On a machine whose TLS is intercepted, container traffic is re-signed too and
# the base image does not trust the proxy's root. uv's rustls then fails with
# `invalid peer certificate: UnknownIssuer`. The fix that KEEPS verification on
# is to hand the build a CA bundle carrying that root, as a secret so it never
# lands in a layer:
#   docker build --secret id=ca-bundle,src=$HOME/.ssl/cacert.pem .
# Without the secret the mount is empty, the guard is a no-op, and the build
# verifies against the base image's own store -- which is what a machine with
# unintercepted TLS wants, CI included.
#
# UV_INSECURE_HOST stays as a last resort for a proxy whose root cannot be
# obtained. It switches verification off for those hosts, so prefer the secret.
ARG UV_INSECURE_HOST=""

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INSECURE_HOST=${UV_INSECURE_HOST}

WORKDIR /build

# Dependencies first, from bind-mounted pyproject.toml and uv.lock only: a
# change to the source or the README leaves this layer cached.
# --locked: fails when uv.lock has drifted from pyproject.toml, the same
# guarantee CI enforces.
# --no-default-groups: no dev, no training (mlflow, kagglehub); the image
# serves the model, it does not train it.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=secret,id=ca-bundle,target=/run/secrets/ca-bundle.pem \
    if [ -s /run/secrets/ca-bundle.pem ]; then \
        export SSL_CERT_FILE=/run/secrets/ca-bundle.pem; \
    fi; \
    uv sync --locked --no-default-groups --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# --no-editable: uv installs the project as editable by default, pointing at
# /build/src, which the runtime stage does not have: "No module named api".
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=ca-bundle,target=/run/secrets/ca-bundle.pem \
    if [ -s /run/secrets/ca-bundle.pem ]; then \
        export SSL_CERT_FILE=/run/secrets/ca-bundle.pem; \
    fi; \
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

# Same TLS story as the serving builder; see the comment on that stage.
ARG UV_INSECURE_HOST=""

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_INSECURE_HOST=${UV_INSECURE_HOST}

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=secret,id=ca-bundle,target=/run/secrets/ca-bundle.pem \
    if [ -s /run/secrets/ca-bundle.pem ]; then \
        export SSL_CERT_FILE=/run/secrets/ca-bundle.pem; \
    fi; \
    uv sync --locked --no-default-groups --group training --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=ca-bundle,target=/run/secrets/ca-bundle.pem \
    if [ -s /run/secrets/ca-bundle.pem ]; then \
        export SSL_CERT_FILE=/run/secrets/ca-bundle.pem; \
    fi; \
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
