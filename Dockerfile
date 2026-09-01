# Etap budowania: uv rozwiązuje zależności z uv.lock do venv-a.
FROM python:3.12-slim AS builder

# Przypięta wersja, nie `latest`: to narzędzie materializuje graf zależności,
# więc pływający tag przeczyłby całemu sensowi `--locked` trzy linijki niżej.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# --locked: przerywa, gdy uv.lock rozjechał się z pyproject.toml — ta sama
# gwarancja, którą wymusza CI.
# --no-default-groups: bez dev i bez training (mlflow, kagglehub) — obraz
# serwuje model, nie trenuje go.
# --no-editable: uv domyślnie instaluje projekt jako editable ze wskazaniem na
# /build/src, którego w etapie runtime nie ma — stąd "No module named api".
RUN uv sync --locked --no-default-groups --no-editable

# Etap runtime: sam venv i kod. Bez uv — w spike'u kosztowało 55 MB.
FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Artefakt jest gitignorowany, więc istnieje na dysku, ale nie w repozytorium.
# Konsekwencja: ten obraz buduje się tylko lokalnie, po treningu. Faza 6
# rozwiązuje to na poziomie CI.
COPY models/pipeline.joblib ./models/pipeline.joblib

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

ENV CS_MODEL_PATH=/app/models/pipeline.joblib

# `exec` jest tu krytyczne. Bez niego uvicorn jest dzieckiem /bin/sh, SIGTERM
# trafia w powłokę i kontener ginie od SIGKILL po pełnym timeoucie (zmierzone
# w spike'u: 10.9 s, exit 137 — zamiast 0.67 s i exit 0).
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
