#!/usr/bin/env bash
# Put the trained model artifact on the kind node, where the api pod's
# hostPath volume expects to find it.
#
# The cluster's equivalent of the compose stack's `model-artifact` volume:
# the image stays model-free.
#
# Usage: deploy/kind/sync-artifact.sh [ARTIFACT_PATH]
set -euo pipefail

CLUSTER="${KIND_CLUSTER:-credit-scoring}"
NODE="${CLUSTER}-control-plane"
NODE_DIR="/opt/credit-scoring/models"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT="${1:-$REPO_ROOT/models/pipeline.joblib}"

if [ ! -f "$ARTIFACT" ]; then
  echo "artifact not found: $ARTIFACT" >&2
  echo "run the training flow first, or pass the path as an argument" >&2
  exit 66
fi

echo ">> $ARTIFACT -> $NODE:$NODE_DIR/pipeline.joblib"

# Git Bash rewrites container-side paths: without this, `docker exec mkdir`
# creates `C:/Program Files/Git/opt/...` inside the node and the `docker cp`
# fails with "Could not find the file /opt/credit-scoring/models".
export MSYS_NO_PATHCONV=1

# With conversion off, the host path needs converting by hand: Docker for
# Windows cannot resolve Git Bash's `/h/AI kurs` form.
if command -v cygpath >/dev/null 2>&1; then
  ARTIFACT_HOST="$(cygpath -w "$ARTIFACT")"
else
  ARTIFACT_HOST="$ARTIFACT"
fi

docker exec "$NODE" mkdir -p "$NODE_DIR"
docker cp "$ARTIFACT_HOST" "$NODE:$NODE_DIR/pipeline.joblib"
# The file arrives owned by root; the api container runs as uid 1000.
docker exec "$NODE" chmod 0644 "$NODE_DIR/pipeline.joblib"
docker exec "$NODE" ls -l "$NODE_DIR"
