#!/usr/bin/env bash
# Build, side-load and apply the service to the kind cluster.
#
# Assumes deploy/kind/bootstrap.sh has already created the cluster and
# installed the ingress controller and metrics-server.
#
# Usage: deploy/k8s/deploy.sh [OVERLAY]   (OVERLAY defaults to dev)
set -euo pipefail

OVERLAY="${1:-dev}"
CLUSTER="${KIND_CLUSTER:-credit-scoring}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY_DIR="$REPO_ROOT/deploy/k8s/overlays/$OVERLAY"
NAMESPACE="credit-$OVERLAY"

IMAGE="credit-scoring/api:$OVERLAY"
REDIS_IMAGE="redis:7-alpine"

if [ ! -d "$OVERLAY_DIR" ]; then
  echo "no such overlay: $OVERLAY_DIR" >&2
  exit 64
fi

# shellcheck source=../kubectl.sh
. "$REPO_ROOT/deploy/kubectl.sh"

# Generated once: a redeploy must not roll the password out from under a
# running Redis.
SECRET_ENV="$OVERLAY_DIR/secret.env"
if [ ! -f "$SECRET_ENV" ]; then
  echo "== generating $SECRET_ENV"
  printf 'REDIS_PASSWORD=%s\n' "$(openssl rand -hex 24)" > "$SECRET_ENV"
fi

echo "== building $IMAGE"
# uv inside the build cannot verify the intercepted TLS without the machine's
# CA bundle. Same convention as `CS_CA_BUNDLE` in docker-compose.yml.
CA_BUNDLE="${CS_CA_BUNDLE:-${SSL_CERT_FILE:-$HOME/.ssl/cacert.pem}}"
build_args=(--target serve -t "$IMAGE")
if [ -s "$CA_BUNDLE" ]; then
  echo "   using CA bundle $CA_BUNDLE"
  build_args+=(--secret "id=ca-bundle,src=$CA_BUNDLE")
fi
docker build "${build_args[@]}" "$REPO_ROOT"

echo "== side-loading images"
docker image inspect "$REDIS_IMAGE" >/dev/null 2>&1 || docker pull "$REDIS_IMAGE"
KIND_CLUSTER="$CLUSTER" "$REPO_ROOT/deploy/kind/load-image.sh" \
  "$IMAGE" "$REDIS_IMAGE"

echo "== syncing model artifact"
KIND_CLUSTER="$CLUSTER" "$REPO_ROOT/deploy/kind/sync-artifact.sh"

echo "== applying overlay '$OVERLAY'"
k_value apply -k "$OVERLAY_DIR"

# A rebuild of changed source renders a byte-identical pod template -- fixed
# tag, static ConfigMap, unchanged Secret hash -- so `apply` reports
# "unchanged" and the old pods keep serving the old code, while the
# `rollout status` below returns success at once and certifies it.
echo "== restarting api to pick up the rebuilt image"
k_value -n "$NAMESPACE" rollout restart deployment/api

# A stalled rollout is the first symptom of a bad probe, a missing image or
# an exhausted quota.
echo "== waiting for rollout"
k -n "$NAMESPACE" rollout status deployment/redis --timeout=180s
k -n "$NAMESPACE" rollout status deployment/api --timeout=300s

echo
k_value -n "$NAMESPACE" get deploy,svc,ingress,hpa
echo
echo "== try it"
echo "   curl -s http://credit-$OVERLAY.localtest.me/health"
