#!/usr/bin/env bash
# Load local Docker images into the kind cluster's containerd.
#
# Not `kind load docker-image`: with Docker Desktop's containerd image store,
# `docker save` writes an index listing only the local platform, and kind's
# `ctr images import --all-platforms` fails with
# `ctr: content digest sha256:...: not found`. Importing without that flag
# works, but does not preserve the manifest digest -- hence the digest
# stripping in bootstrap.sh.
#
# Usage: deploy/kind/load-image.sh IMAGE [IMAGE...]
set -euo pipefail

CLUSTER="${KIND_CLUSTER:-credit-scoring}"
NODE="${CLUSTER}-control-plane"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 IMAGE [IMAGE...]" >&2
  exit 64
fi

for image in "$@"; do
  echo ">> $image -> $NODE"
  docker save "$image" \
    | docker exec -i "$NODE" ctr --namespace=k8s.io images import -
done
