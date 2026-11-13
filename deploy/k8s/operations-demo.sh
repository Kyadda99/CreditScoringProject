#!/usr/bin/env bash
# Day-two operations on a running deployment: rolling update, rollback, manual
# scale against a live HPA, a deliberately slow start-up, and pod diagnosis.
#
# Usage: deploy/k8s/operations-demo.sh [OVERLAY]      (default: dev)
set -euo pipefail

OVERLAY="${1:-dev}"
NS="credit-$OVERLAY"
CLUSTER="${KIND_CLUSTER:-credit-scoring}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

IMAGE="credit-scoring/api:$OVERLAY"
REVISED_IMAGE="credit-scoring/api:$OVERLAY-rev2"

# shellcheck source=../kubectl.sh
. "$REPO_ROOT/deploy/kubectl.sh"
# Short calls retry; the rollout waits stream. See deploy/kubectl.sh.
kubectl_ns() { k_value -n "$NS" "$@"; }
kubectl_stream() { k -n "$NS" "$@"; }

step() { printf '\n=========== %s\n' "$*"; }

step "starting state"
kubectl_ns get deploy,hpa,pods

# ---------------------------------------------------------------------------
step "1. rolling update to $REVISED_IMAGE"
# A Deployment rolls on a pod-template change, so a new tag on the same image
# content is a real rolling update.
docker tag "$IMAGE" "$REVISED_IMAGE"
KIND_CLUSTER="$CLUSTER" "$REPO_ROOT/deploy/kind/load-image.sh" "$REVISED_IMAGE" >/dev/null
kubectl_ns set image deployment/api "api=$REVISED_IMAGE"
# maxUnavailable: 0: the old pod serves until the new one passes readiness.
kubectl_stream rollout status deployment/api --timeout=300s
kubectl_ns get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}'

step "2. rollout history"
kubectl_ns rollout history deployment/api

# ---------------------------------------------------------------------------
step "3. deliberate rollback"
kubectl_ns rollout undo deployment/api
kubectl_stream rollout status deployment/api --timeout=300s
echo "-- image after undo (expect :$OVERLAY):"
kubectl_ns get deployment api -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'

# ---------------------------------------------------------------------------
step "4. manual scale against a live HPA"
kubectl_ns scale deployment/api --replicas=3
sleep 10
echo "-- replicas right after scale:"
kubectl_ns get deployment api -o jsonpath='{.status.replicas}{"\n"}'
# The HPA owns this field: a manual scale holds only until its next
# evaluation. Holding a count by hand needs the HPA deleted or suspended.
echo "-- HPA (it will take this field back):"
kubectl_ns get hpa api

# ---------------------------------------------------------------------------
step "5. deliberate slow start -- does the startup probe tolerate it?"
# The sleep keeps the port closed for 45s: longer than the liveness probe's
# budget, inside the startup probe's.
#
# Without the trap, a failure before the undo below leaves the patch applied,
# and with it uvicorn as a child of `sh` -- PID 1 swallowing SIGTERM.
undo_slow_start() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    echo "!! aborting with the slow-start patch applied; reverting it" >&2
    kubectl_ns rollout undo deployment/api >&2 || true
  fi
  return "$status"
}
trap undo_slow_start EXIT INT TERM

kubectl_ns patch deployment/api --type=json -p '[{
  "op": "add",
  "path": "/spec/template/spec/containers/0/command",
  "value": ["sh", "-c", "sleep 45; exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
}]'
kubectl_stream rollout status deployment/api --timeout=300s
echo "-- restart count for the slow pod (expect 0: the probe waited):"
# custom-columns, not jsonpath: a pod whose container has not started has no
# `status.containerStatuses`, and jsonpath errors on the missing field.
kubectl_ns get pods -l app.kubernetes.io/name=api \
  -o custom-columns='NAME:.metadata.name,RESTARTS:.status.containerStatuses[0].restartCount,PHASE:.status.phase'

step "6. undo the slow start"
kubectl_ns rollout undo deployment/api
kubectl_stream rollout status deployment/api --timeout=300s
# The patch is gone; nothing left to revert.
trap - EXIT INT TERM

# ---------------------------------------------------------------------------
step "7. diagnosing a pod: logs, describe, events"
# --field-selector: items[0] during a rollout is as likely to be a
# terminating pod, whose logs show a shutdown rather than traffic.
POD="$(kubectl_ns get pods -l app.kubernetes.io/name=api \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')"
if [ -z "$POD" ]; then
  echo "no Running api pod to diagnose" >&2
  exit 1
fi
echo "-- logs from $POD (JSON request events, interleaved with uvicorn's own"
echo "   plain-text access lines -- both go to stdout):"
kubectl_ns logs "$POD" --tail=5
echo
echo "-- describe, the probe and resource lines:"
# -A2: `Limits:` and `Requests:` are headers; the values are below them.
kubectl_ns describe pod "$POD" | grep -E -A2 "Image:|Liveness:|Readiness:|Startup:|Limits:|Requests:" || true
echo
echo "-- events, most recent last:"
kubectl_ns get events --sort-by=.lastTimestamp | tail -12

step "final state"
kubectl_ns get deploy,hpa,pods
