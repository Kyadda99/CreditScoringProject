#!/usr/bin/env bash
# Sample the autoscaler while a load test runs, one line per interval.
#
# Run it beside the load generator:
#
#     deploy/k8s/hpa-watch.sh 120 10 > hpa-trace.txt &
#     uv run --group load locust -f tests/load/locustfile.py \
#         --host http://credit-dev.localtest.me \
#         --headless --users 120 --spawn-rate 10 --run-time 4m
#
# Usage: deploy/k8s/hpa-watch.sh [SAMPLES] [INTERVAL_SECONDS] [OVERLAY]
set -euo pipefail

SAMPLES="${1:-60}"
INTERVAL="${2:-10}"
OVERLAY="${3:-dev}"
NS="credit-$OVERLAY"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../kubectl.sh
. "$REPO_ROOT/deploy/kubectl.sh"
# k_value keeps stderr out of the captured value, where awk would parse it
# into the trace as a measurement.
kubectl_ns() { k_value -n "$NS" "$@"; }

echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) namespace=$NS samples=$SAMPLES interval=${INTERVAL}s"

for _ in $(seq 1 "$SAMPLES"); do
  # One field per call, and `|| true` on each: under `set -euo pipefail` a
  # failed kubectl would otherwise kill the watch mid-run.
  hpa="$(kubectl_ns get hpa api --no-headers | awk '{print $4" replicas="$7}' || true)"
  pods="$(kubectl_ns get pods -l app.kubernetes.io/name=api --no-headers \
          | awk '{print $3}' | sort | uniq -c | tr -s ' \n' ' ' || true)"
  cpu="$(kubectl_ns top pods -l app.kubernetes.io/name=api --no-headers \
         | awk '{printf "%s ", $2}' || true)"
  printf '%s hpa=%s pods=[%s] cpu=[%s]\n' \
    "$(date +%H:%M:%S)" "${hpa:-unavailable}" "${pods:- }" "${cpu:- }"
  sleep "$INTERVAL"
done
