#!/usr/bin/env bash
# Smoke-test a running scoring service: /health, then one /score.
#
# Usage: deploy/k8s/smoke.sh BASE_URL [HOST]
#
#   deploy/k8s/smoke.sh http://localhost:8080
#   deploy/k8s/smoke.sh http://credit-dev.localtest.me credit-dev.localtest.me
#
# With HOST, the name is pinned to 127.0.0.1 so the Ingress sees the right
# Host header without depending on DNS. SMOKE_TIMEOUT (seconds, default 120)
# bounds the wait for /health.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 BASE_URL [HOST]" >&2
  exit 64
fi

BASE_URL="${1%/}"
HOST="${2:-}"
TIMEOUT="${SMOKE_TIMEOUT:-120}"

curl_args=(--silent --show-error --max-time 5)
if [ -n "$HOST" ]; then
  curl_args+=(--resolve "$HOST:80:127.0.0.1")
fi

echo "== waiting up to ${TIMEOUT}s for $BASE_URL/health"
deadline=$((SECONDS + TIMEOUT))
body=""
while :; do
  body="$(curl "${curl_args[@]}" "$BASE_URL/health" 2>/dev/null || true)"
  if [ "$body" = '{"status":"ok"}' ]; then
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "FAIL /health did not answer {\"status\":\"ok\"} within ${TIMEOUT}s" >&2
    echo "     last body: ${body:-<none>}" >&2
    exit 1
  fi
  sleep 3
done
echo "   $body"

echo "== POST $BASE_URL/score"
response="$(curl "${curl_args[@]}" --write-out '\n%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5, "AMT_ANNUITY": 24700.5, "DAYS_BIRTH": -9461, "DAYS_EMPLOYED": -637, "CODE_GENDER": "M", "NAME_CONTRACT_TYPE": "Cash loans"}' \
  "$BASE_URL/score" || true)"
status="${response##*$'\n'}"
body="${response%$'\n'*}"
echo "   HTTP $status $body"
if [ "$status" != "200" ] || [[ "$body" != *'"probability"'* ]]; then
  echo "FAIL /score did not return 200 with a probability" >&2
  exit 1
fi

echo "== smoke test passed"
