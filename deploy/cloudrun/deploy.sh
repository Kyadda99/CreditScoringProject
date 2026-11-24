#!/usr/bin/env bash
# Deploy an image to Cloud Run with the model mounted from Cloud Storage.
#
# Usage: deploy/cloudrun/deploy.sh IMAGE TAG
#
# Required env: CS_GCP_REGION, CS_RUN_SERVICE, CS_RUNTIME_SA, CS_MODEL_BUCKET.
# CS_MODEL_VERSION_FILE overrides deploy/model-version.
#
# An existing service gets a new revision with no traffic, reachable only at
# its tag URL. A missing service is created public and takes the traffic.
# Stdout carries exactly two lines, revision=NAME and url=TAG_URL, so it can
# be appended to $GITHUB_OUTPUT; everything else goes to stderr.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "usage: $0 IMAGE TAG" >&2
  echo "  env: CS_GCP_REGION CS_RUN_SERVICE CS_RUNTIME_SA CS_MODEL_BUCKET" >&2
  exit 64
}

[ "$#" -eq 2 ] || usage
IMAGE="$1"
TAG="$2"
for name in CS_GCP_REGION CS_RUN_SERVICE CS_RUNTIME_SA CS_MODEL_BUCKET; do
  if [ -z "${!name:-}" ]; then
    echo "missing env: $name" >&2
    usage
  fi
done

VERSION_FILE="${CS_MODEL_VERSION_FILE:-$HERE/../model-version}"
VERSION="$(tr -d '[:space:]' < "$VERSION_FILE")"
if ! [[ "$VERSION" =~ ^[1-9][0-9]*$ ]]; then
  echo "model version in $VERSION_FILE is not a registry version: '$VERSION'" >&2
  exit 1
fi

region=(--region "$CS_GCP_REGION")

if describe_err="$(gcloud run services describe "$CS_RUN_SERVICE" "${region[@]}" \
    --format='value(metadata.name)' 2>&1 >/dev/null)"; then
  mode=(--no-traffic)
elif grep -qE 'NOT_FOUND|[Cc]annot find service|could not be found' <<<"$describe_err"; then
  echo "service $CS_RUN_SERVICE not found; creating it" >&2
  mode=(--allow-unauthenticated)
else
  echo "$describe_err" >&2
  exit 1
fi

gcloud run deploy "$CS_RUN_SERVICE" "${region[@]}" \
  --image "$IMAGE" \
  --tag "$TAG" \
  "${mode[@]}" \
  --service-account "$CS_RUNTIME_SA" \
  --execution-environment gen2 \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "CS_MODEL_PATH=/models/$VERSION/pipeline.joblib" \
  --add-volume "name=models,type=cloud-storage,bucket=$CS_MODEL_BUCKET,readonly=true" \
  --add-volume-mount "volume=models,mount-path=/models" \
  --quiet >&2

# Revision and URL come from the same traffic entry, so a deploy landing in
# between cannot swap in a revision that was never smoke-tested. `describe`
# has no --filter, so the tag is matched here.
entry="$(gcloud run services describe "$CS_RUN_SERVICE" "${region[@]}" \
  --flatten=status.traffic \
  --format='value(status.traffic.tag,status.traffic.revisionName,status.traffic.url)' \
  | awk -F'\t' -v tag="$TAG" '$1 == tag { print $2 "\t" $3; exit }')"
revision="${entry%%$'\t'*}"
url="${entry#*$'\t'}"
if [ -z "$entry" ] || [ -z "$revision" ] || [ -z "$url" ]; then
  echo "could not read the revision or the tag URL for $TAG" >&2
  exit 1
fi
echo "revision=$revision"
echo "url=$url"
