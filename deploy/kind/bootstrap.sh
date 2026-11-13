#!/usr/bin/env bash
# Create the kind cluster and install its add-ons: an ingress controller, and
# metrics-server, without which the HPA has no CPU numbers to scale on.
# Idempotent. Every version is pinned.
set -euo pipefail

CLUSTER="${KIND_CLUSTER:-credit-scoring}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INGRESS_NGINX_VERSION="controller-v1.15.1"
INGRESS_NGINX_URL="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_NGINX_VERSION}/deploy/static/provider/kind/deploy.yaml"
INGRESS_NGINX_IMAGES=(
  "registry.k8s.io/ingress-nginx/controller:v1.15.1"
  "registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.6.9"
)

METRICS_SERVER_VERSION="v0.9.1"
METRICS_SERVER_URL="https://github.com/kubernetes-sigs/metrics-server/releases/download/${METRICS_SERVER_VERSION}/components.yaml"
METRICS_SERVER_IMAGE="registry.k8s.io/metrics-server/metrics-server:${METRICS_SERVER_VERSION}"

# winget installs kind on the Windows PATH, which a Git Bash shell started
# earlier does not have.
find_kind() {
  if [ -n "${KIND_BIN:-}" ]; then echo "$KIND_BIN"; return; fi
  if command -v kind >/dev/null 2>&1; then command -v kind; return; fi
  local winget="/c/Users/${USERNAME:-$USER}/AppData/Local/Microsoft/WinGet/Packages/Kubernetes.kind_Microsoft.Winget.Source_8wekyb3d8bbwe/kind.exe"
  if [ -x "$winget" ]; then echo "$winget"; return; fi
  echo "kind not found; set KIND_BIN" >&2
  exit 69
}
KIND="$(find_kind)"

# shellcheck source=../kubectl.sh
. "$HERE/../kubectl.sh"

# Avast's Web/Mail Shield re-signs the API server certificate under a root it
# deliberately keeps out of the trust store, so no CA configuration can
# validate it. Scope of this workaround: one kubeconfig context, and only when
# the interception is detected. Turning off Avast's HTTPS scanning (Menu >
# Settings > Protection > Core Shields > Web Shield) makes it a no-op.
relax_tls_for_local_cluster() {
  # From the kubeconfig, because --name makes the address a parameter.
  local issuer server
  server="$(k_value config view \
    -o "jsonpath={.clusters[?(@.name=='kind-${CLUSTER}')].cluster.server}" || true)"
  server="${server#https://}"
  if [ -z "$server" ]; then
    echo "== no kubeconfig entry for kind-${CLUSTER}; skipping the TLS probe"
    return 0
  fi
  issuer="$(echo | openssl s_client -connect "$server" 2>/dev/null \
            | openssl x509 -noout -issuer 2>/dev/null || true)"
  case "$issuer" in
    *"Untrusted Root"*)
      echo "== TLS interception detected: ${issuer#issuer=}"
      echo "   skipping certificate verification for context kind-${CLUSTER} only"
      k_value config set-cluster "kind-${CLUSTER}" --insecure-skip-tls-verify=true >/dev/null
      k_value config unset "clusters.kind-${CLUSTER}.certificate-authority-data" >/dev/null
      ;;
    *)
      echo "== API server certificate verifies normally"
      ;;
  esac
}

echo "== kind $("$KIND" --version | awk '{print $3}')"

if "$KIND" get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "== cluster '$CLUSTER' already exists"
else
  echo "== creating cluster '$CLUSTER'"
  # --name overrides the config's `name:`, so a KIND_CLUSTER override creates
  # the cluster the rest of the run looks for.
  "$KIND" create cluster --name "$CLUSTER" --config "$HERE/cluster.yaml"
fi

relax_tls_for_local_cluster

# The node's containerd cannot verify the intercepted TLS and fails every
# pull with `x509: certificate signed by unknown authority`. The host's Docker
# can, so images are pulled here and side-loaded.
echo "== preloading add-on images"
for image in "${INGRESS_NGINX_IMAGES[@]}" "$METRICS_SERVER_IMAGE"; do
  docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
done
KIND_CLUSTER="$CLUSTER" "$HERE/load-image.sh" \
  "${INGRESS_NGINX_IMAGES[@]}" "$METRICS_SERVER_IMAGE"

# Side-loaded images carry a different manifest digest, so a `tag@sha256:...`
# reference misses them and kubelet falls back to pulling: ErrImagePull on an
# image `crictl images` lists. Stripping the digest leaves the tag pinned.
undigest() {
  sed -E 's/@sha256:[0-9a-f]{64}//g'
}

echo "== installing ingress-nginx ($INGRESS_NGINX_VERSION)"
# A Job's pod template is immutable: re-applying with a changed image fails
# with `field is immutable`. These two carry ttlSecondsAfterFinished: 0.
k_value delete job -n ingress-nginx \
  ingress-nginx-admission-create ingress-nginx-admission-patch \
  --ignore-not-found
curl -sSL "$INGRESS_NGINX_URL" | undigest | k_pipe apply -f -

echo "== installing metrics-server ($METRICS_SERVER_VERSION)"
curl -sSL "$METRICS_SERVER_URL" | undigest | k_pipe apply -f -
# kind's kubelets serve metrics under a self-signed certificate; without this
# flag metrics-server never becomes ready and the HPA reports `unknown`.
# Guarded because `op: add` on `args/-` appends a copy on every run.
if k_value get deployment metrics-server -n kube-system \
     -o jsonpath='{.spec.template.spec.containers[0].args}' \
   | grep -q -- "--kubelet-insecure-tls"; then
  echo "   --kubelet-insecure-tls already set"
else
  k_value patch deployment metrics-server -n kube-system --type=json \
    -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
fi

echo "== waiting for ingress-nginx"
k wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller --timeout=300s

echo "== waiting for metrics-server"
k wait --namespace kube-system \
  --for=condition=available deployment/metrics-server --timeout=300s

echo
echo "== cluster ready"
k_value get nodes
# The Metrics API serves a scrape interval or so after metrics-server reports
# available, so on a fresh cluster this can fail without anything being wrong.
k_value top nodes 2>&1 | head -3 || echo "   node metrics not served yet"
