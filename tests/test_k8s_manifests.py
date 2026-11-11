"""Contract tests for the Kubernetes manifests, rendered without a cluster.

Each overlay is rendered from a copy in a temporary directory with a throwaway
`secret.env`: the real one is gitignored, so rendering in place would skip
every test in CI and in any fresh clone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
K8S_DIR = PROJECT_ROOT / "deploy" / "k8s"

pytestmark = pytest.mark.skipif(
    shutil.which("kubectl") is None,
    reason="kubectl is not installed; manifest rendering is unavailable",
)

# SSL_CERT_FILE is removed for the same reason the deploy scripts remove it:
# on a machine with intercepted TLS it points Go clients at a CA bundle that
# breaks their verification, and kubectl checks for a cluster on start-up.
_ENV = {k: v for k, v in os.environ.items() if k != "SSL_CERT_FILE"}


OVERLAYS = sorted(p.name for p in (K8S_DIR / "overlays").iterdir() if p.is_dir())


@pytest.fixture(scope="module", params=OVERLAYS)
def overlay(request: pytest.FixtureRequest) -> str:
    """Return each overlay's name in turn."""
    return str(request.param)


@pytest.fixture(scope="module")
def rendered(
    overlay: str, tmp_path_factory: pytest.TempPathFactory
) -> list[dict[str, Any]]:
    """Render one overlay from a disposable copy of the manifests."""
    workdir = tmp_path_factory.mktemp(f"k8s-{overlay}") / "k8s"
    shutil.copytree(K8S_DIR, workdir)
    for env_dir in (workdir / "overlays").iterdir():
        (env_dir / "secret.env").write_text(
            "REDIS_PASSWORD=not-a-real-password\n", encoding="utf-8"
        )

    result = subprocess.run(
        ["kubectl", "kustomize", str(workdir / "overlays" / overlay)],
        capture_output=True,
        text=True,
        check=True,
        env=_ENV,
    )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _by_kind(objects: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Return every object of one kind."""
    return [obj for obj in objects if obj["kind"] == kind]


def _one(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """Return the single object of a kind and name."""
    matches = [o for o in _by_kind(objects, kind) if o["metadata"]["name"] == name]
    assert len(matches) == 1, f"expected exactly one {kind}/{name}, got {len(matches)}"
    return matches[0]


def _containers(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a Deployment's containers."""
    return deployment["spec"]["template"]["spec"]["containers"]


def _quantity(value: str) -> float:
    """Parse the subset of Kubernetes quantity suffixes these manifests use."""
    text = str(value)
    if text.endswith("m"):
        return float(text[:-1]) / 1000
    if text.endswith("Mi"):
        return float(text[:-2])
    if text.endswith("Gi"):
        return float(text[:-2]) * 1024
    return float(text)


def test_the_limit_range_lets_an_unannotated_pod_schedule(
    rendered: list[dict[str, Any]],
) -> None:
    """A quota on requests rejects any pod that declares none."""
    quota = _by_kind(rendered, "ResourceQuota")[0]["spec"]["hard"]
    limits = _by_kind(rendered, "LimitRange")[0]["spec"]["limits"][0]

    for resource in ("cpu", "memory"):
        if f"requests.{resource}" in quota:
            assert resource in limits["defaultRequest"], (
                f"quota caps requests.{resource} but the LimitRange supplies "
                f"no default request for it"
            )
        assert _quantity(limits["defaultRequest"][resource]) <= _quantity(
            limits["default"][resource]
        )
        assert _quantity(limits["default"][resource]) <= _quantity(
            limits["max"][resource]
        )


def test_startup_probe_outlasts_a_cold_model_load(
    rendered: list[dict[str, Any]],
) -> None:
    """The startup budget covers a slow load, so liveness cannot kill it."""
    container = _containers(_one(rendered, "Deployment", "api"))[0]
    startup = container["startupProbe"]
    budget = startup["periodSeconds"] * startup["failureThreshold"]
    assert budget >= 60, f"only {budget}s to load the model before a restart"
    # Liveness must be the slower of the two to react, or it becomes the thing
    # that kills a pod the startup probe was still waiting for.
    liveness = container["livenessProbe"]
    assert liveness["periodSeconds"] > startup["periodSeconds"]


def test_api_deployment_leaves_replicas_to_the_autoscaler(
    rendered: list[dict[str, Any]],
) -> None:
    """A Deployment that declares replicas fights its HPA on every apply."""
    assert "replicas" not in _one(rendered, "Deployment", "api")["spec"]


def test_every_container_declares_requests_and_limits(
    rendered: list[dict[str, Any]],
) -> None:
    """Unbounded containers are what a ResourceQuota exists to prevent."""
    for deployment in _by_kind(rendered, "Deployment"):
        for container in _containers(deployment):
            resources = container.get("resources", {})
            for field in ("requests", "limits"):
                assert {"cpu", "memory"} <= set(resources.get(field, {})), (
                    f"{deployment['metadata']['name']}/{container['name']} "
                    f"has no {field}"
                )


def test_quota_covers_a_full_scale_up_mid_rollout(
    rendered: list[dict[str, Any]],
) -> None:
    """The worst case is maxReplicas plus maxSurge, in CPU and in memory."""
    hpa = _one(rendered, "HorizontalPodAutoscaler", "api")
    api = _one(rendered, "Deployment", "api")
    quota = _by_kind(rendered, "ResourceQuota")[0]["spec"]["hard"]

    surge = api["spec"]["strategy"]["rollingUpdate"]["maxSurge"]
    assert isinstance(surge, int), "a percentage maxSurge would need other maths"
    peak_api_pods = hpa["spec"]["maxReplicas"] + surge

    worst: dict[str, float] = {"cpu": 0.0, "memory": 0.0}
    worst_requests: dict[str, float] = {"cpu": 0.0, "memory": 0.0}
    for deployment in _by_kind(rendered, "Deployment"):
        count = (
            peak_api_pods
            if deployment["metadata"]["name"] == "api"
            else deployment["spec"].get("replicas", 1)
        )
        for container in _containers(deployment):
            for resource in ("cpu", "memory"):
                worst[resource] += (
                    _quantity(container["resources"]["limits"][resource]) * count
                )
                worst_requests[resource] += (
                    _quantity(container["resources"]["requests"][resource]) * count
                )

    assert int(quota["pods"]) >= peak_api_pods + 1
    for resource in ("cpu", "memory"):
        key = f"requests.{resource}"
        if key in quota:
            assert _quantity(quota[key]) >= worst_requests[resource], (
                f"quota {key}={quota[key]} below worst case "
                f"{worst_requests[resource]:.2f} ({peak_api_pods} api pods + redis)"
            )
    assert _quantity(quota["limits.cpu"]) >= worst["cpu"], (
        f"quota limits.cpu={quota['limits.cpu']} below worst case "
        f"{worst['cpu']:.2f} cores ({peak_api_pods} api pods + redis)"
    )
    assert _quantity(quota["limits.memory"]) >= worst["memory"], (
        f"quota limits.memory={quota['limits.memory']} below worst case "
        f"{worst['memory']:.0f}Mi ({peak_api_pods} api pods + redis)"
    )


def test_redis_password_comes_from_a_secret_in_both_workloads(
    rendered: list[dict[str, Any]],
) -> None:
    """One generated Secret, found by both workloads through its name hash."""
    secrets = _by_kind(rendered, "Secret")
    assert len(secrets) == 1
    secret_name = secrets[0]["metadata"]["name"]
    assert secret_name != "api-secrets", "expected a content-hash suffix"

    for workload in ("api", "redis"):
        container = _containers(_one(rendered, "Deployment", workload))[0]
        source = next(e for e in container["env"] if e["name"] == "REDIS_PASSWORD")[
            "valueFrom"
        ]["secretKeyRef"]
        assert source == {"name": secret_name, "key": "REDIS_PASSWORD"}


def test_no_image_can_float_or_be_pulled(rendered: list[dict[str, Any]]) -> None:
    """No `:latest`, whose default policy is `Always`, and no `Always`."""
    for deployment in _by_kind(rendered, "Deployment"):
        for container in _containers(deployment):
            image = container["image"]
            assert ":" in image, f"{image} has no tag"
            assert not image.endswith(":latest"), image
            assert container.get("imagePullPolicy") != "Always", image


def test_each_overlay_claims_its_own_ingress_host(
    rendered: list[dict[str, Any]], overlay: str
) -> None:
    """Every overlay replaces the base's placeholder with a host of its own."""
    rule = _one(rendered, "Ingress", "api")["spec"]["rules"][0]
    assert rule["host"] == f"credit-{overlay}.localtest.me"
    backend = rule["http"]["paths"][0]["backend"]["service"]
    assert backend["name"] == "api"


def test_overlays_do_not_share_a_namespace_or_an_image_tag() -> None:
    """Two environments that collide on either are one environment."""
    namespaces: set[str] = set()
    tags: set[str] = set()
    for name in OVERLAYS:
        spec = yaml.safe_load(
            (K8S_DIR / "overlays" / name / "kustomization.yaml").read_text(
                encoding="utf-8"
            )
        )
        namespaces.add(spec["namespace"])
        tags.add(spec["images"][0]["newTag"])
    assert len(namespaces) == len(OVERLAYS), namespaces
    assert len(tags) == len(OVERLAYS), tags


def test_the_generated_secret_is_ignored_and_untracked() -> None:
    """Only git can say whether a credential is committed."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0:
        pytest.skip("not a git work tree")

    for name in OVERLAYS:
        candidate = K8S_DIR / "overlays" / name / "secret.env"
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(candidate)],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        assert ignored.returncode == 0, f"{name}/secret.env is not gitignored"

    tracked = subprocess.run(
        ["git", "ls-files", "deploy/"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    offenders = [
        line for line in tracked.stdout.splitlines() if line.endswith("secret.env")
    ]
    assert not offenders, f"a credential file is tracked: {offenders}"


def test_every_overlay_ships_an_example_secret_file() -> None:
    """`secret.env.example` is the only guide a fresh clone gets."""
    for name in OVERLAYS:
        example = K8S_DIR / "overlays" / name / "secret.env.example"
        assert example.exists(), f"{name} has no secret.env.example"
        assert "REDIS_PASSWORD=" in example.read_text(encoding="utf-8")
