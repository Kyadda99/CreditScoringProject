r"""Load profile for the scoring service running on the cluster.

Run against the dev ingress:

    uv run --group load locust -f tests/load/locustfile.py \
        --host http://credit-dev.localtest.me

Headless, the steady run, writing the CSVs that get reported:

    uv run --group load locust -f tests/load/locustfile.py \
        --host http://credit-dev.localtest.me \
        --headless --users 20 --spawn-rate 2 --run-time 3m \
        --csv load-steady

At the default repeat rate that run saturates the container and drives the HPA
to its ceiling, so there is no separate autoscaling scenario. Pair it with
`deploy/k8s/hpa-watch.sh` to record what the autoscaler did.

Service, cluster and load generator are all on one laptop, so the absolute RPS
measures the laptop.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

# Every feature is optional; the preprocessor imputes the rest. The cache key
# is a hash of the payload, so varying AMT_CREDIT is what makes a miss.
_BASE_APPLICATION: dict[str, float | str] = {
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "CODE_GENDER": "M",
    "NAME_CONTRACT_TYPE": "Cash loans",
}

# The cache hit rate is a parameter of the measurement: against a 300 s TTL,
# a small pool of payloads warms the cache within seconds and the run then
# reports the cache's latency rather than the model's. REPEAT_RATE of requests
# reuse the warm pool; the rest should miss. 0.0 characterises the uncached
# scorer, 1.0 the cache.
REPEAT_RATE = float(os.environ.get("CS_LOAD_REPEAT_RATE", "0.3"))

# Small enough to stay warm inside the TTL.
_WARM_POOL = [float(100_000 + 25_000 * i) for i in range(8)]

# Wide enough that a collision inside one run is unlikely.
_COLD_RANGE = (50_000_00, 950_000_00)


class ScoringUser(HttpUser):
    """A client scoring applications one at a time."""

    wait_time = between(0.1, 0.5)

    def _application(self) -> dict[str, float | str]:
        """Return one application, either a cache-warm repeat or a fresh miss."""
        if random.random() < REPEAT_RATE:
            amount = random.choice(_WARM_POOL)
        else:
            amount = random.randint(*_COLD_RANGE) / 100.0
        return {**_BASE_APPLICATION, "AMT_CREDIT": amount}

    @task(10)
    def score(self) -> None:
        """Score a single application."""
        self.client.post("/score", json=self._application())

    @task(2)
    def score_batch(self) -> None:
        """Score a batch, the endpoint that fans out across the thread pool."""
        applications = [self._application() for _ in range(10)]
        self.client.post("/score/batch", json={"applications": applications})

    @task(1)
    def health(self) -> None:
        """Hit the endpoint all three probes use, under load."""
        self.client.get("/health")
