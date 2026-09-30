# Credit Scoring

End-to-end credit default risk scoring: data to model to HTTP service to
container to cloud. Dataset: [Home Credit Default
Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(`application_train.csv`).

Working documentation is kept locally and deliberately untracked, so the
implementation notes and conventions stay out of the repository.

## Quick start

```bash
uv sync                      # restore the environment from uv.lock
uv run python verify_env.py  # library versions, GPU, whether the data is present
uv run cs-download           # download application_train.csv into data/raw/
uv run pytest                # unit tests plus the data contract
```

`uv run cs-download` needs Kaggle credentials (`~/.kaggle/kaggle.json`) **and**
the [competition
rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
accepted in a browser; without the second, the API returns 403 even with valid
keys.

Without the data, the contract tests (`requires_data`) skip themselves, so
`uv run pytest` also passes on a fresh clone and in CI.

## Status

Data analysis, modelling and the scoring service are in place. The training
path is being refactored into classes with an orchestrated flow.
