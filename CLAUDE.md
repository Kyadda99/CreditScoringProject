# CLAUDE.md

This file gives Claude Code context for the project. Read it at the start of every session.

## Project rules

1. **TESTS**: All new fetures must be accompined by tests
2. **Commits**:Regardles of plan all commits needs to be pushed after the implementation plan finishes and they first neeed to pass my verifiacation. When commiting at the end limmits apply:
    - 2 commits max
    - 6-8 words per commit
    - coauthored by CLAUDE note is forbidden
3. **Tags**: tagging is forbidden i will do it myself





## About the project

Final project for the "Inżynier AI" course (Kodołamacz by Sages). End-to-end project:
analysis → model training → implementation → deployment.

**Deadline:** 2026-12-04 (per the brief — two months after the last session)

**Sources of record — read these before changing scope:**
- `docs/references/Projekt końcowy Inzynier AI grupa 28.03.2026-1.pdf` — the
  instructor's brief. The binding requirements.
- `docs/references/theme.md` — the project idea as submitted.
- `PLAN.md` — the phased implementation plan, with clause-by-clause
  requirements traceability.

**Important — project goal:**
- The goal is NOT to build the best possible model (brief, clause 5).
- The goal is to engage with **every** stage (EDA, modeling, implementation,
  deployment), especially the one that's most challenging (brief, clause 6).
- This is my first end-to-end ML project — I'm a beginner in every one of these
  areas. Priority: understanding, not just working code.
- The project is **optional** in the course. It exists to produce evidence for
  four micro-credentials: *Budowanie modeli uczenia maszynowego*, *Budowanie
  modeli z wykorzystaniem sieci neuronowych*, *Implementacja systemów AI*,
  *Wdrażanie modeli AI*.

**Instructor:** Marcin Rybiński (Slack). The brief explicitly offers help with
topic, data sources, methodology, algorithms and **infrastructure**, plus
end-of-project feedback. `PLAN.md` schedules four touchpoints — use them.

> The conventions below are distilled from the course modules (ML intro,
> classification, deep learning, NLP, computer vision, advanced Python,
> AI-system implementation, REST deployment) and, above all, from the
> instructor's own reference ML project. When in doubt, follow the patterns the
> course used rather than inventing new ones.

## Topic

**Credit scoring** — predicting the probability of loan default based on
client data.

- **Dataset:** Kaggle "Home Credit Default Risk" — **`application_train.csv`
  only**. Chosen for genuine categorical columns (so the NN's embeddings are
  real, not binned numerics), heavy missingness (so the EDA stage has
  substance), and realistic class imbalance. The auxiliary tables (`bureau`,
  `previous_application`, …) are explicitly out of scope — a stretch goal, not
  a requirement.
- **Target variable:** `TARGET`, binary classification (1 = default).
- Pull datasets programmatically (`kagglehub`) rather than committing raw files.
  The dataset's exact shape is asserted by a data-contract test, not assumed.

## Required scope (end-to-end)

1. **EDA**
   - feature distributions, missing data analysis and root causes, outlier
     analysis, new feature creation (e.g. debt-to-income, number of credit
     inquiries).
2. **Modeling**
   - several classical algorithms (logistic regression, Random Forest,
     XGBoost/LightGBM) + hyperparameter tuning.
   - plus a neural network (MLP with embeddings for categorical features) —
     compared against the classical approach.
3. **Implementation**
   - object-oriented programming, SOLID principles, design patterns where they
     make sense (e.g. Strategy for swapping models).
   - data pipelines as separate, testable components.
   - unit tests (pytest).
4. **Deployment**
   - API (FastAPI) for scoring.
   - containerization (Docker).
   - CI/CD (GitHub Actions).
   - **deployment to a cluster** — the brief says *"wdrożenie na klaster"*, and
     Cloud Run is serverless, so it does not literally satisfy the clause.
     **Decision: do both** — the live service on Cloud Run, plus Kubernetes
     manifests applied to a local cluster (kind/minikube) as the cluster
     evidence. GKE Autopilot rejected on cost and risk. See `PLAN.md` Phase 6.
   - local deployment → eventually Google Cloud Platform.
   - API design patterns to follow (see "API design patterns" section below).

## Tech stack

- Python 3.11+ (course modules pinned specific interpreters per module via
  `.python-version`; pick one and pin it).
- **uv** for everything (environment + dependencies). Never pip / poetry / conda.
- pandas, numpy, scikit-learn.
- imbalanced-learn (`imblearn`) — resampling for the class imbalance.
- XGBoost or LightGBM.
- PyTorch for the neural network; PyTorch Lightning is also acceptable (both
  were used in the course).
- **MLflow** — experiment tracking + model registry (the instructor's reference
  project is built around it).
- **Optuna** (+ `optuna-integration[mlflow]`) — hyperparameter tuning.
- FastAPI + uvicorn.
- pydantic / pydantic-settings for schemas and config (`.env` file, never
  hardcoded).
- pytest.
- Docker / docker-compose.
- GitHub Actions (CI/CD).
- GCP Cloud Run for the live service.
- Kubernetes (kind or minikube, **local**) — manifests for the same image, to
  satisfy the brief's *"wdrożenie na klaster"* clause without GKE's billing.

## Environment & tooling (uv)

Every course module used the same setup — mirror it:

- `pyproject.toml` is the single source of truth: `[project]` dependencies,
  `requires-python`, `[project.scripts]` entrypoints, tool config.
- `uv.lock` is committed and never hand-edited; `uv sync` reproduces `.venv/`
  exactly from it.
- `.python-version` pins the interpreter.
- Common commands: `uv sync`, `uv run <cmd>` (e.g. `uv run jupyter lab`,
  `uv run pytest`), `uv add <pkg>`. No manual venv activation needed.
- Runnable commands are declared in `[project.scripts]` and backed by a thin
  `main()` function; a root `main.py` just calls into `src/`.
- A `verify_env.py` script that checks key libraries, GPU availability and that
  the data is present — handy first thing in a session.
- PyTorch CUDA wheels come from a dedicated index — declare
  `[[tool.uv.index]]` / `[tool.uv.sources]` pointing at
  `https://download.pytorch.org/whl/<cuda>` with platform markers so macOS
  still gets MPS builds.

## Project structure

`src/` layout (`[tool.setuptools.packages.find] where = ["src"]`). Tests mirror
`src/`.

```
.
├── data/                 # raw and processed data (gitignored — pulled programmatically)
├── notebooks/            # EDA, prototyping — refactored into src/ once stable
├── src/
│   ├── config.py         # constants: DATA_PATH, TARGET, feature-group lists
│   ├── data.py           # load_data(path) -> DataFrame: read + clean + type-coerce
│   ├── preprocessing.py  # build_preprocessor() -> ColumnTransformer (no fitting)
│   ├── features/         # feature engineering (debt-to-income, inquiry counts, …)
│   ├── models.py         # get_models() -> dict[str, Estimator]  (named registry)
│   ├── evaluation.py     # evaluate / cross_validate / feature-importance reports
│   ├── train.py          # orchestration: split → pipeline → loop models → register
│   ├── tune.py           # Optuna HPO for the chosen model
│   ├── quality_gate.py   # metric thresholds a model must pass before promotion
│   ├── registry.py       # thin MLflow Model Registry wrapper
│   ├── nn/               # PyTorch model + training loop
│   └── api/              # FastAPI app (app.py, schemas.py, config.py)
├── tests/                # pytest, mirrors src/
├── docker/
├── .github/workflows/    # CI/CD
├── verify_env.py
├── .env.example          # placeholder config values (real .env is gitignored)
├── pyproject.toml        # deps (uv) + [project.scripts] + [tool.pytest.ini_options]
├── uv.lock
└── CLAUDE.md
```

## ML workflow conventions

Follow the instructor's reference project structure closely:

- **`config.py` holds the column contract.** Module-level constants:
  `DATA_PATH`, `TARGET`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`,
  `BINARY_FEATURES`. Everything else imports these — one place to change.
- **`data.py` = loading + cleaning only.** Read the file, coerce types
  (`pd.to_numeric(..., errors="coerce")`), drop/clip bad target rows, normalize
  binary columns. No feature engineering, no splitting.
- **`preprocessing.py` builds, never fits.** Per-group sklearn `Pipeline`s
  (impute → scale for numeric; impute → encode for categorical) composed in a
  `ColumnTransformer`. Returned unfitted.
- **Everything is one `Pipeline`.** `Pipeline([("prep", preprocessor),
  ("model", estimator)])` — fit and predict on raw columns so there is no
  train/serve skew. When resampling, use an `imblearn` pipeline so sampling
  only touches training folds.
- **`models.py` is a named registry.** `get_models()` returns
  `{"LogReg": ..., "RandomForest": ..., "XGBoost": ...}` with baseline
  hyperparameters.
- **`train.py` orchestrates.** Split → for each model: open one MLflow run,
  fit the pipeline, evaluate on the test set, log metrics, register the model.
  Then pick the best by the chosen metric and promote it.
- **`tune.py` = Optuna.** `objective(trial)` with `trial.suggest_*`,
  `StratifiedKFold` CV, `MLflowCallback`, `TPESampler(seed=42)`. Refit the best
  params on the full training data, then log + register + promote.
- **`quality_gate.py` gates promotion.** Hard thresholds (min ROC-AUC, min
  recall on the default class, etc.); `quality_gate(metrics) -> bool`. A model
  that fails is not promoted.
- **`registry.py` wraps MLflow Model Registry.** `register(run_id, name)`,
  `promote(mv, alias="production")`, `load_production(name)`, `list_models()`.
  Use **aliases** (`@production`), not deprecated stages.
- **Reproducibility:** `random_state=42` everywhere, seeded samplers,
  `n_jobs=-1` for CV/search.
- **Experiment tracking:** `mlflow.set_experiment(...)`,
  `mlflow.sklearn.autolog(...)`, one `start_run(run_name=...)` per model,
  `mlflow.log_metrics(dict)`. Local backend (`mlflow.db` + `mlruns/`).
- **Keep a results table** comparing models/experiments (accuracy, F1, ROC-AUC,
  PR-AUC, chosen threshold) — a habit from the classification module.

## Classification specifics (imbalanced binary — directly relevant here)

- **Don't judge by accuracy.** Use confusion matrix, precision / recall / F1,
  ROC-AUC, and the precision–recall curve. Choose the primary metric from the
  business cost — missing a default is expensive, so recall on the default
  class matters.
- **Tune the decision threshold.** Work from `predict_proba`, not the default
  0.5. The threshold can be a runtime parameter of the scorer; a middle band
  can map to "refer for manual review" ("don't know").
- **Handle imbalance two ways and compare them:** resampling (`imblearn` —
  SMOTE / undersampling, inside the pipeline) and class weights
  (`class_weight`, XGBoost `scale_pos_weight`).
- **Hyperparameter search:** `GridSearchCV` / `RandomizedSearchCV` for the
  classical models (course), Optuna for the finalist. Always `StratifiedKFold`.
- **Feature selection** fitted on the training split only: L1/Lasso,
  `SequentialFeatureSelector`, model feature importances.

## Neural network conventions

From the deep-learning / NLP modules:

- Explicit `device` selection (`cuda` → `mps` → `cpu`).
- `Dataset` / `DataLoader` (+ `collate_fn` when needed); `nn.Module` subclass.
- `nn.Embedding` per categorical feature, concatenated with the scaled numeric
  block, into an MLP head.
- `Adam` with `weight_decay` (L2); `ReduceLROnPlateau` scheduler; gradient
  clipping (`clip_grad_norm_`).
- **Early stopping with patience**: track validation loss, save the best
  `state_dict` to `best_model.pt`, reload it for the final test evaluation.
- Hyperparameters as UPPERCASE module-level constants, each with a short comment
  explaining the choice.
- Log the run to MLflow like the classical models so the comparison is
  apples-to-apples.

## API design patterns

For the scoring API (patterns taken from the course's REST-deployment module):

- **Lifespan startup loading.** Load the trained pipeline (via
  `registry.load_production(...)`), settings, and threshold once at startup
  (FastAPI `lifespan`), store on `app.state`. Never reload per request.
- **`/health` endpoint** returning `{"status": "ok"}`.
- **`schemas.py`** — pydantic request/response models, separate from business
  logic. `Field(..., examples=[...])`, validation on inputs (`ge=`, `le=`,
  `min_length=`); response models may inherit request models.
- **`config.py`** — `Settings(BaseSettings)` with
  `SettingsConfigDict(env_file=".env", extra="ignore")`, a default for every
  field, `SecretStr` for secrets. Ship `.env.example`, never commit `.env`.
- **Middleware** for a request-ID (`X-Request-ID`) and request timing
  (`X-Process-Time-Ms` + a structured log line per request).
- **Background tasks for logging.** Log every prediction (input + output +
  UTC-ISO timestamp) to a `.jsonl` file via `BackgroundTasks`, catching
  `OSError` — useful for monitoring model behaviour in production.
- Optional in-memory TTL cache for repeated identical requests, surfaced with
  an `X-Cache: HIT/MISS` header.
- **Correct HTTP semantics:** `HTTPException` with real status codes (404, 415,
  422, 204 for no-content deletes).
- `async def` endpoints; keep CPU-bound work (batch scoring, retraining) off
  the event loop.
- `uvicorn.run(...)` inside a `main()` wired to `[project.scripts]`.

## OOP / SOLID patterns from the course

**The class names are committed.** `docs/references/theme.md` — the project
idea submitted to the instructor — names four classes explicitly:
**`DataLoader`, `Preprocessor`, `ModelTrainer`, `Predictor`**, plus the
**Strategy** pattern for swapping models and **unit tests of the data
transformations**. Use those names. Diverging from the accepted proposal
without saying so is the kind of thing a reviewer notices.

These arrive in Phase 5 of `PLAN.md`, not before — earlier phases build the
same behaviour as functions, deliberately, because abstracting before you know
what varies produces the wrong abstraction.

- **ABCs define contracts** (`abc.ABC`, `@abstractmethod`); the docstring states
  the contract.
- **Strategy** = a list of interchangeable implementations behind one interface,
  iterated by a coordinator (course example: notification channels; here:
  swappable models / scorers).
- **Mixins** for cross-cutting concerns (logging, retry), composed via MRO with
  `super()` chaining; document the resulting MRO in the class docstring.
- **`@dataclass`** for config-carrying coordinators; `field(default_factory=…)`
  for sensible defaults.
- **`from_settings(cls, settings)` classmethod factories** build the object
  graph from `Settings` and degrade gracefully when optional config is absent.
- One class, one responsibility — compose small classes rather than growing one.

## Coding conventions

- `from __future__ import annotations` at the top of every module.
- Type hints on every signature; write `-> None` explicitly.
- Google-style docstrings for public classes/functions — short, state the
  contract or the "why".
- Module-level constants UPPERCASE; `_`-prefix private module state.
- `pathlib.Path`, not `os.path`.
- `logging.getLogger(__name__)` per module for library/production code (`print`
  is fine in notebooks and CLI/demo scripts).
- English identifiers; Polish is acceptable in comments/docstrings/log strings
  (course convention), but this file and the code stay English.
- Formatting: black + ruff. PEP8 naming.
- pytest config in `pyproject.toml` under `[tool.pytest.ini_options]`
  (`testpaths = ["tests"]`, `pythonpath = ["src", "tests"]`), not a separate
  `pytest.ini`. Run with `uv run pytest`.

## Data & gitignore

Gitignore: `.venv/`, `*.egg-info`, `.env`, `*.pyc`, `data/` (large files),
prediction logs (`*.jsonl`), model artifacts (`*.pt`, `*.pth`), `mlruns/`,
`mlflow.db`, any local DBs.

## How I want to work with Claude Code

This is my first project of this kind, so:

- **Work in small steps.** One piece of functionality at a time — don't generate
  whole modules at once without explanation.
- **Explain the "why".** Before implementing a design pattern, technique, or
  library, briefly explain why we're using it here.
- **Ask before big architectural decisions**, don't just assume.
- **Commit often**, with descriptive messages — makes it easier to roll back to
  a working version.
- **Write tests as you go**, not at the end of the project.
- **Don't over-optimize the model.** The priority is a complete end-to-end
  pipeline, not the best possible score.

## Project status

- [ ] Dataset selection and download
- [ ] EDA
- [ ] Baseline (logistic regression)
- [ ] Classical models + tuning
- [ ] Neural network
- [ ] Refactor into production code (OOP, SOLID)
- [ ] Unit tests
- [ ] MLflow tracking + model registry + quality gate
- [ ] API (FastAPI)
- [ ] Containerization (Docker)
- [ ] CI/CD
- [ ] Deployment to GCP
