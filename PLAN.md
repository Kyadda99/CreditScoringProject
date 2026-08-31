# Credit Scoring — End-to-End Implementation Plan

> **For agentic workers:** This plan is a *guide*, not a code listing. It defines
> what each phase must achieve, how it is proven done, and in what order.
> Deliberately contains no source code — the code is written session by session
> against the criteria below. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a credit-default scoring system that is exercised end-to-end —
data → model → service → cloud — with every stage of the course engaged, not
just the comfortable ones.

**Architecture:** Vertical slices. Phase 1 drives one thin path through all four
layers (data, model, service, container) with the dumbest possible model. Every
later phase thickens exactly one layer while keeping the path green. Nothing is
built that the previous phase has not already proved is needed.

**Tech Stack:** Python 3.11+ · uv · pandas / scikit-learn / imbalanced-learn ·
XGBoost or LightGBM · PyTorch · MLflow · Optuna · FastAPI + uvicorn ·
pydantic-settings · pytest · Docker · GitHub Actions · GCP Cloud Run ·
Kubernetes (kind/minikube, local)

**Spec:** [`CLAUDE.md`](CLAUDE.md) — conventions, module layout, API patterns,
coding rules. This plan argues from that spec; read both.

**Requirements of record:** [`docs/references/Projekt końcowy Inzynier AI grupa
28.03.2026-1.pdf`](docs/references/) — the instructor's brief. See
[Requirements traceability](#requirements-traceability) for the clause-by-clause
mapping. **Project idea of record:**
[`docs/references/theme.md`](docs/references/theme.md).

**Dataset:** Home Credit Default Risk — `application_train.csv` only.
Auxiliary tables (`bureau`, `previous_application`, …) are explicitly out of
scope; see Stretch Goals.

**Deadline:** 2026-12-04 · **Plan written:** 2026-08-31 · **13.5 weeks**

---

## Global Constraints

Every task's requirements implicitly include this section. Values copied
verbatim from `CLAUDE.md`.

| Constraint | Value |
|---|---|
| Python | 3.11+, pinned in `.python-version` |
| Dependency tool | **uv only** — never pip, poetry, or conda |
| Lockfile | `uv.lock` committed, never hand-edited |
| Layout | `src/` layout; `tests/` mirrors `src/` |
| Model artifact | Always one sklearn `Pipeline([("prep", …), ("model", …)])` — fit and predict on raw columns, no train/serve skew |
| Reproducibility | `random_state=42` everywhere; seeded samplers; `n_jobs=-1` for CV |
| Module header | `from __future__ import annotations` at top of every module |
| Typing | Type hints on every signature, `-> None` written explicitly |
| Docstrings | Google style, on public classes and functions |
| Paths | `pathlib.Path`, never `os.path` |
| Logging | `logging.getLogger(__name__)` in library code; `print` only in notebooks and CLI |
| Formatting | black + ruff, PEP8 naming |
| Test config | `pyproject.toml` → `[tool.pytest.ini_options]`, `testpaths = ["tests"]`, `pythonpath = ["src", "tests"]` |
| Test command | `uv run pytest` |
| Model registry | MLflow **aliases** (`@production`) — never deprecated stages |
| Secrets | `.env` never committed; `.env.example` always committed |
| Data | `data/` gitignored; datasets pulled programmatically |
| Language | English identifiers; Polish allowed in comments and log strings |

---

## Requirements traceability

Every clause of the instructor's brief (PDF p.3, *End-to-end*), mapped to the
phase that produces its evidence. Verified 2026-08-31.

| # | Requirement (as written) | Phase | Status |
|---|---|---|---|
| 1a | Analiza rozkładów cech | 2 | covered |
| 1b | Analiza braków danych **i ich powodów** | 2 | covered — the *reasons* are required, not just the counts |
| 1c | Analiza wartości odstających | 2 | covered |
| 1d | Utworzenie nowych cech | 2 | covered — ≥3 domain features |
| 2a | Przetestowanie kilku algorytmów | 3 | covered — ≥3 algorithms |
| 2b | Optymalizacja hiperparametrów | 3 | covered — Optuna, `RandomizedSearchCV` as fallback |
| 3a | Programowanie obiektowo, wzorce, SOLID | 5 | covered — Strategy + named classes |
| 3b | Pipeline'y do przetwarzania danych | 1, 2 | covered — one sklearn `Pipeline`, prep + estimator |
| 3c | Testy jednostkowe | all | covered — every phase gates on `uv run pytest` |
| 4a | Implementacja API | 1 | covered |
| 4a | Konteneryzacja | 1 | covered — pulled forward deliberately |
| 4a | **Wdrożenie na klaster** | 6 | covered — Cloud Run **+ local k8s cluster**; see below |
| 4a | CICD | 6 | covered |
| 4b | Lokalnie lub na GCP | 6 | covered |
| 5 | Nie skupiamy się na najlepszym modelu | all | explicit in `CLAUDE.md` and in every phase's DoD |
| 6 | Zmierzyć się z każdym obszarem, **szczególnie z tym który stanowi wyzwanie** | all | the vertical-slice structure exists for this; see [Challenge area](#challenge-area--where-to-spend-the-margin) |

### Resolved: "wdrożenie na klaster" — Cloud Run **and** a local cluster

The brief says *deployment to a **cluster***. Cloud Run is serverless and is
**not** a cluster; `theme.md` hedges with "Cloud Run/GKE". Clause 4b softens
this ("locally **or** on GCP") but does not remove the word *klaster*.

**Decision (2026-08-31): do both.** Phase 6 delivers the live service on Cloud
Run *and* Kubernetes manifests applied to a local cluster (kind or minikube),
documented as the cluster evidence. This satisfies both readings of the clause,
costs nothing, and keeps GKE Autopilot's billing and complexity out of the
project's highest-risk phase.

**GKE Autopilot is rejected** — it would stack a managed-Kubernetes learning
curve on top of the already-flagged weak area, under deadline pressure, for
money.

**The instructor question is retained but deferred.** Ask Marcin whether Cloud
Run alone would have sufficed at the *start of Phase 6*, not in Phase 0 — by
then the question is concrete and his answer can only simplify the work, never
expand it. If he says Cloud Run counts on its own, the local-cluster half
becomes optional and Phase 6 gives back a session. Asking early would have
meant acting on an answer to a question that had not yet taken its real shape.

## Instructor touchpoints

PDF p.5 offers help with topic selection, data sources, methodology, algorithms,
**infrastructure**, and end-of-project feedback. Contact: **Marcin Rybiński**,
on Slack. This is free expert review and the plan should use it deliberately
rather than incidentally.

| When | Ask |
|---|---|
| **Phase 0** | Confirm the topic and the Home Credit dataset choice. Cheap, and a wrong topic caught in September costs nothing. |
| **End of Phase 3** | Show the model comparison table; ask whether the classical evidence is sufficient for the ML micro-credential. |
| **Start of Phase 6** | Infrastructure review before building the pipeline — the phase where outside help is worth the most. **Ask the "klaster" question here**: does Cloud Run alone satisfy clause 4a? A yes makes the local-cluster work optional and hands back a session; a no confirms the plan already in place. Either way the plan does not stall waiting for the answer. |
| **Phase 7** | Request the end-of-project feedback the brief promises. |

## What "one element" means per phase

The organising principle is a **walking skeleton**: build the thinnest thing
that touches every layer, then thicken one layer at a time. A phase is "one
element" in the sense that it thickens exactly **one** layer — the other three
are kept alive but not advanced.

| Phase | The one element being built | Everything else is |
|---|---|---|
| **0 — Foundation** | The ground itself: repo, env, data contract, CI | *(not a slice — see note)* |
| **1 — Walking skeleton** | The **path**: raw columns → logistic regression → HTTP → container | as dumb as legally possible |
| **2 — Data** | **EDA + feature engineering** | model stays logistic regression |
| **3 — Classical models** | **Model selection, tuning, thresholding, registry** | features frozen from Phase 2 |
| **4 — Neural network** | **The PyTorch MLP** and its honest comparison | pipeline and API unchanged in shape |
| **5 — Implementation** | **OOP / SOLID / test suite / observability** | no new modelling at all |
| **6 — Deployment** | **CI/CD → Artifact Registry → Cloud Run** | code frozen except deploy config |
| **7 — Delivery** | **The submission**: report, README, demo, tag | nothing new is built |

> **Note on Phase 0.** It is the one phase that is *not* a vertical slice, and
> pretending otherwise would be dishonest. You cannot slice through layers that
> do not exist yet. Phase 0 exists to make Phase 1 a two-week job instead of a
> five-week one. It is deliberately small and deliberately first.

---

## Why this order and not another

**Why the container in Phase 1 instead of Phase 6.**
This is the most important decision in the plan, and it is deliberately
contrary to the obvious ordering. Deployment carries the most unknown-unknowns
and — per the challenge-area flag below — the least prior evidence. Finding out
in late November that the image will not build, that the model artifact will
not load inside the container, or that Cloud Run rejects the service, is fatal
with three days left. Finding out in mid-September is a Tuesday afternoon.
Pulling Docker forward converts the project's largest risk into its earliest
one. Secondary benefit: from week three onward there is **always something
submittable**, even if every later phase collapses.

**Why EDA is Phase 2 and not Phase 1.**
EDA without a metric is browsing. Once Phase 1 has produced a baseline ROC-AUC
and a working scoring path, every EDA finding becomes a testable claim — "does
this feature move the baseline?" — instead of an interesting plot. The
discipline is worth the two-week delay.

**Why the neural network is Phase 4 and not earlier.**
Three independent reasons, any one sufficient. *(a)* Its embedding layer needs
the categorical-cardinality decisions that Phase 2 makes — building it first
means building it twice. *(b)* It needs the tuned classical results from Phase 3
as its comparison baseline; a neural network with nothing to beat demonstrates
nothing, and the course requirement is explicitly a *comparison*. *(c)* It is
the single component most likely to consume unbounded time, so it must sit
*after* the classical evidence is already banked and safe.

**Why the SOLID refactor is Phase 5 and not Phase 1.**
Refactoring toward abstractions before you know what varies is guesswork, and
guessed abstractions are worse than none. After Phases 3 and 4 you know
precisely what varies — the estimator, the preprocessing, the decision
threshold — so the Strategy seams are read off the code rather than invented.
Writing `ModelStrategy` in week two would mean writing the wrong one.

**Why cloud is last.**
It is the only phase whose dependencies are entirely produced by other phases:
a container that works (1), a model in a registry (3), a green test suite (5).
It cannot meaningfully start earlier, which is exactly why Phase 1 rehearses its
riskiest part in miniature.

---

## Micro-credential coverage

Which phase produces the *evidence* for each of the four areas.

| Micro-credential | Primary evidence | Supporting evidence | Insurance |
|---|---|---|---|
| **Classical ML** | Phase 3 — several algorithms, Optuna tuning, imbalance handling, threshold optimisation, model comparison table | Phase 2 — EDA, missing-data analysis, engineered features | Phase 1 baseline |
| **Neural networks** | Phase 4 — PyTorch MLP with categorical embeddings, trained with early stopping, compared like-for-like against the classical champion | Phase 3 — the baseline it is measured against | — |
| **Implementation** | Phase 5 — OOP/SOLID refactor, Strategy for model swapping, pytest suite, structured logging | Phase 0/1 — `src/` layout, module boundaries, CI from day one | Phase 1 structure |
| **Deployment** | Phase 6 — GitHub Actions CI/CD, Artifact Registry, Cloud Run, monitoring | Phase 1 — Dockerfile and local container; Phase 3 — registry-driven model loading | **Phase 1 container** |

The **Insurance** column is the point of the walking skeleton: if the project
runs out of runway, Phase 1 alone constitutes a thin but genuine, defensible
submission against all four areas. Everything after it raises the grade rather
than creating the evidence from nothing.

---

## Challenge area — where to spend the margin

`CLAUDE.md` says the goal is to engage with every stage *"especially the one
that's most challenging"* but does not name it. I do not have your
self-assessment, so this is inferred from the course folders — **please correct
me if it is wrong, because the time budget below depends on it.**

**Evidence gathered from `h:\AI kurs\`:**

- Across eleven course-module folders there is **not one Dockerfile, not one
  `docker-compose.yml`, and not one GitHub Actions workflow**. Zero prior
  artifacts.
- Likewise **no cloud deployment artifacts** of any kind.
- Your own Python in the course folders is **script-style and procedural** —
  flat modules, module-level constants, functions. The ABC / mixin / MRO
  material in `AdvancedPython\agno_oop` appears to be instructor-supplied
  rather than authored by you.
- Conversely, there is **strong** evidence of competence in environment and
  tooling (the Computer Vision README's CUDA and TLS-certificate diagnosis is
  genuinely expert work), in **FastAPI** (a working service with middleware,
  background tasks, streaming responses and a TTL cache), and in **MLflow**
  (your own `mlruns/` under `z5\Archive`).

**Conclusion:** the weak spot is **containerisation → CI/CD → cloud deploy**,
with the **OOP/SOLID production refactor** a clear second. Notably it is *not*
the API layer or the ML itself, both of which you have already done.

**How the plan absorbs this:**

1. Docker is pulled all the way forward into **Phase 1**, so the first
   container failure happens in week 3 with ten weeks of runway behind it.
2. **Phase 6 gets the largest session budget** of any phase (4 sessions) *and*
   sits in front of the schedule buffer, so it can overrun without eating the
   deadline.
3. Both Phase 1 and Phase 6 carry an explicit **timeboxed spike** — a
   throwaway experiment run *before* the real task, to surface unknowns while
   they are still cheap.
4. Phase 5 is scheduled *after* the modelling work, so the SOLID refactor is
   applied to code you already understand rather than to a blank page.

---

## Schedule

**Assumption: ~2 focused sessions per week, a session being 2–3 uninterrupted
hours.** Adjust the calendar if your real pace differs — the session counts are
the durable part, the dates are derived.

| Phase | Sessions | Target completion | Cumulative |
|---|---|---|---|
| 0 — Foundation | 2 | 2026-09-07 | 1.0 wk |
| 1 — Walking skeleton | 4 | 2026-09-21 | 3.0 wk |
| 2 — Data | 4 | 2026-10-05 | 5.0 wk |
| 3 — Classical models | 4 | 2026-10-19 | 7.0 wk |
| 4 — Neural network | 3 | 2026-10-29 | 8.5 wk |
| 5 — Implementation | 3 | 2026-11-09 | 10.0 wk |
| 6 — Deployment | 5 | 2026-11-27 | 12.5 wk |
| 7 — Delivery | 2 | 2026-12-04 | 13.5 wk |
| **Buffer** | — | — | **none** |

**27 sessions, ~13.5 weeks, zero slack.** Be clear-eyed about this: the
*klaster* decision spent the buffer. Phase 7 now lands on the deadline itself,
which means the plan has no tolerance for a single slipped session.

Three ways to buy margin back, in the order they should be taken:

1. **Marcin's answer at the start of Phase 6.** If Cloud Run alone satisfies
   clause 4a, the fifth session is returned immediately and the buffer is
   restored. This is the cheapest margin available and costs one Slack message.
2. **Pre-authorise cut #1.** Take the Optuna → `RandomizedSearchCV` cut at the
   *first* sign of slippage in Phase 3 rather than deliberating about it. One
   session, no requirement lost — the brief asks for hyperparameter
   optimisation, not for Optuna specifically.
3. **Raise cadence.** Three sessions in any one week restores four days. The
   schedule assumes two; a single heavier week early is far cheaper than a
   compressed Phase 7.

Treat any phase finishing more than one session late as a schedule event, not a
detail — go to the cut list rather than silently compressing Phase 6 or 7.

### Cut list — in this order, if you fall behind

Breadth beats depth: the stated goal is engaging every stage, so cuts always
remove *depth within* a phase, never a whole phase.

1. **Optuna** in Phase 3 → fall back to `RandomizedSearchCV`. Costs ~1 session,
   loses little evidence.
2. **Auxiliary EDA depth** in Phase 2 → keep the missing-data and target
   analysis, drop the long tail of univariate plots.
3. **The third and fourth classical model** in Phase 3 → logistic regression
   plus one gradient-boosted model is a sufficient comparison.
4. **`docker-compose`** in Phase 5 → the plain Dockerfile from Phase 1 is
   enough for Cloud Run.
5. **Neural-network tuning** in Phase 4 → train one sensible architecture,
   report it honestly, do not tune it.
6. **Monitoring/observability polish** in Phase 6 → keep prediction logging,
   drop dashboards.

**Never cut:** the Phase 1 skeleton, the Phase 6 Cloud Run deploy, the local
cluster evidence for clause 4a *(unless Marcin waives it)*, or the Phase 7
report. Those are the submission.

---

## Conventions that apply to every phase

- **Branch per phase**, named `phase-N-<slug>`; merge to `main` at the gate and
  tag `v0.N`. Phase 7 tags `v1.0`.
- **Every phase ends with two artifacts**: a green `uv run pytest`, and a
  findings document at `docs/findings/NN-<slug>.md` committed to the repo.
  The findings file is what turns a notebook conclusion into something a
  reviewer — or you in six weeks — can actually read.
- **EDA-phase tests assert data contracts**, not statistics: expected columns
  present, dtypes as declared, target rate inside a plausible band, no
  post-outcome leakage columns surviving into the feature set. This makes the
  notebook's conclusions executable and catches silent upstream changes.
- **Commit often**, with descriptive messages. Small steps, one concern each.
- **Ask before architectural decisions.** Per `CLAUDE.md`, do not assume.
- **Explain the why** before introducing a pattern or library.

---

# Phase 0 — Foundation

**Phase goal.** Turn an empty repository into a working, tested, CI-backed uv
project with the dataset downloaded and its shape locked down in assertions.

**Scope across the four layers.**
- *EDA* — no analysis, but the **data contract** is asserted: row count, column
  count, dtypes, target positive rate, presence of the declared feature groups.
- *Modeling* — none. Deliberately deferred to Phase 1.
- *Implementation* — repo scaffolding, `src/` layout, `config.py` column
  contract, `data.py` loader, pytest wired up, black/ruff configured.
- *Deployment* — a CI workflow that runs lint and tests on push. Thin, but it
  means CI is never "bolted on later".

**Completion criteria (definition of done).**
- [ ] `uv sync` reproduces the environment from a committed `uv.lock`.
- [ ] `uv run python verify_env.py` reports library versions, GPU availability,
      and confirms the dataset is present — exiting non-zero if not.
- [ ] `uv run pytest` passes, with at least one real assertion about the data.
- [ ] The data contract test fails loudly if the CSV is swapped or truncated.
- [ ] CI runs on push and is green on `main`.
- [ ] `.gitignore` covers `data/`, `mlruns/`, `mlflow.db`, `*.pt`, `*.pth`,
      `*.jsonl` — verify with `git status` that no data file is ever staged.

**Final artifact.** Tag `v0.0` on `main`: a repository someone else could clone,
run `uv sync && uv run pytest` in, and get green.

**Dependencies.** None. This is the start.

**Estimated time.** 2 sessions.

**Known risks / unknowns.**
- The Kaggle download requires API credentials — Home Credit is a competition
  dataset and needs competition rules accepted in the browser first. Budget
  time for this; it is a common half-hour sink.
- `application_train.csv` is ~160 MB. Confirm it is gitignored **before** the
  first `git add`, not after.
- The figures quoted below are from memory and **must be verified, not
  trusted** — that is precisely what the contract test is for.

**Task checklist.**
- [ ] **Message Marcin on Slack**: confirm the topic and the Home Credit dataset
      choice. (The "klaster" question waits until Phase 6 — see the touchpoints
      table.)
- [ ] Initialise the uv project; pin the interpreter in `.python-version`.
- [ ] Declare initial dependencies in `pyproject.toml`; commit `uv.lock`.
- [ ] Add `[tool.pytest.ini_options]` with `testpaths` and `pythonpath` per the
      global constraints; add black/ruff config.
- [ ] Extend `.gitignore` with the data, MLflow, model-artifact and log entries.
- [ ] Set up Kaggle API credentials; download `application_train.csv` into
      `data/raw/` via `kagglehub`, driven by a small script so it is repeatable.
- [ ] Write `verify_env.py`: library versions, GPU check, dataset presence.
- [ ] Create `src/config.py` — `DATA_PATH`, `TARGET`, and the three feature-group
      lists. Derive the group membership from the actual dtypes rather than
      typing 122 column names by hand.
- [ ] Create `src/data.py` with the loader only — read and type-coerce. No
      cleaning decisions yet; those need Phase 2's evidence.
- [ ] Write the data-contract tests: shape, dtypes, target rate in a band,
      declared feature groups all present in the frame.
- [ ] Add the GitHub Actions workflow: checkout, install uv, `uv sync`, ruff,
      `uv run pytest`. Tests that need the dataset must skip cleanly in CI —
      CI has no Kaggle credentials and must not try to download 160 MB.
- [ ] Write `docs/findings/00-data-contract.md`: the verified shape of the data,
      and any figure that differed from what this plan assumed.
- [ ] Tag `v0.0`.

---

# Phase 1 — Walking skeleton

**Phase goal.** Get a prediction to travel the entire path — from a raw CSV row
to an HTTP response served from inside a Docker container — using the dumbest
model that will fit.

**Scope across the four layers.**
- *EDA* — none beyond Phase 0's contract. Feed the model raw columns with
  crude imputation and accept that it is bad.
- *Modeling* — logistic regression, default hyperparameters, no imbalance
  handling, no threshold tuning. Its ROC-AUC is the number every later phase is
  measured against.
- *Implementation* — `preprocessing.py`, `models.py`, `evaluation.py`,
  `train.py` and the FastAPI app, each thin but real and in its final location.
- *Deployment* — a Dockerfile; the container starts, loads the model, and
  answers a scoring request on localhost.

**Completion criteria (definition of done).**
- [ ] `uv run python -m src.train` trains, evaluates, and logs one MLflow run
      with ROC-AUC recorded.
- [ ] The trained artifact is a single sklearn `Pipeline` — preprocessing and
      estimator together — persisted to disk.
- [ ] `GET /health` returns `{"status": "ok"}`.
- [ ] `POST /score` accepts a client payload and returns a default probability.
- [ ] The model is loaded **once at startup** via `lifespan` onto `app.state` —
      prove it by logging at load time and confirming one line for many requests.
- [ ] `docker build` succeeds and `docker run` serves both endpoints, verified
      with a real request from the host.
- [ ] Tests cover: the pipeline fits and predicts; `/health`; `/score` returns a
      probability in `[0, 1]`; `/score` rejects a malformed payload with 422.
- [ ] The baseline ROC-AUC is written down in the findings file.

**Final artifact.** Tag `v0.1`: a Docker image that scores a credit
application. Thin, honest, and end-to-end.

**Dependencies.** Phase 0 complete — the loader and the column contract must
exist, because the API's request schema is generated from the same feature
lists.

**Estimated time.** 4 sessions — *including a timeboxed 1-session Docker
spike*, per the challenge-area mitigation.

**Known risks / unknowns.**
- **The container is the risk.** Image size with the full scientific stack,
  getting the model artifact into the image, and matching the Python version
  are all first-time problems. Run the spike first: a container that serves
  `/health` and nothing else. Only then add the model.
- The request schema for a 122-column table is unwieldy. Decide early whether
  `/score` takes every column or a documented subset — this decision propagates
  to every later phase, so make it deliberately and write it down.
- Resist tuning anything. The value of this phase is that it is *finished*, not
  that it is good.

**Task checklist.**
- [ ] **Spike first:** minimal Dockerfile, FastAPI, `/health` only. Build, run,
      curl it. Throw the result away; keep the knowledge.
- [ ] `src/preprocessing.py` — `build_preprocessor()` returning an unfitted
      `ColumnTransformer` with per-group impute/scale/encode pipelines.
- [ ] `src/models.py` — `get_models()` returning a dict with logistic regression
      as its only entry.
- [ ] `src/evaluation.py` — an evaluate function returning a metrics dict
      including ROC-AUC.
- [ ] `src/train.py` — stratified split, compose the pipeline, fit, evaluate,
      log to MLflow, persist the artifact.
- [ ] Run it; record the baseline ROC-AUC.
- [ ] `src/api/schemas.py` — pydantic request and response models with field
      validation and `Field(examples=[...])` so the OpenAPI docs are usable.
- [ ] `src/api/config.py` — `Settings(BaseSettings)` reading `.env`; commit
      `.env.example`.
- [ ] `src/api/app.py` — `lifespan` loading model and settings onto
      `app.state`; `/health`; `/score`.
- [ ] Tests: pipeline, `/health`, `/score` happy path, `/score` validation
      failure.
- [ ] Real Dockerfile; build; run; score a request against the container.
- [ ] `docs/findings/01-skeleton.md`: baseline ROC-AUC, the payload-shape
      decision and its rationale, and what the Docker spike taught you.
- [ ] Merge, tag `v0.1`.

---

# Phase 2 — Data

**Phase goal.** Replace the raw-column shortcut with a deliberate, analysed,
tested feature set — and prove it moves the baseline.

**Scope across the four layers.**
- *EDA* — **the thick layer this phase.** Distributions, missing-data analysis
  including *why* each column is missing, outliers, target relationships,
  cardinality of every categorical.
- *Modeling* — still logistic regression. Holding the model constant is what
  makes the feature comparison honest.
- *Implementation* — a `features/` module with engineered features as pure,
  independently testable functions; cleaning decisions moved into `data.py`.
- *Deployment* — the API serves the new pipeline; the request schema follows the
  final feature set; the container is rebuilt and re-verified.

**Completion criteria (definition of done).**
- [ ] An EDA notebook exists under `notebooks/`, run top to bottom without error.
- [ ] Every column is classified: keep, drop, or engineer — with a written
      reason for each drop.
- [ ] Missing data is handled per *cause*, not with one blanket strategy, and
      the reasoning is recorded.
- [ ] At least three domain features exist — debt-to-income, credit-to-income,
      employment-duration ratio, or similar — each as a tested pure function.
- [ ] Leakage check performed and documented: no feature encodes information
      unavailable at application time.
- [ ] The same logistic regression is re-scored on the new features and the
      before/after ROC-AUC is recorded. **A drop is a valid, reportable
      result** — investigate and write it up, do not quietly revert.
- [ ] Tests: each engineered feature against hand-computed values; the
      edge cases (zero income, missing denominator) behave as designed; the
      updated data contract holds.
- [ ] Container rebuilt; `/score` still works with the new schema.

**Final artifact.** Tag `v0.2`, plus `docs/findings/02-eda.md` — the document
that carries most of your classical-ML evidence for the missing-data and
feature-engineering requirement.

**Dependencies.** Phase 1's baseline ROC-AUC. Without it this phase has no
success criterion.

**Estimated time.** 4 sessions.

**Known risks / unknowns.**
- 122 columns is enough to disappear into for two weeks. Timebox the univariate
  work; the marks are in the missing-data reasoning and the engineered features,
  not in plot count.
- Some columns are heavily missing (the building/apartment block, and one of
  the external-source scores, are the usual suspects — **verify rather than
  assume**). Decide per column: drop, impute, or add an explicit
  missing-indicator, and say why.
- Categorical cardinality measured here **feeds Phase 4's embedding
  dimensions**. Record the cardinalities in the findings file; you will need
  them again in six weeks.

**Task checklist.**
- [ ] EDA notebook: target rate, class imbalance, per-column distributions.
- [ ] Missing-data analysis — pattern, proportion, and probable cause per column.
- [ ] Outlier analysis; decide clip / drop / leave, and record it.
- [ ] Categorical cardinality table → into the findings file.
- [ ] Correlation and target-relationship review; identify redundant columns.
- [ ] Explicit leakage review of every retained column.
- [ ] Write the keep/drop/engineer decision table.
- [ ] Implement engineered features as pure functions in `src/features/`.
- [ ] Unit-test each feature, including its edge cases.
- [ ] Update `config.py` feature groups and `data.py` cleaning to match.
- [ ] Re-run Phase 1's training; log as a new MLflow run; compare ROC-AUC.
- [ ] Update the API request schema; rebuild the container; re-verify `/score`.
- [ ] `docs/findings/02-eda.md`: decisions, reasons, cardinalities, before/after
      metric.
- [ ] Merge, tag `v0.2`.

---

# Phase 3 — Classical models

**Phase goal.** Turn one baseline into a defended champion — several algorithms,
tuned, imbalance handled, threshold chosen on business grounds, promoted through
a quality gate into the model registry.

**Scope across the four layers.**
- *EDA* — frozen. Feature set does not change; that is what makes the model
  comparison valid.
- *Modeling* — **the thick layer.** Multiple algorithms, hyperparameter search,
  imbalance strategies compared, decision-threshold optimisation.
- *Implementation* — `tune.py`, `quality_gate.py`, `registry.py` per the
  reference project's structure.
- *Deployment* — the API stops loading a file and starts loading
  `models:/<name>@production` from the registry.

**Completion criteria (definition of done).**
- [ ] At least three algorithms trained and compared: logistic regression,
      a tree ensemble, a gradient-boosted model.
- [ ] Both imbalance strategies tried and compared — resampling via `imblearn`
      *inside* the pipeline, and class weights / `scale_pos_weight`. The
      resampling must not leak into validation folds; state how you ensured it.
- [ ] Hyperparameter search run — Optuna preferred, `RandomizedSearchCV`
      acceptable per the cut list — with `StratifiedKFold` CV.
- [ ] Decision threshold tuned from `predict_proba` rather than defaulting to
      0.5, chosen against a stated cost assumption (a missed default costs more
      than a false alarm). The chosen threshold and its justification are
      recorded.
- [ ] A model comparison table exists: algorithm, imbalance strategy, ROC-AUC,
      PR-AUC, precision, recall, F1, chosen threshold.
- [ ] `quality_gate.py` defines hard thresholds; a model failing them is not
      promoted, and this is demonstrated — not merely coded.
- [ ] Champion registered in MLflow and promoted to the `@production` alias.
- [ ] The API loads the production model by alias; the container is rebuilt and
      verified.
- [ ] Tests: the quality gate rejects a deliberately bad metrics dict and
      accepts a good one; threshold application maps probabilities to labels
      correctly at the boundary; the registry wrapper round-trips.

**Final artifact.** Tag `v0.3`, a populated MLflow registry with a `@production`
alias, and `docs/findings/03-model-selection.md` — the primary evidence for the
classical-ML micro-credential.

**Dependencies.** Phase 2's frozen feature set.

**Estimated time.** 4 sessions.

**Known risks / unknowns.**
- Optuna over 300k rows with cross-validation is slow. Decide the trial budget
  up front and subsample for the search if needed — state that you did.
- SMOTE on a wide mixed-type frame is awkward and slow; class weights may
  simply win. That is a legitimate finding, not a failure.
- MLflow registry semantics (aliases, not stages) are easy to get subtly wrong.
  The reference project at `h:\AI kurs\z5\projekt_ml-main\` is the pattern to
  follow.

**Task checklist.**
- [ ] Expand `models.py` with the additional algorithms and baseline params.
- [ ] Extend `evaluation.py` with PR-AUC, confusion matrix, per-class metrics.
- [ ] Train all algorithms, one MLflow run each; build the comparison table.
- [ ] Run the imbalance experiment: resampling vs class weights, same splits.
- [ ] Implement `tune.py`; run the search on the leading algorithm.
- [ ] Implement threshold optimisation; state the cost assumption; pick and
      record the threshold.
- [ ] Implement `quality_gate.py` with justified thresholds.
- [ ] Implement `registry.py`: register, promote by alias, load production.
- [ ] Wire the gate into `train.py` so promotion is conditional.
- [ ] Repoint the API at the registry alias; rebuild; verify.
- [ ] Tests: gate accept/reject, threshold boundary behaviour, registry round-trip.
- [ ] `docs/findings/03-model-selection.md`: comparison table, imbalance
      finding, threshold rationale, champion and why.
- [ ] Merge, tag `v0.3`.

---

# Phase 4 — Neural network

**Phase goal.** Build a PyTorch MLP with categorical embeddings and compare it
honestly against the Phase 3 champion — including the possibility that it loses.

**Scope across the four layers.**
- *EDA* — frozen.
- *Modeling* — **the thick layer.** Embedding sizing, architecture, training
  loop with early stopping, like-for-like evaluation.
- *Implementation* — `src/nn/` containing dataset, model and training loop as
  separate concerns.
- *Deployment* — the API can serve either model family. **This is where the
  Strategy pattern finally earns its place** — do not introduce it earlier.

**Completion criteria (definition of done).**
- [ ] A `Dataset` yields numeric block and categorical indices separately.
- [ ] An MLP with one `nn.Embedding` per categorical feature, sized from the
      cardinalities recorded in Phase 2.
- [ ] Training loop per the course conventions: explicit device selection, Adam
      with weight decay, `ReduceLROnPlateau`, gradient clipping, **early
      stopping on validation loss with best-`state_dict` checkpointing**, and
      the best checkpoint reloaded for the final test evaluation.
- [ ] Evaluated on **the same test split with the same metrics** as Phase 3 —
      otherwise the comparison is meaningless.
- [ ] Logged to MLflow so classical and neural runs sit side by side.
- [ ] The API can serve either model behind one interface, selected by config.
- [ ] Tests: the model's forward pass produces the expected output shape;
      embedding indices stay within bounds for unseen categories; early
      stopping actually halts on a synthetic non-improving run; both strategies
      satisfy the scorer interface.
- [ ] **A written honest verdict.** If the MLP loses to gradient boosting on
      tabular data, say so and explain why — that is the expected result on this
      kind of data and demonstrates more understanding than a rigged win.

**Final artifact.** Tag `v0.4` and `docs/findings/04-neural-network.md` — the
primary evidence for the neural-network micro-credential.

**Dependencies.** Phase 3's champion and metrics (the comparison target) and
Phase 2's cardinality table (the embedding dimensions).

**Estimated time.** 3 sessions.

**Known risks / unknowns.**
- Unseen categories at inference time will index out of bounds unless an
  explicit unknown bucket is reserved. Plan for it; test it.
- Class imbalance needs handling again, differently — a weighted loss rather
  than resampling.
- This phase can absorb infinite tuning time. Per the cut list, one sensible
  architecture reported honestly beats three days of hyperparameter search.

**Task checklist.**
- [ ] `src/nn/dataset.py` — Dataset and DataLoader with numeric/categorical
      separation and an unknown-category bucket.
- [ ] `src/nn/model.py` — embedding-plus-MLP module; hyperparameters as
      commented UPPERCASE constants.
- [ ] `src/nn/train.py` — the training loop with early stopping and
      checkpointing.
- [ ] Train; reload the best checkpoint; evaluate on the Phase 3 test split.
- [ ] Log to MLflow alongside the classical runs.
- [ ] Introduce the scorer interface; adapt both model families to it.
- [ ] Repoint the API at the interface; make the choice config-driven; rebuild.
- [ ] Tests: forward-pass shape, unseen-category safety, early-stopping trigger,
      interface conformance for both strategies.
- [ ] `docs/findings/04-neural-network.md`: architecture, training curve,
      comparison table, and the honest verdict.
- [ ] Merge, tag `v0.4`.

---

# Phase 5 — Implementation

**Phase goal.** Turn working code into production code — SOLID structure, a
real test suite, observability — without changing a single model.

**Scope across the four layers.**
- *EDA* — untouched.
- *Modeling* — **no new modelling.** Any metric change in this phase is a
  regression and must be investigated.
- *Implementation* — **the thick layer.** OOP/SOLID refactor, design patterns
  where they earn their place, full pytest suite, structured logging,
  prediction logging, error handling.
- *Deployment* — `docker-compose` for local orchestration; CI runs the full
  suite with coverage.

**Completion criteria (definition of done).**
- [ ] The refactor is justified in writing, pattern by pattern: what varies,
      why an abstraction, what would break without it. **An unjustified pattern
      is worse than none** — say so explicitly where you chose *not* to
      abstract.
- [ ] Strategy for model swapping (from Phase 4), with the ABC stating its
      contract in the docstring.
- [ ] **The four classes named in `theme.md` exist by those names**:
      `DataLoader`, `Preprocessor`, `ModelTrainer`, `Predictor`. The project
      idea submitted to the instructor names them explicitly, so the code should
      match the proposal rather than quietly diverging from it. Where a phase
      built the behaviour as functions (Phases 1–3 did), this is where it
      becomes a class — and where you justify in writing whether the class
      earns its existence or is ceremony.
- [ ] A `from_settings` classmethod factory builds the object graph from
      configuration and degrades gracefully when optional config is absent.
- [ ] Every module has a single clear responsibility; no module does two jobs.
- [ ] Test suite covers data loading, preprocessing, features, evaluation,
      quality gate, registry, both scorers and every endpoint. Coverage measured
      and reported — the number matters less than knowing it.
- [ ] Structured logging via `logging.getLogger(__name__)`; no stray `print` in
      library code.
- [ ] Request-ID and timing middleware on the API.
- [ ] Prediction logging to `.jsonl` via `BackgroundTasks`, `OSError` caught.
- [ ] Error handling: real HTTP status codes, no unhandled exception reaching
      the client.
- [ ] Metrics identical to Phase 4 — proven by re-running, not assumed.
- [ ] `docker-compose up` brings the service up locally.

**Final artifact.** Tag `v0.5` and `docs/findings/05-architecture.md` — the
primary evidence for the implementation micro-credential. A diagram of module
responsibilities and dependencies belongs here.

**Dependencies.** Phases 3 and 4 complete — you cannot correctly abstract what
varies until you have seen it vary.

**Estimated time.** 3 sessions.

**Known risks / unknowns.**
- This is challenge-area #2. Expect the refactor to feel slower than writing new
  code, and expect to be tempted into over-abstraction. The written
  justification requirement exists specifically to catch that.
- Refactoring without tests first is how metrics silently change. Write the
  characterisation tests *before* moving code.

**Task checklist.**
- [ ] Write characterisation tests capturing current behaviour **before**
      refactoring anything.
- [ ] Define the scorer ABC with its contract in the docstring.
- [ ] Refactor both model families behind it.
- [ ] Introduce the `from_settings` factory; make graceful degradation explicit.
- [ ] Review every module for single responsibility; split what does two jobs.
- [ ] Fill test-suite gaps; measure coverage.
- [ ] Replace `print` with module loggers in all library code.
- [ ] Add request-ID and timing middleware.
- [ ] Add prediction logging via background tasks.
- [ ] Audit error handling and status codes across every endpoint.
- [ ] Re-run training; confirm metrics unchanged.
- [ ] Add `docker-compose.yml`; verify local bring-up.
- [ ] Extend CI to run the full suite with coverage.
- [ ] `docs/findings/05-architecture.md`: module diagram, pattern
      justifications, and the deliberate non-abstractions.
- [ ] Merge, tag `v0.5`.

---

# Phase 6 — Deployment

**Phase goal.** Get the service running on Google Cloud Run, deployed by a
pipeline rather than by hand.

**Scope across the four layers.**
- *EDA / Modeling* — frozen. Code changes in this phase are deploy config only.
- *Implementation* — cloud-readiness only: port from the environment,
  container-appropriate logging, secrets from the platform not from `.env`.
- *Deployment* — **the thick layer.** CI/CD, Artifact Registry, Cloud Run,
  a local Kubernetes cluster for the *klaster* clause, monitoring.

**Completion criteria (definition of done).**
- [ ] The container reads its port from the `PORT` environment variable — Cloud
      Run assigns it and the service must not hardcode it.
- [ ] Secrets come from the platform, never a committed file.
- [ ] The image is built by CI and pushed to Artifact Registry, tagged with the
      commit SHA rather than only `latest`.
- [ ] The pipeline deploys to Cloud Run on merge to `main`.
- [ ] The deployed service answers `/health` and `/score` over HTTPS from a
      public URL — demonstrated with a real request.
- [ ] The model artifact reaches the container by a documented mechanism —
      baked into the image, or fetched at startup. State which and why.
- [ ] Startup time is acceptable; the model still loads once, not per request.
- [ ] Logs are visible in Cloud Logging.
- [ ] The deploy is documented well enough to be repeated from scratch.
- [ ] **The "wdrożenie na klaster" clause is satisfied and the evidence is
      named** — Kubernetes manifests (deployment + service) applied to a
      running local cluster, the service reachable through them, and the whole
      thing documented. Waived only if Marcin confirms Cloud Run alone
      suffices, in which case record his answer as the evidence instead.
- [ ] The same image runs in both targets — Cloud Run and the local cluster —
      with no image-level differences. If they diverge, the container is not
      actually portable and that is worth knowing.
- [ ] Tests: a smoke test runs against the deployed URL after deploy and fails
      the pipeline if the service is unhealthy.

**Final artifact.** A live Cloud Run URL, `docs/findings/06-deployment.md`, and
tag `v0.6`. Primary evidence for the deployment micro-credential.

**Dependencies.** Phase 5's green suite and working container. A GCP project
with billing enabled — **set this up before the phase starts, not during it.**

**Estimated time.** 5 sessions — the largest budget in the plan, *including a
timeboxed 1-session cloud spike*, per the challenge-area mitigation. The fifth
session is the local-cluster work added by the *klaster* decision; it is handed
back if Marcin confirms Cloud Run alone suffices.

**Known risks / unknowns.**
- **Highest-risk phase in the plan.** GCP IAM and service-account permissions
  are the classic multi-hour sink for a first deploy. The spike exists to hit
  that wall early.
- Cold starts with a large scientific-stack image may be slow. Measure it; if it
  is bad, note it as a finding rather than rebuilding the image under deadline.
- Free-tier limits and billing: check quotas before deploying.
- **Consult current documentation via Context7 for Cloud Run, Artifact
  Registry and the GitHub Actions authentication flow before writing the
  config.** These change often and my training data may be stale — do not take
  remembered flags on trust.

**Task checklist.**
- [ ] **Ask Marcin the "klaster" question** and request the infrastructure
      review the brief offers. Do this *first* — a yes on Cloud Run makes the
      local-cluster tasks below optional and hands back a session.
- [ ] Create the GCP project; enable billing, Cloud Run and Artifact Registry.
- [ ] **Spike first:** deploy a hello-world container to Cloud Run by hand.
      Discard it; keep the IAM knowledge.
- [ ] Make the app read `PORT` from the environment.
- [ ] Move secrets to platform-provided configuration.
- [ ] Decide and document how the model artifact reaches the container.
- [ ] Create the Artifact Registry repository.
- [ ] Set up the CI service account and workload identity / key authentication.
- [ ] Extend CI: build, tag with the commit SHA, push to Artifact Registry.
- [ ] Add the deploy job to Cloud Run on merge to `main`.
- [ ] Add the post-deploy smoke test that gates the pipeline.
- [ ] Verify `/health` and `/score` against the public URL.
- [ ] Confirm logs reach Cloud Logging; measure cold-start time.
- [ ] Install a local cluster (kind or minikube); write the Kubernetes
      deployment and service manifests for the same image.
- [ ] Apply them; reach the service through the cluster; capture the evidence.
- [ ] `docs/findings/06-deployment.md`: architecture, the repeat-from-scratch
      runbook, cold-start measurement, the cluster evidence (manifests +
      `kubectl get` output), Marcin's answer on clause 4a, and what the spike
      taught you.
- [ ] Merge, tag `v0.6`.

---

# Phase 7 — Delivery

**Phase goal.** Turn six phases of work into something a reviewer can read,
run, and assess in twenty minutes.

**Scope across the four layers.** No new work in any layer. This phase packages
what exists.

**Completion criteria (definition of done).**
- [ ] `README.md` covers: the problem, the dataset, results, architecture, how
      to run locally, how to run the tests, and the live URL.
- [ ] A final report consolidates the six findings documents into one narrative:
      what was built, what was learned, what was decided and why, what you would
      do differently.
- [ ] Each of the four micro-credentials has an explicit pointer to its
      evidence — the reviewer should not have to hunt for it.
- [ ] A clean clone passes: `uv sync && uv run pytest`.
- [ ] The demo path works: a request to the live URL returns a sensible score.
- [ ] Known limitations stated honestly — a report that claims no weaknesses is
      less credible, not more.
- [ ] Tag `v1.0`.

**Final artifact.** Tag `v1.0`, the final report, and a repository ready to
submit.

**Dependencies.** All prior phases.

**Estimated time.** 2 sessions.

**Known risks / unknowns.**
- Packaging always takes longer than expected; that is why it has its own phase
  and sits in front of the four-day buffer rather than inside it.
- Verify the clean-clone path on a genuinely fresh clone. "It works on my
  machine" fails at exactly this step.

**Task checklist.**
- [ ] Write the README.
- [ ] Consolidate the findings into the final report.
- [ ] Add the micro-credential evidence map.
- [ ] Clean-clone test in a fresh directory.
- [ ] End-to-end demo run against the live service.
- [ ] Write the limitations and future-work section.
- [ ] Final tidy: dead code, stale TODOs, unused dependencies.
- [ ] Tag `v1.0`.

---

## Stretch goals — only if genuinely ahead

Attempt none of these before Phase 7 is done. They are ways to spend surplus
time, not commitments.

- Join Home Credit's auxiliary tables (`bureau`, `previous_application`) with
  aggregate features — the largest realistic accuracy gain available.
- SHAP explanations for individual scoring decisions — highly relevant to credit
  scoring, where adverse-action reasons are a regulatory requirement.
- A drift-monitoring endpoint comparing live input distributions to training.
- A scheduled retraining job.
- A minimal front end for the demo.

---

## Open questions to resolve before Phase 0

- [ ] **Confirm or correct the challenge-area inference** above. The Phase 6
      session budget and the Phase 1 Docker spike both depend on it.
- [ ] Confirm the session cadence assumption (~2 sessions/week). **With the
      buffer now at zero this matters more than it did** — if the real pace is
      lower, redraw the calendar from the session counts and take cut #1
      pre-emptively rather than discovering the shortfall in November.
- [ ] Confirm there is a GitHub remote for this repository — CI/CD in Phase 6
      assumes GitHub Actions, and the repo is currently local-only.

### Resolved

- ~~Which dataset~~ → Home Credit Default Risk, `application_train.csv` only.
  *(2026-08-31)*
- ~~"Wdrożenie na klaster" target~~ → Cloud Run **+** local Kubernetes cluster;
  GKE Autopilot rejected on cost and risk. Marcin asked at the start of Phase 6,
  which can only give a session back. *(2026-08-31)*
- ~~Whether MLflow/Optuna belong, given they are not in the brief~~ → kept.
  They align with the course's own modelling module and materially strengthen
  the comparison evidence; Optuna remains cut #1 if the schedule bites.
  *(2026-08-31)*
