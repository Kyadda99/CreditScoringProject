# Phase 0 — Foundation & data contract

**Status:** ⚠️ **Incomplete — blocked on Kaggle credentials.** Everything that
does not require the dataset is done and green. The data contract is *written
but unverified*.

**Date:** 2026-08-31 · **Branch:** `phase-0-foundation` · **Tag:** not yet cut

---

## What is done

| Deliverable | State |
|---|---|
| `uv sync` reproduces the env from committed `uv.lock` | ✅ |
| `uv run python verify_env.py` reports versions, GPU, dataset presence | ✅ exits 1 when data absent |
| `uv run pytest` green | ✅ 8 passed, 10 skipped |
| Contract test fails loudly on a swapped/truncated CSV | ⚠️ written, never executed |
| `[project.scripts]` entrypoint (`cs-download`) | ✅ |
| CI runs lint + tests on push | ⚠️ committed, not yet observed green |
| `.gitignore` covers data / MLflow / artefacts | ✅ verified with `git check-ignore` |

## What is blocked

`uv run cs-download` fails with **`User is not authenticated`**. There is
no `~/.kaggle/kaggle.json` and no `KAGGLE_USERNAME` / `KAGGLE_KEY` on this
machine. Home Credit is a *competition* dataset, so two separate things are
required and both are manual:

1. An API token from <https://www.kaggle.com/settings> → *Create New Token*,
   saved as `~/.kaggle/kaggle.json`.
2. **Accepting the competition rules in a browser** at
   <https://www.kaggle.com/competitions/home-credit-default-risk/rules>.
   Without this the API returns 403 *even with valid keys* — this is the step
   that costs people half an hour.

Until then the seven contract tests skip rather than fail, by design
(`requires_data` marker, `tests/conftest.py`).

## Decisions taken, and why

**Flat modules under `src/`, imported by bare name.** `CLAUDE.md` and `PLAN.md`
disagreed: the former specifies `pythonpath = ["src", "tests"]` and
`where = ["src"]`, the latter says `uv run python -m src.train` — which only
works if `src` is itself a package, and that makes the setuptools line
meaningless. Settled by looking at the instructor's own
`Deployment_REST/chatbot_project`, which uses `py-modules = ["api", "schemas",
…]` for flat modules at the src root. Consequence: **runnable commands come
from `[project.scripts]`, not `python -m src.x`** — which is what `CLAUDE.md`
asks for anyway. `PLAN.md`'s `python -m src.train` should be read as
"the training entrypoint".

**Feature groups are derived, not enumerated.** Typing 122 column names by hand
gives a list that silently rots. `config.split_feature_groups(df)` computes the
three groups from dtypes and cardinality, and `config.py` stays importable with
no data on disk — which is what lets CI run at all.

**The group split branches on `is_numeric_dtype`, not `dtype == "object"`.**
This one was nearly a silent bug. `uv sync` resolved **pandas 3.0.5**, and
pandas 3.0 infers a dedicated `str` dtype for text columns instead of `object`
([migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html)).
An `object` test would have routed *every* categorical column into the numeric
group, and it would not have blown up here — it would have blown up in Phase 1
inside the encoder, far from the cause. `test_text_columns_never_land_in_numeric`
pins the behaviour.

**No `pandas<3` pin.** The lockfile already guarantees reproducibility, so a
defensive upper bound would only be a guess about ecosystem breakage not yet
observed. Revisit if Phase 3/4 hits it — one line in `pyproject.toml`.

**Dependencies stay minimal.** pandas + kagglehub only. scikit-learn lands in
Phase 1, PyTorch in Phase 4. `verify_env.py` reports GPU status *only if* torch
is importable, so Phase 0 does not pull a ~2.5 GB wheel to answer a question it
does not yet need to ask.

## The contract, as asserted (PROVISIONAL)

`tests/test_data_contract.py` asserts the following. **These numbers come from
the dataset's public description, not from a measurement on this machine.**
`PLAN.md` is explicit that its figures "must be verified, not trusted" — this
table is exactly what is not yet verified.

| Property | Asserted | Verified? |
|---|---|---|
| Rows | 307,511 | ❌ |
| Column names | `tests/fixtures/column_manifest.json` | ❌ not yet generated |
| dtype kinds of 6 key columns | `SK_ID_CURR`/`TARGET`/`DAYS_BIRTH`/`CNT_CHILDREN` int, `AMT_*` float | ❌ |
| Columns | 122 | ❌ |
| Target column | `TARGET`, values `{0, 1}` | ❌ |
| Target positive rate | band `(0.05, 0.12)` | ❌ |
| `SK_ID_CURR` | unique per row | ❌ |
| Identifiers in feature groups | none | ❌ |
| Each of the three groups | non-empty | ❌ |

Also asserted: every column is accounted for (groups + target + id == 122), so
no column can silently vanish from the feature set.

## Environment as found

| Tool | Version | Note |
|---|---|---|
| Python | 3.12.9 | pinned in `.python-version` |
| uv | 0.11.14 | |
| pandas | 3.0.5 | see the dtype decision above |
| numpy | 2.5.2 | |
| Docker | 28.3.3 | installed; **daemon not running** — needed for Phase 1 |
| `gh` CLI | absent | optional |
| Kaggle credentials | absent | **blocker** |

## To close this phase

1. Set up Kaggle credentials and accept the competition rules.
2. `uv run cs-download`
3. `CS_WRITE_MANIFEST=1 uv run pytest` once, to write and commit
   `tests/fixtures/column_manifest.json`.
4. `uv run pytest` — the 10 contract tests must now *run*, not skip.
5. Correct any figure above that the data contradicts, and record what differed.
6. Confirm `git status` shows nothing under `data/` as staged.
7. Merge to `main`, confirm CI is green, tag `v0.0`.

---

## Code review — findings and fixes

A review of this phase raised 5 important items and 8 minor ones. All were
reproduced before being fixed; none were rejected.

**CI's lockfile pin did not exist.** The comment claimed the build would abort
on a stale lock, but `--frozen` only means "do not update the lock" — the flag
that asserts it is `--locked`. Worse, the bare `uv run` steps re-sync and would
have re-locked. Demonstrated by adding a dependency to `pyproject.toml`:
`uv sync --frozen` installed it silently, while `UV_LOCKED=1` refused with
*"the lockfile needs to be updated"*. Now set as a job-level `env`, so every
uv invocation asserts it.

**`load_data` narrowed types before the contract could see the file.**
`astype("int8")` silently wraps — verified: `300` became `44`. Because the
coercion ran inside the loader, the contract tests were asserting the loader's
output rather than the CSV. Now `pd.to_numeric(errors="coerce")` per
`CLAUDE.md`, with no narrowing: `300` survives and unparseable values become
NaN, which `test_coercion_introduced_no_nulls` catches.

**Three of seven contract tests could not fail.**
`test_every_column_is_accounted_for` was true by construction of
`split_feature_groups`, and the "each group is non-empty" tests passed for any
CSV with one number, one string and one two-valued column — a renamed column or
an int→float switch stayed green. Replaced with: a committed column manifest
(exact names, in order), `dtype.kind` assertions on the six columns the rest of
the project stands on, and named-column-to-named-group assertions. Contract
tests went from 7 to 10.

**`FLAG_OWN_CAR` holds `"Y"`/`"N"`, not `0`/`1`.** The docstring used it as the
example of a binary flag and the test fixture encoded the same wrong
assumption, so the fixture agreed with the bug. Corrected throughout;
`FLAG_MOBIL` is the numeric flag.

**The binary rule was `nunique() <= 2`, which is a property of the sample.**
An all-NaN column (0 distinct) and a constant `5.0` column (1 distinct) both
classified as flags. Now the test is on values — a numeric column is binary
only if its non-null values are a non-empty subset of `{0, 1}`.

**Train/serve skew in the group derivation — partially fixed, rest deferred to
Phase 1.** On a one-row frame — exactly what `/score` receives in Phase 6 —
every numeric feature used to route to `binary` and `numeric` came back empty.
The value-based rule above removes most of it, but not all: `CNT_CHILDREN == 0`
on a single row is still indistinguishable from a flag. The complete fix is to
derive the groups **once** on the training frame and persist them with the
model, which needs the Phase 1 pipeline artifact to exist. What landed now:
`FeatureGroups.to_dict()` / `from_dict()` so the groups are serialisable, a
loud warning in the `split_feature_groups` docstring, and
`test_derivation_is_sample_dependent_so_groups_must_be_frozen`, which pins the
divergence so it cannot be forgotten. **Phase 1 must persist the groups
alongside the pipeline and reconstruct them at serve time.**

**`[project.scripts]` was missing** although `CLAUDE.md:130` requires it and two
`main()` functions assumed it. `scripts/download_data.py` also sat outside
`src/`, so it was not importable and could not be wired. Moved to
`src/download_data.py`, added to `py-modules`, exposed as `cs-download`.

Minor items also fixed: the inert `# noqa: BLE001` (BLE is not in ruff's
`select`), `path or DATA_PATH` → `path is None`, a `permissions: contents: read`
block, `cancel-in-progress` no longer cancelling runs on `main`, and a README
that is now a usable quick start. Removed `verify_env.py`'s `sys.path` insert,
which was unnecessary once the package is installed via `py-modules` and
implied otherwise.
