"""Enforces the project comment rules across the repository.

Rules 2 (English) and 4 (no plan or spec references) are checked mechanically.
Rules 1 and 3 are judgment calls and are checked at review.

Scope is every Python file plus the configuration files and the notebook, since
all of them carry prose a maintainer reads.
"""

from __future__ import annotations

import io
import json
import re
import tokenize
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Text files are discovered, not listed: a hardcoded list cannot catch a file
# nobody remembered to add. Anything matching these patterns is scanned.
DISCOVERY_GLOBS = (
    "*.py",
    "*.toml",
    "*.cfg",
    "*.ini",
    "*.md",
    "*.yml",
    "*.yaml",
    "*.ipynb",
    "Dockerfile*",
    ".dockerignore",
    ".env.example",
    "src/**/*.py",
    "tests/**/*.py",
    ".github/**/*.yml",
    "notebooks/*.ipynb",
    # The deployment tree carries as much hand-written explanation as the
    # source does -- the Kubernetes manifests are mostly comments -- so it is
    # held to the same standard.
    "deploy/**/*.yaml",
    "deploy/**/*.sh",
)

# Directories that hold no hand-written prose.
EXCLUDED_PARTS = frozenset(
    {
        ".venv",
        ".git",
        "__pycache__",
        "build",
        "mlruns",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".superpowers",
        "docs",
        "data",
        "models",
        "egg-info",
    }
)

# Not scanned. CLAUDE.md and PLAN.md are the planning documents themselves, so
# banning plan references in them is meaningless, and both are gitignored.
EXEMPT: frozenset[str] = frozenset({"CLAUDE.md", "PLAN.md"})

# This file is excluded from its own scan: it necessarily contains the Polish
# vocabulary and the reference patterns it searches for.
SELF = Path(__file__).resolve()

DIACRITICS = frozenset("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

# Polish words seen in this project, plus function words, all chosen to have no
# English collision. Diacritics alone are not enough: much of the original
# prose was written without them.
POLISH_WORDS = re.compile(
    r"\b("
    r"jest|nie|sie|ktore|ktory|ktora|zeby|wiec|tego|tym|tej|dla|przez|oraz"
    r"|tylko|juz|czy|moze|musi|trzeba|jesli|gdy|bardzo|dlatego|poniewaz|bez"
    r"|nad|przy|ani|albo|kazdy|kazda|wszystko|wszystkie"
    r"|artefakt\w*|bazowe|brak|braku|braki|brakiem|brakow|brzegowe|cech|cechach|championa|czas|daje"
    r"|estymatora|gotowy|idealne|innego|innym|izolowany|jednego|kolumn\w*"
    r"|konfiguracji|kontrakt\w*|loguje|lokalnego|manifestu|metryki|minimalnie"
    r"|modelu|niezbalansowania|nieznany|odczytu|odsetek|parametry|pasmem"
    r"|pliku|pozytywnej|prawdziwego|przypadki|razy|rejestru|serwisu"
    r"|strategii|takiej|testowym|testu|testy|transformera|treningu|trywialny"
    r"|tutaj|uruchom|ustawienia|uszeregowanie|warstwy|wczytany|zacommituj"
    r"|zamiast|zapisano|zapisu|zapisz|zbiorze|zmiana|zmienia|zwraca"
    r"|identyfikatory|wierszy|podzial|podzialu"
    r")\b",
    re.IGNORECASE,
)

# Plan, spec and phase references. A bare `.md` is not banned: pyproject.toml
# legitimately names README.md. Only planning documents and phase or decision
# ids are.
PLAN_REFERENCES = re.compile(
    r"(spec\s+D\d|decision\s+D\d|Faz[aeyąę]\w*|Etap\s+\d|\bphase\s+\d"
    r"|PLAN\.md|CLAUDE\.md|docs/findings|docs/superpowers"
    r"|\bT\d\b(?=\s*[:.])|\bM\d\b(?=\s*[:.]))",
    re.IGNORECASE,
)


def _discover() -> list[Path]:
    """Returns every text file in the project that carries hand-written prose."""
    found: set[Path] = set()
    for pattern in DISCOVERY_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if not path.is_file():
                continue
            if EXCLUDED_PARTS & set(path.relative_to(PROJECT_ROOT).parts):
                continue
            if any(part.endswith(".egg-info") for part in path.parts):
                continue
            if path.resolve() == SELF:
                continue
            found.add(path)
    return sorted(found, key=lambda p: p.relative_to(PROJECT_ROOT).as_posix())


def _relative(path: Path) -> str:
    """Returns the repo-relative POSIX path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def _python_prose(path: Path) -> list[tuple[int, str]]:
    """Returns every comment and string in a Python file as (line, text).

    String literals are included, not only docstrings: log messages, CLI help,
    exception text and assertion messages are read by people too. The tokenizer
    splits f-strings into FSTRING_MIDDLE pieces, which must be collected
    explicitly or their text is invisible.
    """
    source = path.read_text(encoding="utf-8")
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    wanted = (
        tokenize.COMMENT,
        tokenize.STRING,
        getattr(tokenize, "FSTRING_MIDDLE", -1),
    )
    return [(t.start[0], t.string) for t in tokens if t.type in wanted]


def _notebook_prose(path: Path) -> list[tuple[int, str]]:
    """Returns a notebook's source and every text output, including data payloads.

    `outputs[].data` holds the text/plain and text/html renderings, which are
    just as readable as `outputs[].text` and must be scanned too. Images cannot
    be scanned; `test_notebook_outputs_match_their_source` covers those.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    found: list[tuple[int, str]] = []
    for index, cell in enumerate(nb["cells"]):
        found.append((index, "".join(cell["source"])))
        for output in cell.get("outputs", []):
            found.append((index, "".join(output.get("text", []))))
            for mime, payload in output.get("data", {}).items():
                if mime.startswith("image/"):
                    continue
                found.append(
                    (
                        index,
                        "".join(payload) if isinstance(payload, list) else str(payload),
                    )
                )
    return found


def _text_prose(path: Path) -> list[tuple[int, str]]:
    """Returns every line of a plain-text file.

    Every line, not only comment lines: a placeholder value, a YAML `name:`
    label and the text of a `RUN echo` are all prose someone reads.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return list(enumerate(lines, 1))


def _prose(path: Path) -> list[tuple[int, str]]:
    """Returns the human-readable prose of any scanned file."""
    if path.suffix == ".py":
        return _python_prose(path)
    if path.suffix == ".ipynb":
        return _notebook_prose(path)
    return _text_prose(path)


ALL_FILES = [p for p in _discover() if _relative(p) not in EXEMPT]


@pytest.mark.parametrize("path", ALL_FILES, ids=_relative)
def test_prose_is_english(path: Path) -> None:
    offenders = [
        f"{_relative(path)}:{line}: {text.strip()[:70]}"
        for line, text in _prose(path)
        if any(c in DIACRITICS for c in text) or POLISH_WORDS.search(text)
    ]
    assert not offenders, "Polish prose found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", ALL_FILES, ids=_relative)
def test_prose_has_no_plan_references(path: Path) -> None:
    offenders = [
        f"{_relative(path)}:{line}: {text.strip()[:70]}"
        for line, text in _prose(path)
        if PLAN_REFERENCES.search(text)
    ]
    assert not offenders, "Plan or spec reference found:\n" + "\n".join(offenders)


def test_discovery_found_the_expected_files() -> None:
    """Discovery returning little would make every other check vacuous."""
    names = {_relative(p) for p in ALL_FILES}
    for expected in (
        "pyproject.toml",
        "Dockerfile",
        ".dockerignore",
        ".env.example",
        "README.md",
        "verify_env.py",
        ".github/workflows/ci.yml",
        "notebooks/01-eda.ipynb",
        "src/api/app.py",
        "tests/conftest.py",
    ):
        assert expected in names, f"discovery missed {expected}"
    assert len(names) > 55, f"discovery found only {len(names)} files"


def test_notebook_outputs_match_their_source() -> None:
    """Stored outputs must come from the committed code, not from hand editing.

    Figures cannot be text-scanned, so a hand-edited notebook can show Polish
    axis labels while the code above says English. Every `print` literal in a
    cell's source must appear in that cell's stored output; a cell whose text
    output is current is strong evidence its figures were regenerated too.
    """
    import json

    nb = json.loads((PROJECT_ROOT / "notebooks/01-eda.ipynb").read_text("utf-8"))
    literal = re.compile(r'print\(f?"([^"{}]{8,})"')
    stale: list[str] = []
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        rendered = "".join(
            "".join(o.get("text", []))
            + "".join(
                "".join(v) if isinstance(v, list) else str(v)
                for m, v in o.get("data", {}).items()
                if not m.startswith("image/")
            )
            for o in cell.get("outputs", [])
        )
        for raw in literal.findall("".join(cell["source"])):
            text = raw.replace(chr(92) + "n", "").strip()
            if text and text not in rendered:
                stale.append(f"cell {index}: {text!r} is not in the stored output")
    assert not stale, "Notebook outputs are stale; re-run it:\n" + "\n".join(stale)


def test_the_detector_actually_detects() -> None:
    """Pins the detectors, so a broken regex cannot quietly pass everything."""
    assert POLISH_WORDS.search("Zwraca wersje dystrybucji")
    assert POLISH_WORDS.search("brak manifestu kolumn")
    assert POLISH_WORDS.search("Testy warstwy HTTP")
    assert any(c in DIACRITICS for c in "Ładuję model")
    assert PLAN_REFERENCES.search("see spec D4")
    assert PLAN_REFERENCES.search("Faza 2 introduced this")
    assert PLAN_REFERENCES.search("Phase 4 will size it")
    assert PLAN_REFERENCES.search("described in PLAN.md")
    for clean in (
        "Return the installed distribution version, or None.",
        "Each feature group gets its own pipeline.",
        "The threshold comes from out-of-fold predictions, never from test.",
        'readme = "README.md"',
        "A missing artifact is fatal and stops the container from starting.",
        "Reaches the model and moves the score.",
    ):
        assert not POLISH_WORDS.search(clean), clean
        assert not PLAN_REFERENCES.search(clean), clean
