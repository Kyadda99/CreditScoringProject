# Credit Scoring

Scoring ryzyka niespłacenia kredytu, prowadzony end-to-end: dane → model →
usługa HTTP → kontener → chmura. Zbiór: [Home Credit Default
Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(`application_train.csv`).

Dokumentacja robocza (`PLAN.md`, `CLAUDE.md`, `docs/`) jest trzymana lokalnie
i celowo nieśledzona przez git — plan faz, konwencje kodu i wnioski z kolejnych
etapów nie wchodzą do repozytorium.

## Szybki start

```bash
uv sync                      # odtwarza środowisko z uv.lock
uv run python verify_env.py  # wersje bibliotek, GPU, obecność danych
uv run cs-download           # pobiera application_train.csv do data/raw/
uv run pytest                # testy jednostkowe + kontrakt danych
```

`uv run cs-download` wymaga poświadczeń Kaggle (`~/.kaggle/kaggle.json`) **oraz**
zaakceptowanego w przeglądarce [regulaminu
konkursu](https://www.kaggle.com/competitions/home-credit-default-risk/rules) —
bez tego drugiego API zwraca 403 mimo poprawnych kluczy.

Bez danych testy kontraktu (`requires_data`) pomijają się same, więc `uv run
pytest` przechodzi też na czystym klonie i w CI.

## Stan

Faza 0 (fundament) — w toku.
