# Credit Scoring

Scoring ryzyka niespłacenia kredytu, prowadzony end-to-end: dane → model →
usługa HTTP → kontener → chmura. Zbiór: [Home Credit Default
Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(`application_train.csv`).

- **Plan realizacji:** [PLAN.md](PLAN.md) — fazy, kryteria ukończenia, harmonogram.
- **Konwencje projektu:** [CLAUDE.md](CLAUDE.md) — układ modułów, wzorce, zasady kodu.
- **Wnioski z faz:** [docs/findings/](docs/findings/).

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

Faza 0 (fundament) — w toku. Postęp fazami: [PLAN.md](PLAN.md).
