"""Funkcje analityczne EDA — liczą, nie rysują.

Notebook jest cienkim sterownikiem nad tym modułem (spec D6). Powód: to, co
mieszka w notebooku, nie jest testowalne, a liczby przepisane stamtąd do
`docs/findings/02-eda.md` starzeją się przy pierwszym ponownym uruchomieniu.
Tutaj każda tabela powstaje z kodu, który CI wykonuje na ramkach syntetycznych.

Żadna funkcja nie czyta z dysku, nie rysuje i nie modyfikuje wejścia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TARGET


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Braki danych per kolumna, od najgorszej.

    Args:
        df: Dowolna ramka.

    Returns:
        Ramka z kolumnami `column`, `dtype`, `n_missing`, `pct_missing`,
        posortowana malejąco po `pct_missing`.
    """
    missing = df.isna().sum()
    report = pd.DataFrame(
        {
            "column": missing.index,
            "dtype": [str(df[c].dtype) for c in missing.index],
            "n_missing": missing.to_numpy(),
            "pct_missing": (missing.to_numpy() / max(len(df), 1)) * 100.0,
        }
    )
    return report.sort_values("pct_missing", ascending=False).reset_index(drop=True)


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Liczność kategorii dla kolumn nieliczbowych.

    Faza 4 wymiaruje z tej tabeli `nn.Embedding` — dlatego trafia ona do pliku
    findings, a nie tylko na ekran notebooka.

    Args:
        df: Dowolna ramka.

    Returns:
        Ramka `column`, `n_unique`, `top_value`, `top_share` (udział
        najczęstszej wartości wśród wartości nie-NaN), malejąco po `n_unique`.
    """
    rows: list[dict[str, object]] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        counts = series.value_counts(dropna=True)
        rows.append(
            {
                "column": column,
                "n_unique": int(counts.size),
                "top_value": counts.index[0] if counts.size else None,
                "top_share": (
                    float(counts.iloc[0] / counts.sum()) if counts.size else np.nan
                ),
            }
        )
    report = pd.DataFrame(
        rows, columns=["column", "n_unique", "top_value", "top_share"]
    )
    return report.sort_values("n_unique", ascending=False).reset_index(drop=True)


def target_rate_by_bin(
    df: pd.DataFrame, column: str, bins: int = 10, target: str = TARGET
) -> pd.DataFrame:
    """Udział klasy pozytywnej w kwantylowych koszykach kolumny.

    Kwantyle, nie równe szerokości: rozkłady kwot w tym zbiorze są skrajnie
    skośne, a przy równych szerokościach 99% wierszy wpada do pierwszego
    koszyka i wykres nic nie pokazuje.

    Args:
        df: Ramka zawierająca `column` i `target`.
        column: Nazwa kolumny liczbowej.
        bins: Liczba koszyków.
        target: Nazwa kolumny celu.

    Returns:
        Ramka `bin`, `n`, `target_rate` — jeden wiersz na koszyk.
    """
    buckets = pd.qcut(df[column], q=bins, duplicates="drop")
    grouped = df.groupby(buckets, observed=True)[target].agg(["size", "mean"])
    return pd.DataFrame(
        {
            "bin": grouped.index.astype(str),
            "n": grouped["size"].to_numpy(),
            "target_rate": grouped["mean"].to_numpy(),
        }
    ).reset_index(drop=True)


def correlation_screen(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Pary kolumn liczbowych skorelowane powyżej progu.

    Służy do wyłapania redundancji — w tym zbiorze bloki `_AVG` / `_MEDI` /
    `_MODE` opisują to samo trzy razy.

    Args:
        df: Ramka; kolumny nieliczbowe są pomijane.
        threshold: Próg wartości bezwzględnej korelacji Pearsona.

    Returns:
        Ramka `left`, `right`, `correlation`, malejąco. Każda para raz —
        brany jest górny trójkąt macierzy, nie cała macierz.
    """
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    stacked = upper.stack()
    hits = stacked[stacked >= threshold].sort_values(ascending=False)
    return pd.DataFrame(
        {
            "left": [pair[0] for pair in hits.index],
            "right": [pair[1] for pair in hits.index],
            "correlation": hits.to_numpy(),
        }
    )
