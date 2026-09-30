"""EDA analysis functions: they compute, they do not plot.

The notebook is a thin driver over this module. Every table comes from code,
which CI exercises on synthetic frames.

No function reads from disk, draws, or modifies its input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TARGET


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column, worst first.

    Args:
        df: Any frame.

    Returns:
        A frame of `column`, `dtype`, `n_missing`, `pct_missing`, sorted
        descending by `pct_missing`.
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
    """Category counts for non-numeric columns.

    Intended for sizing categorical embeddings; numeric columns are skipped.

    Args:
        df: Any frame.

    Returns:
        A frame of `column`, `n_unique`, `top_value`, `top_share` (share of the
        most frequent value among non-NaN values), descending by `n_unique`.
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
    """Positive-class rate across quantile bins of a column.

    Quantiles, not equal widths: the amount distributions are highly skewed,
    and equal widths put 99% of rows in the first bin.

    Args:
        df: Frame containing `column` and `target`.
        column: Name of the numeric column.
        bins: Number of bins.
        target: Name of the target column.

    Returns:
        A frame of `bin`, `n`, `target_rate`, one row per bin.
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
    """Numeric column pairs correlated above a threshold.

    Finds redundancy; in this dataset the `_AVG` / `_MEDI` / `_MODE` blocks
    describe the same quantity three times.

    Args:
        df: Frame; non-numeric columns are skipped.
        threshold: Absolute Pearson correlation threshold.

    Returns:
        A frame of `left`, `right`, `correlation`, descending. Each pair once,
        taken from the upper triangle.
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
