"""Testy budowania preprocessora — bez dostępu do prawdziwych danych."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from config import FeatureGroups
from preprocessing import build_preprocessor


@pytest.fixture
def groups() -> FeatureGroups:
    return FeatureGroups(
        numeric=("AMT_INCOME_TOTAL", "CNT_CHILDREN"),
        categorical=("NAME_CONTRACT_TYPE",),
        binary=("FLAG_MOBIL",),
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [1000.0, 2000.0, 3000.0, None],
            "CNT_CHILDREN": [0, 1, 2, 3],
            "NAME_CONTRACT_TYPE": ["Cash", "Revolving", "Cash", None],
            "FLAG_MOBIL": [1, 1, 0, 1],
        }
    )


def test_returns_unfitted_column_transformer(groups: FeatureGroups) -> None:
    pre = build_preprocessor(groups)
    assert isinstance(pre, ColumnTransformer)
    # Niedopasowany transformer nie ma jeszcze atrybutów kończących się na "_".
    assert not hasattr(pre, "transformers_")


def test_covers_exactly_the_group_columns(groups: FeatureGroups) -> None:
    pre = build_preprocessor(groups)
    covered: list[str] = []
    for _name, _transformer, columns in pre.transformers:
        covered.extend(columns)
    assert sorted(covered) == sorted(groups.all_features)


def test_fits_and_transforms(groups: FeatureGroups, frame: pd.DataFrame) -> None:
    pre = build_preprocessor(groups)
    out = pre.fit_transform(frame)
    assert out.shape[0] == len(frame)
    assert np.isfinite(np.asarray(out, dtype=float)).all(), "imputacja zostawiła NaN"


def test_unseen_category_does_not_raise(
    groups: FeatureGroups, frame: pd.DataFrame
) -> None:
    """handle_unknown='ignore' — inaczej /score zwracałoby 500 na nowej wartości."""
    pre = build_preprocessor(groups).fit(frame)
    unseen = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [1500.0],
            "CNT_CHILDREN": [1],
            "NAME_CONTRACT_TYPE": ["Nieznany typ"],
            "FLAG_MOBIL": [1],
        }
    )
    out = pre.transform(unseen)
    assert out.shape[0] == 1


def test_binary_column_is_not_scaled(
    groups: FeatureGroups, frame: pd.DataFrame
) -> None:
    """Flagi 0/1 nie są skalowane — mają zostać sobą."""
    pre = build_preprocessor(groups).fit(frame)
    out = np.asarray(pre.transform(frame), dtype=float)
    binary_column = out[:, -1]
    assert set(np.unique(binary_column)) <= {0.0, 1.0}


# --- Faza 2: gałąź informative-missing (spec D3) ---------------------------


def test_informative_columns_get_their_own_branch() -> None:
    groups = FeatureGroups(numeric=("a", "b"), categorical=(), binary=())
    preprocessor = build_preprocessor(groups, informative_missing=("a",))
    names = [name for name, _, _ in preprocessor.transformers]
    assert "numeric_informative" in names


def test_informative_branch_receives_only_those_columns() -> None:
    groups = FeatureGroups(numeric=("a", "b"), categorical=(), binary=())
    preprocessor = build_preprocessor(groups, informative_missing=("a",))
    by_name = {name: columns for name, _, columns in preprocessor.transformers}
    assert by_name["numeric_informative"] == ["a"]
    assert by_name["numeric"] == ["b"]


def test_informative_branch_emits_an_indicator_column() -> None:
    """Wskaźnik to dodatkowa kolumna — bez niej decyzja D3 jest deklaracją."""
    groups = FeatureGroups(numeric=("a",), categorical=(), binary=())
    preprocessor = build_preprocessor(groups, informative_missing=("a",))
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert preprocessor.fit_transform(frame).shape[1] == 2


def test_no_informative_columns_means_no_extra_branch() -> None:
    """Transza porównawcza bez wskaźników nie może mieć pustego transformera."""
    groups = FeatureGroups(numeric=("a",), categorical=(), binary=())
    preprocessor = build_preprocessor(groups, informative_missing=())
    names = [name for name, _, _ in preprocessor.transformers]
    assert "numeric_informative" not in names


def test_informative_columns_absent_from_groups_are_ignored() -> None:
    """Kolumna usunięta z modelu nie wraca przez INFORMATIVE_MISSING."""
    groups = FeatureGroups(numeric=("a",), categorical=(), binary=())
    preprocessor = build_preprocessor(groups, informative_missing=("zzz",))
    by_name = {name: columns for name, _, columns in preprocessor.transformers}
    assert by_name["numeric"] == ["a"]
    assert "numeric_informative" not in by_name
