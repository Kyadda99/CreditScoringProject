"""Row hygiene tests, without access to real data.

A separate file because `test_data_contract.py` has a module-level
`requires_data` mark. These run on synthetic frames and must be green in CI,
where the dataset is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data import drop_unusable_rows


def test_rows_without_a_target_are_dropped() -> None:
    frame = pd.DataFrame({"TARGET": [0.0, np.nan, 1.0], "CODE_GENDER": ["M", "F", "F"]})
    assert len(drop_unusable_rows(frame)) == 2


def test_xna_gender_rows_are_dropped() -> None:
    """Four rows in 307 thousand; the "XNA" category is missing, not a gender."""
    frame = pd.DataFrame({"TARGET": [0.0, 1.0], "CODE_GENDER": ["M", "XNA"]})
    assert drop_unusable_rows(frame)["CODE_GENDER"].tolist() == ["M"]


def test_hygiene_is_a_no_op_on_clean_data() -> None:
    frame = pd.DataFrame({"TARGET": [0.0, 1.0], "CODE_GENDER": ["M", "F"]})
    pd.testing.assert_frame_equal(drop_unusable_rows(frame), frame)


def test_hygiene_tolerates_a_frame_without_those_columns() -> None:
    """A request frame has neither the target nor, possibly, the gender column."""
    frame = pd.DataFrame({"AMT_CREDIT": [1.0]})
    assert len(drop_unusable_rows(frame)) == 1
