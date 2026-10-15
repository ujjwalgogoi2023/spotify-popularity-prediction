import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from data_prep import RAW, build  # noqa: E402



@pytest.fixture(scope="session")
def raw_sample():
    """A few thousand raw rows, enough to contain repeated track_ids."""
    if not RAW.exists():
        pytest.skip(f"raw data not present: {RAW}")
    import pandas as pd
    return pd.read_csv(RAW, index_col=0, nrows=5_000)


@pytest.fixture(scope="session")
def full():
    """The full prepared dataset; needed only where exact row counts are asserted."""
    if not RAW.exists():
        pytest.skip(f"raw data not present: {RAW}")
    return build()
