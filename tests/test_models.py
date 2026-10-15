import numpy as np
import pandas as pd
import pytest

import models
from models import HurdleRegressor, load_best_params


def test_load_best_params_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "BEST_PARAMS", tmp_path / "missing.json")
    assert load_best_params("regression", "audio only") == {}


def test_hurdle_rejects_target_without_zeros():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    y = pd.Series(rng.integers(1, 100, size=50))
    with pytest.raises(ValueError, match="zero"):
        HurdleRegressor(["a", "b"]).fit(X, y)
