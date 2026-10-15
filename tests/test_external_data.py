import pytest

from data_prep import RAW, load_raw

external_data = pytest.importorskip("external_data")


@pytest.mark.skipif(not RAW.exists() or not any(external_data.EXTERNAL.glob("*.csv")),
                    reason="primary or Hit Predictor data not present")
def test_prepare_removes_all_primary_ids():
    ext = external_data.prepare(verbose=False)
    assert len(ext) > 0
    assert not set(ext.track_id) & set(load_raw().track_id)
