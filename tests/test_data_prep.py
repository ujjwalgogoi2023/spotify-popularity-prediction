import numpy as np
import pandas as pd

from data_prep import deduplicate, engineer_audio, feature_columns, split


def test_split_partitions_are_disjoint_with_expected_sizes(full):
    tr, va, te = split(full)
    assert len(full) == 89_578
    assert (len(tr), len(va), len(te)) == (62_704, 13_437, 13_437)
    ids = pd.concat([tr.track_id, va.track_id, te.track_id])
    assert ids.is_unique, "a track_id appears in more than one partition"


def test_deduplicate_one_row_per_track(raw_sample):
    assert raw_sample.track_id.duplicated().any(), "sample should contain duplicates"
    out = deduplicate(raw_sample)
    assert out.track_id.is_unique
    assert set(out.track_id) == set(raw_sample.track_id)
    assert (out.n_genres >= 1).all()


def test_key_encoding_on_unit_circle():
    df = pd.DataFrame({"key": np.arange(12), "duration_ms": 200_000,
                       "speechiness": 0.1, "instrumentalness": 0.0,
                       "liveness": 0.2, "acousticness": 0.5})
    out = engineer_audio(df)
    np.testing.assert_allclose(out.key_sin ** 2 + out.key_cos ** 2, 1.0)


def test_feature_columns_without_explicit():
    for include_genre in (False, True):
        cols = feature_columns(include_genre=include_genre, include_explicit=False)
        assert "explicit" not in cols
    assert "explicit" in feature_columns()
