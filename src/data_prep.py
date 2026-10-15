"""
Data preparation for the Spotify popularity study.

Handles the two structural hazards in the raw dataset:

1. Duplicate leakage. The raw file is 114 genres x 1000 tracks, so the same
   `track_id` recurs under several genres (21.3% of rows, 16,641 tracks). Splitting
   before deduplication puts the same song in train and test and inflates every score.

2. Zero-inflation. 14.1% of tracks sit at popularity == 0, and those zeros are
   clustered by genre (jazz 68%, iranian 66%) rather than scattered at random --
   evidence of regional availability gaps at scrape time, not of unpopular songs.
"""

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "dataset.csv"
PROCESSED = ROOT / "data" / "processed"

RANDOM_STATE = 42

# HuggingFace mirror of the Spotify Tracks Dataset, pinned to the revision the study used.
RAW_URL = ("https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/"
           "resolve/635b034f69257814eff850a5c2b3346fe458134f/dataset.csv")
RAW_SHA256 = "b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd"

USER_AGENT = "spotify-popularity-prediction (course project; urllib)"
RETRY_STATUS = {429, 500, 502, 503, 504}

# The 11 audio features Spotify exposed, plus the two track descriptors that are
# intrinsic to the recording rather than to its marketing.
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature", "duration_ms",
]

# Heavily right-skewed on inspection; log1p pulls them toward symmetry.
SKEWED = ["speechiness", "instrumentalness", "liveness", "acousticness"]


def fetch(url: str, attempts: int = 4, backoff: float = 5.0, timeout: int = 60) -> bytes:
    """GET `url`, retrying rate limits, server errors and dropped connections.

    Data hosts throttle the default urllib client (4TU answered a first request with
    429), so requests carry an explicit User-Agent and wait between tries -- for as long
    as `Retry-After` asks when the server says, otherwise 5, 15, 45 s.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == attempts:
                raise
            retry_after = e.headers.get("Retry-After", "") if e.headers else ""
            wait = float(retry_after) if retry_after.isdigit() else backoff * 3 ** (attempt - 1)
            reason = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == attempts:
                raise
            wait, reason = backoff * 3 ** (attempt - 1), type(e).__name__
        print(f"  {reason}; retrying in {wait:.0f}s ({attempt}/{attempts - 1})")
        time.sleep(wait)
    raise AssertionError("unreachable")


def download_raw() -> None:
    """Fetch the raw CSV once and verify it byte-for-byte against the study's copy."""
    if RAW.exists():
        print(f"already present: {RAW}")
        return
    blob = fetch(RAW_URL)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != RAW_SHA256:
        raise ValueError(f"SHA-256 mismatch: got {digest}, expected {RAW_SHA256}")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(blob)
    print(f"downloaded: {RAW}")


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, dropping the unnamed export index."""
    return pd.read_csv(RAW, index_col=0)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per track.

    A track appearing under several genres is one observation, not several. We keep
    the first genre as its label and record how many genres it spanned -- breadth of
    genre assignment is itself plausibly informative, and it preserves information
    that a plain drop_duplicates would discard.
    """
    n_genres = df.groupby("track_id")["track_genre"].nunique().rename("n_genres")
    out = df.drop_duplicates(subset="track_id", keep="first").copy()
    return out.merge(n_genres, left_on="track_id", right_index=True, how="left")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records that are physically impossible rather than merely unusual."""
    bad = (df.tempo == 0) | (df.time_signature == 0) | (df.duration_ms == 0)
    return df.loc[~bad].copy()


def engineer_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms that depend only on the audio columns.

    Split out from `engineer` so an external dataset carrying the same Spotify audio
    features -- but no `explicit` flag or popularity score -- goes through identical
    transforms.
    """
    out = df.copy()

    out["duration_min"] = out["duration_ms"] / 60_000
    out["log_duration"] = np.log1p(out["duration_min"])

    for col in SKEWED:
        out[f"log_{col}"] = np.log1p(out[col])

    # Musical key is cyclic: key 11 is adjacent to key 0, so an ordinal encoding
    # would assert a distance that does not exist. Project onto the unit circle.
    out["key_sin"] = np.sin(2 * np.pi * out["key"] / 12)
    out["key_cos"] = np.cos(2 * np.pi * out["key"] / 12)

    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive modelling features from the raw audio columns."""
    out = engineer_audio(df)

    out["explicit"] = out["explicit"].astype(int)
    out["is_zero_pop"] = (out["popularity"] == 0).astype(int)

    return out


def feature_columns(include_genre: bool = False, include_explicit: bool = True) -> list[str]:
    """Model matrix columns.

    The audio-only set answers RQ1 (how far do audio features go alone); adding genre
    answers RQ3 (how much context contributes). Keeping the two sets separate is what
    makes that contrast interpretable.

    `include_explicit=False` drops the one non-audio flag, for the external dataset,
    which does not carry it.
    """
    cols = [
        "danceability", "energy", "loudness", "mode", "valence", "tempo",
        "time_signature", "explicit", "log_duration", "key_sin", "key_cos",
    ] + [f"log_{c}" for c in SKEWED]

    if not include_explicit:
        cols.remove("explicit")
    if include_genre:
        cols = cols + ["track_genre", "n_genres"]
    return cols


def build(exclude_zeros: bool = False) -> pd.DataFrame:
    """Full preparation pipeline.

    `exclude_zeros` switches between the two analyses the report compares:
    Track A retains the zero mass, Track B drops it as a sensitivity check.
    """
    df = engineer(clean(deduplicate(load_raw())))
    if exclude_zeros:
        df = df.loc[df.popularity > 0].copy()
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """70/15/15 train/val/test, stratified on binned popularity.

    Stratifying on deciles of the target keeps the popularity distribution -- including
    the zero spike -- consistent across all three partitions, so validation scores are
    comparable to test scores.
    """
    strata = pd.cut(df.popularity, bins=[-1, 0, 10, 20, 30, 40, 50, 60, 70, 101],
                    labels=False)

    train, temp = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=strata
    )
    temp_strata = strata.loc[temp.index]
    val, test = train_test_split(
        temp, test_size=0.50, random_state=random_state, stratify=temp_strata
    )

    # The whole point of deduplicating: assert the partitions share no track.
    assert not (set(train.track_id) & set(test.track_id)), "track_id leak: train/test"
    assert not (set(train.track_id) & set(val.track_id)), "track_id leak: train/val"
    assert not (set(val.track_id) & set(test.track_id)), "track_id leak: val/test"

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    download_raw()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    deduped = deduplicate(raw)
    cleaned = clean(deduped)
    df = engineer(cleaned)

    print(f"raw rows          : {len(raw):,}")
    print(f"after dedup       : {len(deduped):,}  (-{len(raw)-len(deduped):,})")
    print(f"after cleaning    : {len(cleaned):,}  (-{len(deduped)-len(cleaned):,})")
    print(f"zero-popularity   : {int(df.is_zero_pop.sum()):,}  ({100*df.is_zero_pop.mean():.1f}%)")

    df.to_csv(PROCESSED / "tracks_clean.csv", index=False)

    tr, va, te = split(df)
    print(f"\nsplit             : {len(tr):,} train / {len(va):,} val / {len(te):,} test")
    print("leakage assertions: PASSED")
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        part.to_csv(PROCESSED / f"{name}.csv", index=False)
    print(f"\nwritten to {PROCESSED}")
