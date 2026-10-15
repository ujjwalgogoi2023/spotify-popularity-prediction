"""External validation data: The Spotify Audio Features Hit Predictor Dataset (1960-2019).

Farooq Ansari, 4TU.ResearchData, DOI 10.4121/uuid:d77e74b0-66bc-47ac-8b25-5796d3084478,
GPL-3.0. About 41,000 tracks, one CSV per decade, each with the same Spotify audio
features as the primary dataset and an independent hit/flop label (`target`).

Two things it offers that the primary dataset cannot:

1. A different collection process and target definition, so the audio-only model can be
   tested on data it was not built from (DA1 section 4.3).
2. A decade for every track, so the era confound -- audio features partly proxying for
   release date -- can be measured rather than only acknowledged (DA1 section 8.5).
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import PROCESSED, ROOT, engineer_audio, load_raw

URL = ("https://data.4tu.nl/file/a3094315-d0d6-4326-8194-1536c709fa93/"
       "127746a2-5dbd-4f7c-a067-1d8b82869cbd")
MD5 = "61404a3cc1b4ca580a9f801398814036"
EXTERNAL = ROOT / "data" / "external" / "hit_predictor"


def download() -> None:
    """Fetch the archive once, verify it, and extract only its CSV files.

    Member names are flattened to their basename, so no archive path can write outside
    `EXTERNAL`.
    """
    if any(EXTERNAL.glob("*.csv")):
        print(f"already present: {EXTERNAL}")
        return
    blob = urllib.request.urlopen(URL, timeout=60).read()
    digest = hashlib.md5(blob).hexdigest()
    if digest != MD5:
        raise ValueError(f"MD5 mismatch: got {digest}, expected {MD5}")
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name.lower().endswith(".csv") and not name.startswith("."):
                (EXTERNAL / name).write_bytes(zf.read(member))
                print(f"extracted: {name}")


def load() -> pd.DataFrame:
    """All decades in one frame, with `decade` parsed from the file name."""
    parts = []
    for f in sorted(EXTERNAL.glob("dataset-of-*.csv")):
        yy = int(f.stem.removeprefix("dataset-of-").removesuffix("s"))
        part = pd.read_csv(f)
        part["decade"] = (1900 if yy >= 60 else 2000) + yy
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["track_id"] = df.uri.str.split(":").str[-1]
    return df


def _song_key(artist: pd.Series, title: pd.Series) -> pd.Series:
    """Lower-cased alphanumerics of lead artist and title, for cross-dataset matching."""
    norm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return norm(artist) + "|" + norm(title)


def prepare(verbose: bool = True) -> pd.DataFrame:
    """Deduplicate, clean, engineer, and remove every track the primary dataset holds.

    The overlap removal is what makes this an external test: a track scored here must
    not have been seen by the model in any form.
    """
    raw = load()

    # A track listed more than once is one observation. If its copies disagree on
    # hit/flop the label is ambiguous, so all copies are dropped.
    labels = raw.groupby("track_id").target.nunique()
    conflicted = labels.index[labels > 1]
    df = raw.loc[~raw.track_id.isin(conflicted)].drop_duplicates("track_id", keep="first")

    bad = (df.tempo == 0) | (df.time_signature == 0) | (df.duration_ms == 0)
    cleaned = df.loc[~bad]

    # Overlap is checked twice: by Spotify ID, and by normalised artist + title, because
    # re-releases and compilations give the same song a new ID. The second pass matters --
    # the songs it catches are disproportionately hits, so leaving them in would flatter
    # the transfer score.
    primary = load_raw()
    by_id = cleaned.track_id.isin(set(primary.track_id))
    primary_keys = set(_song_key(primary.artists.str.split(";").str[0], primary.track_name))
    by_title = _song_key(cleaned.artist.str.split(" Featuring").str[0],
                         cleaned.track).isin(primary_keys) & ~by_id
    overlap = by_id | by_title
    out = engineer_audio(cleaned.loc[~overlap]).reset_index(drop=True)

    if verbose:
        print(f"raw rows               : {len(raw):,}")
        print(f"after dedup            : {len(df):,}  (-{len(raw)-len(df):,}, "
              f"incl. {len(conflicted):,} tracks with conflicting labels)")
        print(f"after cleaning         : {len(cleaned):,}  (-{len(df)-len(cleaned):,})")
        print(f"overlap with primary   : {int(overlap.sum()):,} tracks removed "
              f"({int(by_id.sum()):,} by ID, {int(by_title.sum()):,} by artist + title)")
        print(f"external analysis set  : {len(out):,}  (hit rate {out.target.mean():.1%})")
        print(out.groupby("decade").target.agg(tracks="size", hit_rate="mean")
              .round(3).to_string())
    return out


if __name__ == "__main__":
    download()
    ext = prepare()
    ext.to_csv(PROCESSED / "external_clean.csv", index=False)
    print(f"\nwritten: {PROCESSED / 'external_clean.csv'}")
