"""Download helpers, exercised offline against a fake urlopen."""
import hashlib
import io
import urllib.error

import pytest

import data_prep


def _http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("https://example.test", code, "err", headers, io.BytesIO())


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Replays `outcomes` in order: an exception is raised, bytes are returned."""
    calls, waits = [], []

    def install(*outcomes):
        queue = list(outcomes)

        def urlopen(req, timeout):
            calls.append(req)
            out = queue.pop(0)
            if isinstance(out, Exception):
                raise out
            return io.BytesIO(out)

        monkeypatch.setattr(data_prep.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(data_prep.time, "sleep", waits.append)
        return calls, waits

    return install


def test_fetch_retries_rate_limit_then_succeeds(fake_urlopen):
    calls, waits = fake_urlopen(_http_error(429), _http_error(503, "2"), b"ok")
    assert data_prep.fetch("https://example.test") == b"ok"
    assert len(calls) == 3
    assert waits == [5.0, 2.0]  # backoff, then the server's Retry-After
    assert calls[0].get_header("User-agent") == data_prep.USER_AGENT


def test_fetch_does_not_retry_client_errors(fake_urlopen):
    calls, _ = fake_urlopen(_http_error(404))
    with pytest.raises(urllib.error.HTTPError):
        data_prep.fetch("https://example.test")
    assert len(calls) == 1


def test_fetch_gives_up_after_last_attempt(fake_urlopen):
    calls, _ = fake_urlopen(*[_http_error(429)] * 3)
    with pytest.raises(urllib.error.HTTPError):
        data_prep.fetch("https://example.test", attempts=3)
    assert len(calls) == 3


def test_download_raw_rejects_wrong_bytes(monkeypatch, tmp_path):
    target = tmp_path / "raw" / "dataset.csv"
    monkeypatch.setattr(data_prep, "RAW", target)
    monkeypatch.setattr(data_prep, "fetch", lambda url: b"not the dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        data_prep.download_raw()
    assert not target.exists()


def test_local_raw_matches_pinned_hash():
    if not data_prep.RAW.exists():
        pytest.skip("raw data not present")
    assert hashlib.sha256(data_prep.RAW.read_bytes()).hexdigest() == data_prep.RAW_SHA256
