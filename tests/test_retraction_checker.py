"""Tests for retraction_checker.py (5th trust axis)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.retraction_checker import (
    RetractionResult,
    SCORE_RETRACTED,
    SCORE_CORRECTED,
    SCORE_CLEAN,
    SCORE_UNKNOWN,
    check_retraction,
    check_retraction_batch,
    retraction_score,
    _init_cache,
    _cache_get,
    _cache_put,
)


# ---------------------------------------------------------------------------
# Score constants
# ---------------------------------------------------------------------------

class TestScoreConstants:
    def test_ordering(self):
        assert SCORE_RETRACTED < SCORE_CORRECTED < SCORE_CLEAN
        assert SCORE_RETRACTED == 0.0
        assert SCORE_CLEAN == 1.0
        assert SCORE_UNKNOWN < SCORE_CLEAN


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class TestSqliteCache:
    def test_init_creates_table(self, tmp_path):
        db = tmp_path / "test.db"
        conn = _init_cache(db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert any("retraction_cache" in t[0] for t in tables)
        conn.close()

    def test_cache_roundtrip(self, tmp_path):
        db = tmp_path / "test.db"
        conn = _init_cache(db)
        r = RetractionResult(
            doi="10.1234/test",
            retracted=True,
            corrected=False,
            retraction_type="retraction",
            retraction_date="2022-03-01",
            source="crossref",
            score=SCORE_RETRACTED,
        )
        _cache_put(conn, r)
        cached = _cache_get(conn, "10.1234/test")
        assert cached is not None
        assert cached.retracted is True
        assert cached.source == "cache"
        assert cached.score == SCORE_RETRACTED
        conn.close()

    def test_cache_miss_returns_none(self, tmp_path):
        db = tmp_path / "test.db"
        conn = _init_cache(db)
        assert _cache_get(conn, "10.9999/notexist") is None
        conn.close()

    def test_cache_clean_paper(self, tmp_path):
        db = tmp_path / "test.db"
        conn = _init_cache(db)
        r = RetractionResult(
            doi="10.5678/clean",
            retracted=False,
            corrected=False,
            retraction_type=None,
            retraction_date=None,
            source="crossref",
            score=SCORE_CLEAN,
        )
        _cache_put(conn, r)
        cached = _cache_get(conn, "10.5678/clean")
        assert cached.retracted is False
        assert cached.score == SCORE_CLEAN
        conn.close()


# ---------------------------------------------------------------------------
# Crossref API parsing
# ---------------------------------------------------------------------------

class TestCrossrefParsing:
    def _mock_response(self, message: dict):
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.return_value = {"message": message}
        return resp

    def test_retracted_paper_detected(self, tmp_path):
        # Retraction is signalled via ``updated-by`` (this work WAS retracted).
        msg = {
            "updated-by": [{"type": "retraction", "updated": {"date-time": "2022-03-01T00:00:00Z"}}]
        }
        with patch("requests.Session.get", return_value=self._mock_response(msg)):
            r = check_retraction("10.1234/retracted", cache_db=tmp_path / "c.db")
        assert r.retracted is True
        assert r.retraction_type == "retraction"
        assert r.retraction_date == "2022-03-01"
        assert r.score == SCORE_RETRACTED

    def test_corrected_paper_detected(self, tmp_path):
        msg = {
            "updated-by": [{"type": "correction", "updated": {"date-time": "2021-06-15T00:00:00Z"}}]
        }
        with patch("requests.Session.get", return_value=self._mock_response(msg)):
            r = check_retraction("10.1234/corrected", cache_db=tmp_path / "c.db")
        assert r.corrected is True
        assert r.retracted is False
        assert r.score == SCORE_CORRECTED

    def test_retraction_notice_itself_is_not_retracted(self, tmp_path):
        # A work whose ``update-to`` retracts another paper is the NOTICE, not a
        # retracted paper. It must not be flagged as retracted.
        msg = {
            "type": "journal-article",
            "update-to": [{"type": "retraction", "updated": {"date-time": "2010-02-06T00:00:00Z"}}],
        }
        with patch("requests.Session.get", return_value=self._mock_response(msg)):
            r = check_retraction("10.1234/the-notice", cache_db=tmp_path / "c.db")
        assert r.retracted is False
        assert r.score == SCORE_CLEAN

    def test_clean_paper(self, tmp_path):
        msg = {"type": "journal-article", "updated-by": []}
        with patch("requests.Session.get", return_value=self._mock_response(msg)):
            r = check_retraction("10.1234/clean", cache_db=tmp_path / "c.db")
        assert r.retracted is False
        assert r.corrected is False
        assert r.score == SCORE_CLEAN

    def test_404_is_red_flag_not_clean(self, tmp_path):
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 404
        with patch("requests.Session.get", return_value=resp):
            from src.data import retraction_checker as rc
            r = rc._query_crossref("10.9999/notexist")
        assert r.retracted is False
        assert r.source == "doi_not_found"
        assert r.score == SCORE_UNKNOWN

    def test_api_error_gives_unknown(self, tmp_path):
        import requests
        with patch("requests.Session.get", side_effect=requests.ConnectionError("no network")):
            from src.data import retraction_checker as rc
            r = rc._query_crossref("10.1234/test")
        assert r.source == "unknown"
        assert r.score == SCORE_UNKNOWN


# ---------------------------------------------------------------------------
# Empty / missing DOI
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_doi_returns_unknown(self):
        r = check_retraction("")
        assert r.source == "unknown"
        assert r.score == SCORE_UNKNOWN

    def test_none_doi_via_score_function(self):
        s = retraction_score(None)
        assert s == SCORE_UNKNOWN

    def test_doi_case_insensitive(self, tmp_path):
        msg = {"type": "journal-article", "update-to": []}
        with patch("requests.Session.get") as mock_get:
            resp = MagicMock()
            resp.ok = True
            resp.status_code = 200
            resp.json.return_value = {"message": msg}
            mock_get.return_value = resp
            # Store with uppercase
            r = check_retraction("10.1234/UPPER", cache_db=tmp_path / "c.db")
            # Retrieve with lowercase — should hit cache
            db = tmp_path / "c.db"
            from src.data.retraction_checker import _init_cache as _ic, _cache_get as _cg
            conn = _ic(db)
            cached = _cg(conn, "10.1234/upper")
            assert cached is not None
            conn.close()


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_empty(self):
        results = check_retraction_batch([])
        assert results == []

    def test_batch_mixed(self, tmp_path):
        msg_retracted = {
            "updated-by": [{"type": "retraction", "updated": {"date-time": "2020-01-01T00:00:00Z"}}]
        }
        msg_clean = {"updated-by": []}

        responses = [
            MagicMock(ok=True, status_code=200, json=MagicMock(return_value={"message": msg_retracted})),
            MagicMock(ok=True, status_code=200, json=MagicMock(return_value={"message": msg_clean})),
        ]
        with patch("requests.Session.get", side_effect=responses):
            results = check_retraction_batch(
                ["10.1234/retracted", "10.5678/clean"],
                cache_db=tmp_path / "c.db",
            )
        assert len(results) == 2
        assert results[0].retracted is True
        assert results[1].retracted is False

    def test_batch_with_empty_doi(self, tmp_path):
        results = check_retraction_batch(["", ""], cache_db=tmp_path / "c.db")
        assert all(r.score == SCORE_UNKNOWN for r in results)
