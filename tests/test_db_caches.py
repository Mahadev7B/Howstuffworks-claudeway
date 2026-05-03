"""Tests for in-memory caches in lesson_platform.db (no real DB)."""
import time
from unittest.mock import MagicMock, patch

import lesson_platform.db as db
from lesson_platform.db import question_hash


class TestQuestionHash:
    def test_same_question_same_hash(self):
        assert question_hash("How do rockets fly?") == question_hash("How do rockets fly?")

    def test_case_insensitive(self):
        assert question_hash("How DO rockets FLY?") == question_hash("how do rockets fly?")

    def test_punctuation_normalized(self):
        assert question_hash("How do rockets fly?") == question_hash("How do rockets fly!!!")

    def test_whitespace_normalized(self):
        assert question_hash("How  do   rockets fly") == question_hash("How do rockets fly")

    def test_different_questions_different_hash(self):
        assert question_hash("How do rockets fly?") != question_hash("Why is the sky blue?")


class TestBudgetCache:
    def setup_method(self):
        db._budget_cache = None

    def teardown_method(self):
        db._budget_cache = None

    def test_returns_zero_when_pool_none(self):
        with patch.object(db, "_pool", None):
            assert db.today_spend_usd() == 0.0

    def test_returns_cached_value_within_ttl(self):
        db._budget_cache = (time.time(), 1.23)
        with patch.object(db, "_direct_connect") as mock_connect:
            value = db.today_spend_usd()
            assert value == 1.23
            mock_connect.assert_not_called()

    def test_requeries_after_ttl_expiry(self):
        db._budget_cache = (time.time() - 400.0, 1.23)  # > 300s ago
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [4.56]
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = lambda *a: None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        with patch.object(db, "_direct_connect", return_value=mock_conn):
            value = db.today_spend_usd()
            assert value == 4.56


class TestIPRateCache:
    def setup_method(self):
        db._IP_RATE_CACHE.clear()

    def teardown_method(self):
        db._IP_RATE_CACHE.clear()

    def test_returns_zero_when_no_ip(self):
        assert db.ip_calls_last_hour(None, ("/api/lesson",)) == 0
        assert db.ip_calls_last_hour("", ("/api/lesson",)) == 0

    def test_caches_within_ttl(self):
        db._IP_RATE_CACHE[("1.2.3.4", ("/api/lesson",))] = (time.time(), 7)
        with patch.object(db, "_direct_connect") as mock_connect:
            count = db.ip_calls_last_hour("1.2.3.4", ("/api/lesson",))
            assert count == 7
            mock_connect.assert_not_called()

    def test_requeries_after_ttl_expiry(self):
        db._IP_RATE_CACHE[("1.2.3.4", ("/api/lesson",))] = (time.time() - 60.0, 7)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [12]
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = lambda *a: None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        with patch.object(db, "_direct_connect", return_value=mock_conn):
            count = db.ip_calls_last_hour("1.2.3.4", ("/api/lesson",))
            assert count == 12
