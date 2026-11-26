"""The destructive-test guard never follows DATABASE_URL."""

import pytest

from tests.pg_guard import UnsafeTestDatabase, local_test_database_url


def test_database_url_alone_is_ignored(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://db.example.com/lilowl")
    assert local_test_database_url() is None


def test_remote_host_is_refused(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://db.example.com/lilowl_test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://127.0.0.1/lilowl_test")
    with pytest.raises(UnsafeTestDatabase):
        local_test_database_url()


def test_non_test_database_name_is_refused(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://127.0.0.1:5432/lilowl")
    with pytest.raises(UnsafeTestDatabase):
        local_test_database_url()
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://localhost/production")
    with pytest.raises(UnsafeTestDatabase):
        local_test_database_url()


def test_localhost_and_loopback_test_databases_are_accepted(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://db.example.com/lilowl")
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://lilowl:secret@localhost:5432/lilowl_test")
    assert local_test_database_url() == "postgresql://lilowl:secret@localhost:5432/lilowl_test"
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql://127.0.0.1:5432/empty_test")
    assert local_test_database_url() == "postgresql://127.0.0.1:5432/empty_test"
