"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _asking_open_by_default(monkeypatch):
    """The live site starts with asking private to the admin; most tests exercise the open site."""
    monkeypatch.setenv("PRIVATE_ASKING", "off")
