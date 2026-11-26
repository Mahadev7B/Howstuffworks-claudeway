"""Keep destructive Postgres tests off every database except a local test one.

These tests read TEST_DATABASE_URL and nothing else. DATABASE_URL is the app's
database and is never consulted here.
"""

import os
from urllib.parse import urlparse


class UnsafeTestDatabase(RuntimeError):
    """TEST_DATABASE_URL is set, but it is not a local database whose name contains 'test'."""


def local_test_database_url() -> str | None:
    """The URL destructive tests may use, or None when TEST_DATABASE_URL is unset.

    Unset means skip. A set URL that is remote, or whose database name does not
    contain 'test', is refused before any connection is opened.
    """
    raw = os.environ.get("TEST_DATABASE_URL")
    if raw is None or not str(raw).strip():
        return None
    url = str(raw).strip()
    host, name = _host_and_database(url)
    if host not in ("localhost", "127.0.0.1") or "test" not in name:
        raise UnsafeTestDatabase(
            "TEST_DATABASE_URL must use host localhost or 127.0.0.1 and a database "
            f"name containing 'test' (host={host!r}, database={name!r})."
        )
    return url


def _host_and_database(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").split("+", 1)[0].lower()
    if scheme not in ("postgres", "postgresql"):
        return "", ""
    host = (parsed.hostname or "").lower()
    name = parsed.path.lstrip("/").split("/", 1)[0]
    return host, name
