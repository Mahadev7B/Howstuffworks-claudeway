"""Prove a grown-up owns an email: a 6-digit code by email, then a signed token on the device.

Codes are stored only as keyed hashes, expire after 15 minutes and allow five
tries. Each email can ask for five codes an hour. Verification is switched on
only when outgoing email is configured (see mailer.py), unless
EMAIL_VERIFICATION=off is set.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import threading
import time

from lesson_platform import db, mailer

logger = logging.getLogger(__name__)

CODE_TTL_S = 15 * 60
MAX_ATTEMPTS = 5
MAX_SENDS_PER_HOUR = 5
TOKEN_MAX_AGE_S = 400 * 24 * 3600

_lock = threading.Lock()
_memory_codes: dict[str, dict] = {}


class TooManyCodes(RuntimeError):
    pass


def required() -> bool:
    return os.getenv("EMAIL_VERIFICATION", "").strip().lower() != "off" and mailer.enabled()


def _secret() -> bytes:
    return os.getenv("FLASK_SECRET_KEY", "dev-secret-key").encode("utf-8")


def _code_hash(email: str, code: str) -> str:
    return hmac.new(_secret(), f"email-code|{email}|{code}".encode(), hashlib.sha256).hexdigest()


def make_token(email: str, issued: int | None = None) -> str:
    payload = base64.urlsafe_b64encode(f"{email}|{int(issued or time.time())}".encode()).decode().rstrip("=")
    signature = hmac.new(_secret(), f"email-token|{payload}".encode(), hashlib.sha256).hexdigest()[:40]
    return f"{payload}.{signature}"


def token_valid(email: str, token: str | None) -> bool:
    try:
        payload, signature = str(token or "").split(".", 1)
        expected = hmac.new(_secret(), f"email-token|{payload}".encode(), hashlib.sha256).hexdigest()[:40]
        if not hmac.compare_digest(signature, expected):
            return False
        owner, issued = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode().rsplit("|", 1)
        return hmac.compare_digest(owner, email) and 0 <= time.time() - int(issued) <= TOKEN_MAX_AGE_S
    except (ValueError, UnicodeDecodeError):
        return False


def _email_text(code: str) -> str:
    return (
        "Hello,\n\n"
        f"Your AskLilOwl confirmation code is {code}. It works for 15 minutes.\n\n"
        "Someone typed this email address into the question box on asklilowl.com. "
        "If that wasn't you, you can ignore this email and nothing will happen.\n\n"
        "AskLilOwl · https://www.asklilowl.com\n"
        "Privacy questions: Asklilowl@gmail.com\n"
    )


def start(email: str) -> None:
    """Create a fresh code for this email and send it. Raises TooManyCodes or mailer.MailUnavailable."""
    code = f"{secrets.randbelow(1_000_000):06d}"
    digest = _code_hash(email, code)
    now = time.time()
    if db._pool is None:
        with _lock:
            row = _memory_codes.get(email)
            sends = row["sends"] + 1 if row and now - row["window"] < 3600 else 1
            if sends > MAX_SENDS_PER_HOUR:
                raise TooManyCodes()
            window = row["window"] if row and now - row["window"] < 3600 else now
            _memory_codes[email] = {"hash": digest, "expires": now + CODE_TTL_S, "attempts": 0, "sends": sends, "window": window}
    else:
        with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO email_codes (email, code_hash, expires_at, attempts, sends, window_start)
                VALUES (%s, %s, NOW() + INTERVAL '15 minutes', 0, 1, NOW())
                ON CONFLICT (email) DO UPDATE SET
                  code_hash = EXCLUDED.code_hash,
                  expires_at = EXCLUDED.expires_at,
                  attempts = 0,
                  sends = CASE WHEN email_codes.window_start > NOW() - INTERVAL '1 hour' THEN email_codes.sends + 1 ELSE 1 END,
                  window_start = CASE WHEN email_codes.window_start > NOW() - INTERVAL '1 hour' THEN email_codes.window_start ELSE NOW() END
                RETURNING sends
                """,
                (email, digest),
            )
            if cur.fetchone()[0] > MAX_SENDS_PER_HOUR:
                raise TooManyCodes()
    mailer.send(email, f"Your AskLilOwl code: {code}", _email_text(code))


def check(email: str, code: str) -> str | None:
    """Return a device token when the code is right; otherwise None."""
    code = "".join(ch for ch in str(code or "") if ch.isdigit())
    if len(code) != 6:
        return None
    digest = _code_hash(email, code)
    if db._pool is None:
        with _lock:
            row = _memory_codes.get(email)
            if not row or time.time() > row["expires"] or row["attempts"] >= MAX_ATTEMPTS:
                return None
            if not hmac.compare_digest(row["hash"], digest):
                row["attempts"] += 1
                return None
            del _memory_codes[email]
        return make_token(email)
    with db._pool.connection(timeout=db._POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE email_codes SET attempts = attempts + 1 WHERE email = %s AND expires_at > NOW() "
            "AND attempts < %s RETURNING code_hash",
            (email, MAX_ATTEMPTS),
        )
        row = cur.fetchone()
        if not row or not hmac.compare_digest(row[0], digest):
            return None
        cur.execute("DELETE FROM email_codes WHERE email = %s", (email,))
        cur.execute(
            "INSERT INTO askers (email, consented_at, verified_at) VALUES (%s, NOW(), NOW()) "
            "ON CONFLICT (email) DO UPDATE SET verified_at = NOW()",
            (email,),
        )
    return make_token(email)


def reset_memory() -> None:
    """Test helper."""
    with _lock:
        _memory_codes.clear()
