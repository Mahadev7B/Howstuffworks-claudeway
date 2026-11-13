"""When email is set up, a grown-up confirms their email with a 6-digit code before asking or subscribing."""

import importlib
import re
import time

import pytest

from lesson_platform import mailer, ratelimit, verification


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name, value in {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "owl@example.com", "SMTP_PASSWORD": "not-real"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("EMAIL_VERIFICATION", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    app_module.askers.reset_memory()
    verification.reset_memory()
    ratelimit.reset()
    sent = []
    monkeypatch.setattr(mailer, "send", lambda to, subject, text: sent.append((to, subject, text)))
    monkeypatch.setattr(app_module, "_track_lesson", lambda e, q, c: ({"title": q, "slides": []}, None))
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    client = app_module.app.test_client()
    client.sent = sent
    yield client, app_module
    verification.reset_memory()
    app_module.askers.reset_memory()
    ratelimit.reset()


def code_from(sent):
    return re.search(r"\b(\d{6})\b", sent[-1][2]).group(1)


def ask(client, token=None, question="Why is the sky blue?"):
    return client.post("/api/lesson", json={"question": question, "email": "mum@example.com",
                                            "parent_consent": True, "email_token": token})


def test_question_needs_a_confirmed_email(site):
    client, _ = site
    response = ask(client)
    assert response.status_code == 401 and response.get_json()["code"] == "verify_required"


def test_code_by_email_then_token_unlocks_questions(site):
    client, _ = site
    assert client.post("/api/email/start", json={"email": "Mum@Example.com", "parent_consent": True}).get_json() == {"ok": True, "required": True}
    assert client.sent[0][0] == "mum@example.com"
    assert "AskLilOwl" in client.sent[0][1]
    token = client.post("/api/email/verify", json={"email": "mum@example.com", "code": code_from(client.sent)}).get_json()["token"]
    assert ask(client, token).status_code == 200
    # A code works only once.
    again = client.post("/api/email/verify", json={"email": "mum@example.com", "code": code_from(client.sent)})
    assert again.status_code == 400


def test_token_is_tied_to_one_email(site):
    client, _ = site
    other = verification.make_token("someone-else@example.com")
    assert ask(client, other).status_code == 401
    assert verification.token_valid("mum@example.com", "garbage") is False
    forged = verification.make_token("mum@example.com")[:-2] + "00"
    assert verification.token_valid("mum@example.com", forged) is False


def test_old_tokens_expire():
    old = verification.make_token("mum@example.com", issued=int(time.time()) - verification.TOKEN_MAX_AGE_S - 60)
    assert verification.token_valid("mum@example.com", old) is False


def test_wrong_codes_lock_after_five_tries(site):
    client, _ = site
    client.post("/api/email/start", json={"email": "mum@example.com", "parent_consent": True})
    right = code_from(client.sent)
    wrong = "000000" if right != "000000" else "111111"
    for _ in range(verification.MAX_ATTEMPTS):
        assert client.post("/api/email/verify", json={"email": "mum@example.com", "code": wrong}).status_code == 400
    assert client.post("/api/email/verify", json={"email": "mum@example.com", "code": right}).status_code == 400


def test_codes_per_email_are_limited(site):
    client, _ = site
    for _ in range(verification.MAX_SENDS_PER_HOUR):
        assert client.post("/api/email/start", json={"email": "mum@example.com", "parent_consent": True}).status_code == 200
    assert client.post("/api/email/start", json={"email": "mum@example.com", "parent_consent": True}).status_code == 429


def test_start_needs_consent_and_reports_mail_trouble(site, monkeypatch):
    client, _ = site
    assert client.post("/api/email/start", json={"email": "mum@example.com"}).get_json()["code"] == "consent_required"

    def broken(*args):
        raise mailer.MailUnavailable("We couldn't send the email just now. Please try again.")

    monkeypatch.setattr(mailer, "send", broken)
    response = client.post("/api/email/start", json={"email": "mum@example.com", "parent_consent": True})
    assert response.status_code == 503 and "couldn't send" in response.get_json()["error"]


def test_subscribe_also_needs_a_confirmed_email(site, monkeypatch):
    client, _ = site
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_PRICE_ID", "price_x")
    response = client.post("/api/subscribe", json={"email": "mum@example.com", "parent_consent": True})
    assert response.status_code == 401 and response.get_json()["code"] == "verify_required"


def test_without_mail_settings_nothing_changes(site, monkeypatch):
    client, _ = site
    monkeypatch.delenv("SMTP_PASSWORD")
    assert ask(client).status_code == 200
    assert client.post("/api/email/start", json={"email": "mum@example.com", "parent_consent": True}).get_json() == {"ok": True, "required": False}


def test_switch_can_turn_verification_off(site, monkeypatch):
    client, _ = site
    monkeypatch.setenv("EMAIL_VERIFICATION", "off")
    assert ask(client).status_code == 200


def test_home_page_has_code_step_when_email_is_set_up(site, monkeypatch):
    client, app_module = site
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = client.get("/").data
    assert b'data-verify="1"' in page and b'id="askerCode"' in page
    assert b"6-digit code" in page


def test_no_js_form_is_refused_when_codes_are_needed(site):
    client, _ = site
    response = client.post("/lesson", data={"question": "Why?", "email": "mum@example.com", "parent_consent": "yes"})
    assert response.status_code == 400 and b"JavaScript" in response.data
