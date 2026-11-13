"""Stripe subscriptions: checkout for the parent's email, and signed webhooks that switch access on and off."""

import hashlib
import hmac
import importlib
import json
import time

import pytest

from lesson_platform import billing, ratelimit

SECRET = "whsec_test_secret"


def signed(event: dict) -> tuple[bytes, str]:
    payload = json.dumps(event).encode()
    stamp = str(int(time.time()))
    digest = hmac.new(SECRET.encode(), stamp.encode() + b"." + payload, hashlib.sha256).hexdigest()
    return payload, f"t={stamp},v1={digest}"


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("STRIPE_PRICE_ID", "price_monthly")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", SECRET)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    app_module.askers.reset_memory()
    ratelimit.reset()
    monkeypatch.setattr(app_module, "_track_lesson", lambda e, q, c: ({"title": q, "slides": []}, None))
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    client = app_module.app.test_client()
    yield client, app_module
    app_module.askers.reset_memory()
    ratelimit.reset()


def ask(client, question, email="mum@example.com"):
    return client.post("/api/lesson", json={"question": question, "email": email, "parent_consent": True})


def test_checkout_is_for_the_parents_email_and_monthly_price(site, monkeypatch):
    client, _ = site
    calls = []
    monkeypatch.setattr(billing.stripe.checkout.Session, "create",
                        lambda **kw: calls.append(kw) or {"url": "https://checkout.stripe.com/c/pay/cs_test_1"})
    response = client.post("/api/subscribe", json={"email": " Mum@Example.com ", "parent_consent": True})
    assert response.get_json() == {"ok": True, "url": "https://checkout.stripe.com/c/pay/cs_test_1"}
    kw = calls[0]
    assert kw["mode"] == "subscription"
    assert kw["line_items"] == [{"price": "price_monthly", "quantity": 1}]
    assert kw["customer_email"] == "mum@example.com"
    assert kw["subscription_data"]["metadata"]["asker_email"] == "mum@example.com"
    assert kw["success_url"].endswith("/?subscribed=1")


@pytest.mark.parametrize("body,code", [({"parent_consent": True}, "email_required"),
                                       ({"email": "mum@example.com"}, "consent_required")])
def test_subscribe_needs_email_and_consent(site, body, code):
    client, _ = site
    response = client.post("/api/subscribe", json=body)
    assert response.status_code == 400 and response.get_json()["code"] == code


def test_subscribe_is_off_until_keys_are_set(site, monkeypatch):
    client, _ = site
    monkeypatch.delenv("STRIPE_SECRET_KEY")
    response = client.post("/api/subscribe", json={"email": "mum@example.com", "parent_consent": True})
    assert response.status_code == 503 and response.get_json()["code"] == "billing_unavailable"


def test_paid_checkout_unlocks_questions_and_cancelling_locks_them(site):
    client, _ = site
    for number in range(3):
        assert ask(client, f"Question {number}?").status_code == 200
    assert ask(client, "Question 4?").status_code == 402

    payload, signature = signed({"type": "checkout.session.completed", "data": {"object": {
        "mode": "subscription", "payment_status": "paid", "metadata": {"asker_email": "Mum@Example.com"},
        "customer": "cus_1", "subscription": "sub_1"}}})
    assert client.post("/stripe/webhook", data=payload, headers={"Stripe-Signature": signature}).status_code == 200
    assert ask(client, "Question 4?").status_code == 200

    payload, signature = signed({"type": "customer.subscription.deleted", "data": {"object": {
        "id": "sub_1", "customer": "cus_1", "status": "canceled", "metadata": {"asker_email": "mum@example.com"}}}})
    assert client.post("/stripe/webhook", data=payload, headers={"Stripe-Signature": signature}).status_code == 200
    assert ask(client, "Question 5?").status_code == 402


def test_forged_webhook_is_rejected(site):
    client, app_module = site
    payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {
        "mode": "subscription", "payment_status": "paid", "metadata": {"asker_email": "mum@example.com"}}}}).encode()
    response = client.post("/stripe/webhook", data=payload, headers={"Stripe-Signature": "t=1,v1=forged"})
    assert response.status_code == 400
    assert app_module.askers.allowance("mum@example.com", "x")["subscribed"] is False


def test_past_due_subscription_does_not_unlock(site):
    client, app_module = site
    payload, signature = signed({"type": "customer.subscription.updated", "data": {"object": {
        "id": "sub_2", "status": "past_due", "metadata": {"asker_email": "dad@example.com"}}}})
    assert client.post("/stripe/webhook", data=payload, headers={"Stripe-Signature": signature}).status_code == 200
    assert app_module.askers.allowance("dad@example.com", "x")["subscribed"] is False


def test_home_shows_subscribe_button_and_test_mode_note(site, monkeypatch):
    client, app_module = site
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = client.get("/").data
    assert b'id="subscribeBtn"' in page
    assert b"Stripe test mode" in page
    assert b"Thank you for subscribing" in client.get("/?subscribed=1").data


def test_privacy_page_mentions_stripe(site):
    client, _ = site
    page = client.get("/privacy").data
    assert b"Stripe" in page and b"never sees or stores card numbers" in page
