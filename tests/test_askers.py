"""Asking your own question needs an email; three questions are free, then a subscription."""

import importlib

import pytest


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app_module = importlib.import_module("app")
    app_module.app.config.update(TESTING=True)
    monkeypatch.setattr(app_module.askers.db, "_pool", None)
    app_module.askers.reset_memory()
    generated = []

    def fake_track(endpoint, question, ctx):
        generated.append(question)
        return {"title": question, "slides": []}, None

    monkeypatch.setattr(app_module, "_track_lesson", fake_track)
    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "_client_context", lambda: {})
    client = app_module.app.test_client()
    client.generated = generated
    yield client
    app_module.askers.reset_memory()


def ask(client, question, email="Parent@Example.com", consent=True):
    return client.post("/api/lesson", json={"question": question, "email": email, "parent_consent": consent})


@pytest.mark.parametrize("email", [None, "", "not-an-email", "a@b", "x" * 250 + "@ex.com"])
def test_question_without_valid_email_is_refused_before_generation(site, email):
    response = ask(site, "Why is the sky blue?", email)
    assert response.status_code == 400
    assert response.get_json()["code"] == "email_required"
    assert site.generated == []


def test_three_free_questions_then_subscription_required(site):
    for number, left in ((1, 2), (2, 1), (3, 0)):
        response = ask(site, f"Question number {number}?")
        assert response.status_code == 200
        assert response.get_json()["free_left"] == left
    blocked = ask(site, "Question number 4?")
    assert blocked.status_code == 402
    body = blocked.get_json()
    assert body["code"] == "subscription_required"
    assert "$4.99" in body["error"]
    assert body["free_left"] == 0
    assert len(site.generated) == 3


def test_repeating_a_question_is_free_and_still_allowed_after_limit(site):
    for number in range(3):
        assert ask(site, f"Question {number}?").status_code == 200
    again = ask(site, "question 0?")
    assert again.status_code == 200
    assert again.get_json()["free_left"] == 0


def test_email_is_case_insensitive_and_counted_per_person(site):
    for number in range(3):
        assert ask(site, f"Question {number}?", "Parent@Example.com").status_code == 200
    assert ask(site, "Another?", "parent@example.com ").status_code == 402
    assert ask(site, "Another?", "other@example.com").status_code == 200


def test_failed_generation_does_not_use_a_free_question(site, monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "_track_lesson", lambda *a: (None, "boom"))
    assert ask(site, "Will this fail?").status_code == 500
    assert app_module.askers.allowance("parent@example.com", "x")["used"] == 0


def test_database_trouble_refuses_instead_of_generating(site, monkeypatch):
    app_module = importlib.import_module("app")

    def broken(email, question):
        raise app_module.askers.AskerUnavailable("down")

    monkeypatch.setattr(app_module.askers, "allowance", broken)
    response = ask(site, "Why?")
    assert response.status_code == 503
    assert response.get_json()["code"] == "asker_unavailable"
    assert site.generated == []


def test_form_post_without_email_is_refused(site):
    response = site.post("/lesson", data={"question": "Why is the sky blue?"})
    assert response.status_code == 400
    assert b"email" in response.data
    assert site.generated == []


def test_home_form_asks_for_email_without_password(site, monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = site.get("/").data
    assert b'type="email" name="email"' in page
    assert b"$4.99 a month" in page
    assert b'type="password"' not in page


@pytest.mark.parametrize("consent", [None, False, "", "no"])
def test_question_without_parent_consent_is_refused(site, consent):
    response = ask(site, "Why is the sky blue?", consent=consent)
    assert response.status_code == 400
    assert response.get_json()["code"] == "consent_required"
    assert site.generated == []


def test_form_post_needs_consent_too(site):
    response = site.post("/lesson", data={"question": "Why?", "email": "mum@example.com"})
    assert response.status_code == 400
    assert site.generated == []
    ok = site.post("/lesson", data={"question": "Why?", "email": "mum@example.com", "parent_consent": "yes"})
    assert ok.status_code == 200
    assert site.generated == ["Why?"]


def test_home_form_has_consent_checkbox_and_privacy_links(site, monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module.library, "list_published_cards", lambda: [])
    page = site.get("/").data
    assert b'name="parent_consent"' in page
    assert b'href="/privacy"' in page
    assert b"mailto:Asklilowl@gmail.com" in page


def test_privacy_page_names_contact_services_and_rights(site):
    response = site.get("/privacy")
    assert response.status_code == 200
    page = response.data
    for text in (b"Asklilowl@gmail.com", b"Children under 13", b"A parent must always be present", b"Parents' rights", b"Anthropic",
                 b"fal.ai", b"OpenAI", b"ipwho.is", b"Render", b"Neon", b"Google Fonts"):
        assert text in page
    assert b"/privacy" in site.get("/sitemap.xml").data


def test_deleting_an_asker_restores_free_questions(site):
    app_module = importlib.import_module("app")
    for number in range(3):
        assert ask(site, f"Question {number}?").status_code == 200
    assert app_module.askers.delete_asker("parent@example.com") is True
    assert app_module.askers.delete_asker("parent@example.com") is False
    assert ask(site, "Question 9?").status_code == 200
