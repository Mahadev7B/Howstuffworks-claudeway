"""Kid questions and client addresses are not written to logs or exception text."""

import importlib
import logging
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from lesson_platform import ai_generation, ratelimit, spam_log
from lesson_platform import db
from lesson_platform import image_gen
from lesson_platform.config import load_settings
from lesson_platform.db import lesson_log_id, question_hash

PII = "My name is Jane Doe, email jane.doe@example.com, phone 555-014-2298. Why do leaves change color?"
NORMAL = "Why is the sky blue?"
QUESTIONS = (PII, NORMAL)
FORWARDED = "203.0.113.44"
CLOUD_HEADER_IP = "203.0.113.45"
REMOTE = "198.51.100.46"
ROOT = Path(__file__).resolve().parents[1]


def _needles(question):
    found = [question, question[:40], question[:80], question[:120]]
    if "jane.doe@example.com" in question:
        found.extend(["Jane Doe", "jane.doe@example.com", "555-014-2298", "leaves change color"])
    return found


def _assert_hidden(question, caplog, *blobs):
    text = caplog.text + "\n".join(str(blob) for blob in blobs)
    for needle in _needles(question):
        assert needle not in text
    for address in (FORWARDED, CLOUD_HEADER_IP, REMOTE):
        assert address not in text


def _lesson(question, explanation="Leaves rest in winter."):
    slide = {
        "number": 1, "title": "Color", "explanation": explanation, "fun_fact": "A fact.",
        "narration": "Listen.", "image_prompt": "A tree in autumn.",
    }
    return {"title": question, "subject": "Nature", "slides": [dict(slide) for _ in range(4)]}


def _headers():
    return {"X-Forwarded-For": FORWARDED, "CF-Connecting-IP": CLOUD_HEADER_IP}


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    module = importlib.import_module("app")
    module.app.config.update(TESTING=True)
    module._RECENT_LESSONS.clear()
    ratelimit.reset()
    module.askers.reset_memory()
    yield module
    module._RECENT_LESSONS.clear()
    ratelimit.reset()
    module.askers.reset_memory()


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return load_settings()


def _drafts(question):
    wrong = _lesson(question, explanation=f"{question} 1/4 + 1/2 = 2/6")
    return [wrong, wrong]


@pytest.mark.parametrize("question", QUESTIONS)
def test_generation_logs_the_lesson_id_not_the_question(question, settings, monkeypatch, caplog):
    drafts = _drafts(question)

    def ask(client, model, request):
        assert question in request  # The model still receives the question. The log must not.
        return drafts.pop(0), 10, 20

    monkeypatch.setattr(ai_generation, "Anthropic", lambda api_key: object())
    monkeypatch.setattr(ai_generation, "_ask_for_lesson", ask)
    caplog.set_level(logging.DEBUG)
    result = ai_generation.generate_lesson(question, settings)
    _assert_hidden(question, caplog, result.get("meta"))
    assert lesson_log_id(question) in caplog.text
    assert result["meta"]["math_retried"] is True


@pytest.mark.parametrize("question", QUESTIONS)
def test_provider_errors_do_not_echo_the_question(question, settings, monkeypatch, caplog):
    def ask(client, model, request):
        raise RuntimeError(f"provider echoed {question}")

    monkeypatch.setattr(ai_generation, "Anthropic", lambda api_key: object())
    monkeypatch.setattr(ai_generation, "_ask_for_lesson", ask)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(RuntimeError) as caught:
        ai_generation.generate_lesson(question, settings)
    rendered = "".join(traceback.format_exception(caught.value))
    _assert_hidden(question, caplog, caught.value, rendered)
    assert str(caught.value) == "Lesson generation failed"
    assert lesson_log_id(question) in caplog.text


@pytest.mark.parametrize("question", QUESTIONS)
def test_lesson_paths_log_only_the_lesson_id(question, app_module, monkeypatch, caplog):
    calls = []
    real_attach = app_module._attach_slide_images
    monkeypatch.setattr(app_module, "record_api_call", lambda **kw: calls.append(kw))
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "delete_cached_lesson", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0.0)
    monkeypatch.setattr(app_module, "db_enabled", True)
    caplog.set_level(logging.DEBUG)
    ctx = {"city": None, "region": None, "country": None, "user_agent": "test"}

    app_module._recent_put(question, _lesson(question))
    cached, err = app_module._track_lesson("/api/lesson", question, ctx)
    assert err is None and cached["title"] == question
    app_module._RECENT_LESSONS.clear()

    stored = _lesson(question)
    for slide in stored["slides"]:
        slide["image_data_url"] = "data:image/webp;base64,AA"
    monkeypatch.setattr(app_module, "get_cached_lesson", lambda *a, **k: stored)
    monkeypatch.setattr(app_module, "_attach_slide_images", lambda *a, **k: None)
    cached, err = app_module._track_lesson("/api/lesson", question, ctx)
    assert err is None

    def add_pictures(lesson, ctx=None):
        for slide in lesson.get("slides", []):
            slide["image_data_url"] = "data:image/webp;base64,AA"

    app_module._RECENT_LESSONS.clear()
    monkeypatch.setattr(app_module, "get_cached_lesson", lambda *a, **k: _lesson(question))
    monkeypatch.setattr(app_module, "_attach_slide_images", add_pictures)
    assert app_module._lookup_only(question, ctx)["meta"]["from_cache"] is True

    app_module._RECENT_LESSONS.clear()
    monkeypatch.setattr(app_module, "get_cached_lesson", lambda *a, **k: _lesson(question))
    monkeypatch.setattr(app_module, "_attach_slide_images", lambda *a, **k: None)
    assert app_module._lookup_only(question, ctx) is None

    def explode(q, settings):
        raise RuntimeError(f"provider echoed {q}")

    monkeypatch.setattr(app_module, "generate_lesson", explode)
    monkeypatch.setattr(app_module, "get_cached_lesson", lambda *a, **k: None)
    with app_module.app.test_request_context("/", headers=_headers(), environ_base={"REMOTE_ADDR": REMOTE}):
        data, err = app_module._track_lesson("/api/lesson", question, ctx)
        assert data is None and err == app_module.LESSON_GENERATION_ERROR
        assert calls and all(question not in str(item.get("error")) for item in calls)

        made = _lesson(question)
        made["meta"] = {"math_problems": [question], "estimated_cost_usd": 0.01, "model": "mock",
                        "input_tokens": 1, "output_tokens": 1, "generation_time_ms": 1}
        monkeypatch.setattr(app_module, "generate_lesson", lambda q, settings: made)
        app_module._track_lesson("/api/lesson", question, ctx)

    prompt = f"{question}. A number line showing 0.5, with a text label."
    app_module._sanitize_flux_prompt(prompt, "Decimals", "fractions", math=True)
    app_module._sanitize_flux_prompt(f"A picture of {question} with a text label", "Title", "history", math=False)

    monkeypatch.setattr(app_module, "generate_image", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(question)))
    app_module._render_with_flux({"number": 2, "image_prompt": question, "title": question, "subject": "Nature"}, ctx)
    monkeypatch.setattr(app_module, "render_spec", lambda spec: (_ for _ in ()).throw(RuntimeError(question)))
    app_module._render_with_matplotlib({"number": 3, "spec": {"shapes": []}})

    monkeypatch.setattr(app_module, "_attach_slide_images", real_attach)
    monkeypatch.setattr(app_module.settings, "image_provider", "flux")
    monkeypatch.setattr(app_module.settings, "fal_api_key", "test-not-real")
    monkeypatch.setattr(app_module, "generate_images_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(question)))
    monkeypatch.setattr(app_module, "_render_with_flux", lambda *a, **k: False)
    monkeypatch.setattr(app_module, "_render_with_matplotlib", lambda *a, **k: False)
    app_module._attach_slide_images(_lesson(question), ctx)

    _assert_hidden(question, caplog, err, *[item.get("error") for item in calls])
    assert lesson_log_id(question) in caplog.text


@pytest.mark.parametrize("question", QUESTIONS)
def test_http_errors_and_request_logs_omit_the_question_and_address(question, app_module, monkeypatch, caplog):
    def explode(q, settings):
        raise RuntimeError(f"provider echoed {q}")

    monkeypatch.setattr(app_module, "check_question", lambda q: None)
    monkeypatch.setattr(app_module, "generate_lesson", explode)
    monkeypatch.setattr(app_module, "generate_quiz", lambda lesson, settings: (_ for _ in ()).throw(RuntimeError(question)))
    monkeypatch.setattr(app_module, "synthesize", lambda text, settings: (_ for _ in ()).throw(RuntimeError(f"tts echoed {text}")))
    monkeypatch.setattr(app_module, "today_spend_usd", lambda: 0.0)
    monkeypatch.setattr(app_module, "save_feedback", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "pin_cached_lesson", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "save_cached_lesson", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "record_api_call", lambda **kw: None)
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module.settings, "openai_api_key", "test-not-real")
    caplog.set_level(logging.DEBUG)
    client = app_module.app.test_client()
    environ = {"REMOTE_ADDR": REMOTE}

    missed = client.get("/lesson", query_string={"q": question}, headers=_headers(), environ_base=environ)
    assert missed.status_code in (302, 303)

    failed = client.post("/api/lesson", json={"question": question, "email": "mum@example.com", "parent_consent": True},
                         headers=_headers(), environ_base=environ)
    assert failed.status_code == 500
    assert failed.get_json()["error"] == app_module.LESSON_GENERATION_ERROR

    quiz = client.post("/api/quiz", json={"lesson": {"title": question, "slides": [{"title": "S", "explanation": question}]}},
                       headers=_headers(), environ_base=environ)
    assert quiz.status_code == 500
    assert question not in quiz.get_json()["error"]

    audio = client.post("/api/tts", json={"text": question}, headers=_headers(), environ_base=environ)
    assert audio.status_code == 500
    assert question not in audio.get_json()["error"]

    thumb = client.post("/api/feedback", json={"question": question, "rating": "up", "client_id": "abc12345-visit"},
                        headers=_headers(), environ_base=environ)
    assert thumb.status_code == 200

    monkeypatch.setattr(app_module.settings, "per_ip_hourly_limit", 0)
    monkeypatch.setattr(spam_log, "_submit", lambda fn, *args: fn(*args))
    with app_module.app.test_request_context("/", headers=_headers(), environ_base=environ):
        limited = app_module._guardrail_check("/api/lesson", {})
        app_module._log_refused("/api/lesson", question, "too long")
    assert limited == app_module.RATE_LIMIT_ERROR
    _assert_hidden(question, caplog, failed.get_data(as_text=True), quiz.get_json(), audio.get_json())


@pytest.mark.parametrize("question", QUESTIONS)
def test_picture_provider_errors_omit_the_prompt(question, monkeypatch, caplog):
    def submit(*args, **kwargs):
        raise RuntimeError(f"fal echoed {question}")

    monkeypatch.setattr(image_gen, "fal_client", SimpleNamespace(submit=submit))
    settings = SimpleNamespace(fal_api_key="test-not-real", flux_model="fal-ai/flux/schnell", flux_image_size="landscape_16_9")
    caplog.set_level(logging.DEBUG)
    with pytest.raises(RuntimeError) as single:
        image_gen.generate_image(question, settings)
    with pytest.raises(RuntimeError) as batch:
        image_gen.generate_images_batch([question, "a tree"], settings)
    rendered = "".join(traceback.format_exception(single.value)) + "".join(traceback.format_exception(batch.value))
    _assert_hidden(question, caplog, single.value, batch.value, rendered)


@pytest.mark.parametrize("question", QUESTIONS)
def test_refused_question_log_and_storage_errors_omit_the_text(question, monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(spam_log, "_PATH", tmp_path / "refused.jsonl")
    monkeypatch.setattr(spam_log, "extract_and_lookup", lambda *a: (FORWARDED, "Austin", "Texas", "United States"))
    caplog.set_level(logging.DEBUG)
    spam_log._write(
        {"at": 1, "endpoint": "/api/lesson", "reason": "too long", "question": question,
         "user_agent": "Test", "device": "abc123"},
        FORWARDED, REMOTE,
    )

    class Bad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def cursor(self):
            raise RuntimeError(question)
        def close(self):
            pass

    monkeypatch.setattr(db, "_direct_connect", lambda: Bad())
    before = dict(db._write_stats)
    try:
        assert db.get_lesson_from_calls(question) is None
        db._record_write_failure("record_api_call", RuntimeError(question))
        reason = db.get_write_stats()["last_failure_reason"]
    finally:
        db._write_stats.clear()
        db._write_stats.update(before)

    class Cursor:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def execute(self, *args, **kwargs):
            pass

    class Conn(Bad):
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(db, "_direct_connect", lambda: Conn())
    db._do_delete_cached_lesson(question_hash(question))
    _assert_hidden(question, caplog, reason)
    assert question_hash(question) in caplog.text
    assert FORWARDED not in (tmp_path / "refused.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("question", QUESTIONS)
def test_admin_delete_logs_the_lesson_id(question, app_module, monkeypatch, caplog):
    monkeypatch.setattr(app_module, "db_enabled", True)
    monkeypatch.setattr(app_module.app, "secret_key", "unit-test-secret")

    class Cursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args, **kwargs):
            if kwargs.get("fail"):
                raise RuntimeError(question)

    class Conn:
        def __init__(self, fail=False):
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            cursor = Cursor()
            if self.fail:
                def execute(*args, **kwargs):
                    raise RuntimeError(question)
                cursor.execute = execute
            return cursor

    class Pool:
        def __init__(self):
            self.fail = False

        def connection(self, timeout=None):
            return Conn(self.fail)

    pool = Pool()
    monkeypatch.setattr(db, "_pool", pool)
    caplog.set_level(logging.DEBUG)
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["admin"] = True
        sess["library_csrf"] = "csrf-token"
    ok = client.post("/admin/delete-cached-lesson", json={"question": question, "csrf_token": "csrf-token"},
                     headers=_headers(), environ_base={"REMOTE_ADDR": REMOTE})
    pool.fail = True
    failed = client.post("/admin/delete-cached-lesson", json={"question": question, "csrf_token": "csrf-token"},
                         headers=_headers(), environ_base={"REMOTE_ADDR": REMOTE})
    assert ok.status_code == 200 and failed.status_code == 500
    _assert_hidden(question, caplog, failed.get_json())
    assert question_hash(question) in caplog.text


def test_gunicorn_access_log_has_no_client_address(monkeypatch):
    import gunicorn.config as gconfig
    from gunicorn.app.base import Application

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(gconfig.Chdir, "default", str(ROOT))
    monkeypatch.setattr(sys, "argv", [
        "gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120",
    ])

    class Probe(Application):
        def init(self, parser, opts, args):
            return None

        def load(self):
            return None

    probe = Probe()
    fmt = probe.cfg.access_log_format
    assert fmt == "%(t)s %(m)s %(U)s %(s)s %(b)s %(M)sms"
    lowered = fmt.lower()
    for token in ("%(h)s", "%(q)s", "%(r)s", "x-forwarded-for", "cf-connecting-ip", "x-real-ip", "remote_addr"):
        assert token not in lowered
    assert probe.cfg.workers == 2
    assert probe.cfg.timeout == 120
    assert "0.0.0.0:10000" in probe.cfg.bind
