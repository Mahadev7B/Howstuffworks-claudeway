"""HowStuffWorks - Claudeway

A Flask app that turns any 'How does X work?' question into a kid-friendly
4-slide illustrated lesson with voiceover.

Powered by Claude (single API call per lesson).
Images: Claude-generated SVG (free).
Voiceover: browser Web Speech API (free).
"""
import base64
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import secrets

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from lesson_platform import (
    admin_load_all,
    admin_lessons_filtered,
    check_question,
    generate_quiz,
    delete_cached_lesson,
    extract_and_lookup,
    generate_image,
    generate_lesson,
    get_cached_lesson,
    get_lesson_from_calls,
    init_db,
    ip_calls_last_hour,
    load_settings,
    pin_cached_lesson,
    record_api_call,
    render_spec,
    save_cached_lesson,
    save_feedback,
    synthesize,
    today_spend_usd,
)

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = load_settings()
app.secret_key = settings.flask_secret_key
from datetime import timedelta as _td
app.permanent_session_lifetime = _td(days=7)
db_enabled = init_db()


# Short-lived in-memory cache so a freshly-generated lesson (POST /api/lesson)
# is served instantly when the user taps Open Lesson (GET /lesson?q=...).
# Keyed by normalized question; entry expires after 10 minutes.
_RECENT_LESSONS: dict[str, tuple[float, dict]] = {}
_RECENT_TTL_S = 3600  # 1 hour — long enough that refresh after reading a lesson stays a hit


def _recent_key(question: str) -> str:
    return question.strip().lower()


def _recent_get(question: str) -> dict | None:
    key = _recent_key(question)
    entry = _RECENT_LESSONS.get(key)
    if entry is None:
        return None
    ts, data = entry
    if time.time() - ts > _RECENT_TTL_S:
        _RECENT_LESSONS.pop(key, None)
        return None
    return data


def _recent_put(question: str, data: dict) -> None:
    _RECENT_LESSONS[_recent_key(question)] = (time.time(), data)
    # Cheap GC: drop expired entries when dict grows
    if len(_RECENT_LESSONS) > 64:
        cutoff = time.time() - _RECENT_TTL_S
        for k in [k for k, (ts, _) in _RECENT_LESSONS.items() if ts < cutoff]:
            _RECENT_LESSONS.pop(k, None)


EXAMPLE_QUESTIONS = [
    "How do rockets fly?",
    "How do aeroplanes fly?",
    "How does rain happen?",
    "How does a volcano erupt?",
    "Why is the sky blue?",
    "How do plants eat sunlight?",
]

# Only POST /api/lesson actually triggers Claude/Flux generation. GET /lesson
# may serve from cache or the in-memory recent-lesson cache, so it must NOT
# be counted toward the hourly lesson limit.
LESSON_ENDPOINTS = ("/api/lesson",)
LESSON_RATE_LIMIT_ENDPOINTS = LESSON_ENDPOINTS
FEEDBACK_HOURLY_LIMIT = 10  # per IP

# Sentinel returned by _guardrail_check when the rate limit is the reason
RATE_LIMIT_ERROR = "rate_limit"

# Phrases that cause Flux to attempt rendering text, producing garbled results.
# Arrows and diagrams are fine visually — only text/word rendering is the issue.
_FLUX_BANNED_PROMPT_WORDS = {
    "text", "label", "labels", "labeled", "labelled",
    "caption", "callout", "callouts", "annotation", "annotated",
    "text overlay", "written", "writing", "word", "words",
    "letter", "letters", "number", "numbers", "says", "reads",
}

_FLUX_STYLE_SUFFIX = (
    "children's educational storybook illustration, "
    "flat bright colors, soft rounded shapes, no text, no labels"
)


def _sanitize_flux_prompt(prompt: str, slide_title: str, subject: str) -> str:
    """Return a safe prompt, replacing it if it contains diagram/arrow language."""
    words = set(prompt.lower().split())
    if words & _FLUX_BANNED_PROMPT_WORDS:
        logger.warning("Flux prompt contained banned words — rewriting: %s", prompt[:120])
        return (
            f"A cheerful illustrated scene showing {subject}, {slide_title.lower()}, "
            f"bright and colorful, simple composition, "
            f"{_FLUX_STYLE_SUFFIX}"
        )
    return prompt


def _render_with_flux(slide: dict, ctx: dict) -> bool:
    """Generate image via Flux Schnell. Returns True on success. Records api_call."""
    prompt = (slide.get("image_prompt") or "").strip()
    if not prompt:
        return False
    prompt = _sanitize_flux_prompt(
        prompt,
        slide.get("title", ""),
        slide.get("subject", ""),
    )
    negative_prompt = (slide.get("image_negative_prompt") or "").strip() or None
    started = time.time()
    try:
        img_bytes, mime, cost = generate_image(prompt, settings, negative_prompt)
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.time() - started) * 1000)
        logger.warning("Flux gen failed for slide %s: %s", slide.get("number"), exc)
        record_api_call(
            endpoint="/internal/flux",
            question=prompt[:500],
            model=settings.flux_model,
            duration_ms=duration_ms,
            success=False,
            error=str(exc),
            **ctx,
        )
        return False
    duration_ms = int((time.time() - started) * 1000)
    b64 = base64.b64encode(img_bytes).decode("ascii")
    slide["image_data_url"] = f"data:{mime};base64,{b64}"
    slide["image_bytes"] = len(img_bytes)
    slide["image_source"] = "flux"
    record_api_call(
        endpoint="/internal/flux",
        question=prompt[:500],
        model=settings.flux_model,
        cost_usd=cost,
        duration_ms=duration_ms,
        success=True,
        **ctx,
    )
    return True


def _render_with_matplotlib(slide: dict) -> bool:
    spec = slide.get("spec")
    if not isinstance(spec, dict):
        return False
    try:
        img_bytes, mime = render_spec(spec)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Matplotlib render failed: %s", exc)
        return False
    b64 = base64.b64encode(img_bytes).decode("ascii")
    slide["image_data_url"] = f"data:{mime};base64,{b64}"
    slide["image_bytes"] = len(img_bytes)
    slide["image_source"] = "matplotlib"
    return True


def _attach_slide_images(lesson: dict, ctx: dict | None = None) -> None:
    """Attach an `image_data_url` to each slide.

    Provider chosen by settings.image_provider. Falls back to matplotlib if
    Flux is unavailable or fails for a given slide. Mutates lesson in place.
    Flux slides are rendered in parallel for speed.
    """
    if ctx is None:
        ctx = {"ip_address": None, "city": None, "region": None,
               "country": None, "user_agent": None}
    use_flux = settings.image_provider == "flux" and bool(settings.fal_api_key)
    subject = lesson.get("subject", "")
    for s in lesson.get("slides", []):
        s.setdefault("subject", subject)
    slides = [s for s in lesson.get("slides", []) if not s.get("image_data_url")]
    rendered_flux = rendered_mpl = 0

    if use_flux and slides:
        def _flux_one(slide):
            return slide, _render_with_flux(slide, ctx)

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_flux_one, s): s for s in slides}
            for fut in as_completed(futures):
                slide, ok = fut.result()
                if ok:
                    rendered_flux += 1
                else:
                    if _render_with_matplotlib(slide):
                        rendered_mpl += 1
                    else:
                        slide["image_data_url"] = None
    else:
        for slide in slides:
            if _render_with_matplotlib(slide):
                rendered_mpl += 1
            else:
                slide["image_data_url"] = None

    logger.info(
        "Rendered slides — flux: %d, matplotlib: %d, total: %d",
        rendered_flux, rendered_mpl, len(lesson.get("slides", [])),
    )


def _client_context() -> dict:
    ip, city, region, country = extract_and_lookup(
        request.headers.get("X-Forwarded-For"), request.remote_addr
    )
    return {
        "ip_address": ip,
        "city": city,
        "region": region,
        "country": country,
        "user_agent": request.headers.get("User-Agent"),
    }


def _guardrail_check(endpoint: str, ctx: dict) -> str | None:
    """Return an error code/message if a guardrail blocks the request, else None.

    Rate-limit failures return the sentinel string ``RATE_LIMIT_ERROR`` so
    callers can surface HTTP 429.
    """
    spent = today_spend_usd()
    if spent >= settings.daily_budget_usd:
        logger.warning("Daily budget exceeded: $%.4f >= $%.2f", spent, settings.daily_budget_usd)
        return (
            "Daily lesson budget reached. Cached lessons still work — try one of the examples."
        )

    # Only check/log the lesson rate limiter on endpoints that actually
    # trigger generation (POST /api/lesson). GET /lesson, /api/tts, static
    # files etc. are excluded.
    if endpoint in LESSON_RATE_LIMIT_ENDPOINTS:
        hits = ip_calls_last_hour(ctx.get("ip_address"), LESSON_RATE_LIMIT_ENDPOINTS)
        limit = settings.per_ip_hourly_limit
        logger.info(
            "Lesson rate-limit check route=%s ip=%s count=%d limit=%d",
            endpoint, ctx.get("ip_address"), hits, limit,
        )
        if hits >= limit:
            logger.warning("IP rate limit hit: %s (%d/%d)", ctx.get("ip_address"), hits, limit)
            return RATE_LIMIT_ERROR
    return None


def _track_lesson(endpoint: str, question: str, ctx: dict) -> tuple[dict | None, str | None]:
    """Serve from cache if possible; otherwise generate, record, and cache."""
    recent = _recent_get(question)
    if recent is not None:
        logger.info("Recent in-memory cache hit: %s", question[:80])
        return recent, None

    cached = get_cached_lesson(question, settings.lesson_cache_ttl_days) if db_enabled else None
    if cached is not None:
        cached.setdefault("meta", {})["from_cache"] = True
        logger.info("Cache hit for question: %s", question)
        # Cache may or may not have rendered images embedded; ensure they exist.
        _attach_slide_images(cached, ctx)
        # If every slide is still missing an image the cached spec is broken — bust it.
        slides = cached.get("slides", [])
        if slides and not any(s.get("image_data_url") for s in slides):
            logger.warning("Cached lesson has no renderable images — busting cache: %s", question)
            delete_cached_lesson(question)
            cached = None
        else:
            record_api_call(
                endpoint=endpoint,
                question=question,
                model=cached.get("meta", {}).get("model", "cache"),
                duration_ms=0,
                cost_usd=0.0,
                success=True,
                **ctx,
            )
            return cached, None

    blocked = _guardrail_check(endpoint, ctx)
    if blocked is not None:
        return None, blocked

    started = time.time()
    try:
        data = generate_lesson(question, settings)
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.time() - started) * 1000)
        logger.exception("Lesson generation failed (%s)", endpoint)
        record_api_call(
            endpoint=endpoint,
            question=question,
            model=settings.anthropic_model,
            duration_ms=duration_ms,
            success=False,
            error=str(exc),
            **ctx,
        )
        return None, str(exc)

    meta = data.get("meta", {})
    cost = meta.get("estimated_cost_usd", 0.0)
    if cost > settings.max_cost_per_lesson_usd:
        logger.warning(
            "Lesson cost $%.4f exceeded cap $%.2f — still served but flagged",
            cost,
            settings.max_cost_per_lesson_usd,
        )

    record_api_call(
        endpoint=endpoint,
        question=question,
        model=meta.get("model", settings.anthropic_model),
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        cost_usd=cost,
        duration_ms=meta.get("generation_time_ms", 0),
        success=True,
        lesson=data,
        **ctx,
    )
    render_started = time.time()
    _attach_slide_images(data, ctx)
    render_ms = int((time.time() - render_started) * 1000)
    total_ms = int((time.time() - started) * 1000)
    logger.info("Flux render %dms | total request %dms", render_ms, total_ms)
    _recent_put(question, data)
    return data, None


@app.route("/", methods=["GET"])
def home():
    prefill = (request.args.get("q") or "").strip()[:200]
    return render_template("index.html", examples=EXAMPLE_QUESTIONS, prefill=prefill)


def _lookup_only(question: str, ctx: dict) -> dict | None:
    """Return a cached lesson without ever calling Claude/Flux.

    Checks the in-memory recent cache and the persisted DB cache. Returns
    None if neither has the lesson — caller should redirect rather than
    regenerate.
    """
    recent = _recent_get(question)
    if recent is not None:
        return recent
    cached = get_cached_lesson(question, settings.lesson_cache_ttl_days) if db_enabled else None
    if cached is None:
        return None
    cached.setdefault("meta", {})["from_cache"] = True
    _attach_slide_images(cached, ctx)
    slides = cached.get("slides", [])
    if slides and not any(s.get("image_data_url") for s in slides):
        logger.warning("Cached lesson has no renderable images — busting cache: %s", question)
        delete_cached_lesson(question)
        return None
    # Refresh the in-memory cache so a follow-up Back/refresh stays instant
    _recent_put(question, cached)
    return cached


@app.route("/lesson", methods=["GET", "POST"])
def lesson():
    if request.method == "GET":
        question = request.args.get("q", "").strip()
    else:
        question = request.form.get("question", "").strip()
    if not question:
        return redirect(url_for("home"))
    if len(question) > 200:
        return render_template("error.html", question=question[:200],
                               error="Your question is a bit too long — try a shorter version!"), 400

    blocked = check_question(question)
    if blocked:
        return render_template("error.html", question=question, error=blocked), 400

    ctx = _client_context()

    # GET = navigation (Back, refresh, share link). NEVER regenerate here.
    # If the lesson isn't cached, send the user back to the home screen with
    # the question pre-filled so they have to explicitly click Ask.
    if request.method == "GET":
        cached = _lookup_only(question, ctx)
        if cached is None:
            logger.info("GET /lesson cache miss — redirecting to home: %s", question[:80])
            return redirect(url_for("home", q=question))
        resp = app.make_response(render_template("lesson.html", question=question, lesson=cached))
        # Browser-cache the rendered page for 10 minutes so a Refresh hits the
        # browser cache and never reaches our server.
        resp.headers["Cache-Control"] = "private, max-age=600"
        return resp

    # POST = explicit form submission (no-JS fallback). Generate if needed.
    data, err = _track_lesson("/lesson", question, ctx)
    if err is not None:
        if err == RATE_LIMIT_ERROR:
            friendly = "Too many questions. Please try again later."
            return render_template("error.html", question=question, error=friendly), 429
        return render_template("error.html", question=question, error=err), 500
    resp = app.make_response(render_template("lesson.html", question=question, lesson=data))
    resp.headers["Cache-Control"] = "private, max-age=600"
    return resp


@app.route("/api/lesson", methods=["POST"])
def api_lesson():
    """JSON endpoint for async / client-side fetches."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400
    if len(question) > 200:
        return jsonify({"ok": False, "error": "Question too long — keep it under 200 characters."}), 400

    blocked = check_question(question)
    if blocked:
        return jsonify({"ok": False, "error": blocked}), 400

    ctx = _client_context()
    data, err = _track_lesson("/api/lesson", question, ctx)
    if err is not None:
        if err == RATE_LIMIT_ERROR:
            return jsonify({
                "ok": False,
                "error": "rate_limit",
                "message": "Too many questions. Please try again later.",
            }), 429
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({"ok": True, "lesson": data})


@app.route("/api/quiz", methods=["POST"])
def api_quiz():
    """Generate a 3-question quiz from a lesson. Expects {lesson: {...}}."""
    payload = request.get_json(silent=True) or {}
    lesson = payload.get("lesson")
    if not isinstance(lesson, dict) or not lesson.get("slides"):
        return jsonify({"ok": False, "error": "lesson is required"}), 400
    try:
        quiz = generate_quiz(lesson, settings)
        return jsonify({"ok": True, "quiz": quiz})
    except Exception as exc:
        logger.exception("Quiz generation failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    if not db_enabled:
        return jsonify({"ok": False, "error": "feedback storage not configured"}), 503
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    rating = (payload.get("rating") or "").strip()
    comment = (payload.get("comment") or "").strip() or None
    if not question or rating not in ("up", "down"):
        return jsonify({"ok": False, "error": "question and rating ('up'|'down') are required"}), 400

    ctx = _client_context()
    if ip_calls_last_hour(ctx.get("ip_address"), ("/api/feedback",)) >= FEEDBACK_HOURLY_LIMIT:
        return jsonify({"ok": False, "error": "Too many feedback submissions. Try again later."}), 429

    try:
        save_feedback(question, rating, comment, **ctx)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Feedback save failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    if rating == "up" and db_enabled:
        lesson_data = get_lesson_from_calls(question)
        if lesson_data:
            save_cached_lesson(question, lesson_data)
            pin_cached_lesson(question)
            logger.info("Cached and pinned thumbs-up lesson: %s", question[:80])

    record_api_call(
        endpoint="/api/feedback",
        question=question[:500],
        model="none",
        duration_ms=0,
        success=True,
        **ctx,
    )
    return jsonify({"ok": True})


@app.route("/api/tts", methods=["POST"])
def api_tts():
    if not settings.openai_api_key:
        return jsonify({"ok": False, "error": "TTS not configured"}), 503
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text is required"}), 400
    if len(text) > 4096:
        text = text[:4096]

    ctx = _client_context()
    # Budget check applies to TTS too — same pool
    spent = today_spend_usd()
    if spent >= settings.daily_budget_usd:
        return jsonify({"ok": False, "error": "Daily budget reached"}), 503

    started = time.time()
    try:
        audio_bytes, cost = synthesize(text, settings)
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.time() - started) * 1000)
        logger.exception("TTS synthesis failed")
        record_api_call(
            endpoint="/api/tts",
            question=text[:500],
            model=settings.openai_tts_model,
            duration_ms=duration_ms,
            success=False,
            error=str(exc),
            **ctx,
        )
        return jsonify({"ok": False, "error": str(exc)}), 500

    duration_ms = int((time.time() - started) * 1000)
    record_api_call(
        endpoint="/api/tts",
        question=text[:500],
        model=settings.openai_tts_model,
        input_chars=len(text),
        cost_usd=cost,
        duration_ms=duration_ms,
        success=True,
        **ctx,
    )
    return Response(audio_bytes, mimetype="audio/mpeg")


@app.route("/admin/clear-cache", methods=["POST"])
def admin_clear_cache():
    secret = os.environ.get("ADMIN_SECRET", "")
    if not secret or request.headers.get("X-Admin-Secret") != secret:
        return jsonify({"ok": False, "error": "forbidden"}), 403
    if not db_enabled:
        return jsonify({"ok": False, "error": "DB not enabled"}), 503
    from lesson_platform.db import _pool, _POOL_TIMEOUT_S
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM cached_lessons")
            deleted = cur.rowcount
        logger.info("Admin cleared cache: %d rows deleted", deleted)
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as exc:
        logger.exception("Admin clear-cache failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/admin/delete-cached-lesson", methods=["POST"])
def admin_delete_cached_lesson():
    """Delete a single cached lesson by question text. Requires admin session."""
    if not _admin_authed():
        return jsonify({"ok": False, "error": "forbidden"}), 403
    if not db_enabled:
        return jsonify({"ok": False, "error": "DB not enabled"}), 503
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400
    from lesson_platform.db import _pool, _POOL_TIMEOUT_S, question_hash
    try:
        qhash = question_hash(question)
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM cached_lessons WHERE question_hash = %s", (qhash,))
            deleted = cur.rowcount
        logger.info("Admin deleted cached lesson: %s (found=%s)", question[:80], deleted > 0)
        return jsonify({"ok": True, "deleted": deleted > 0})
    except Exception as exc:
        logger.exception("Admin delete-cached-lesson failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/admin/cached-lessons", methods=["GET"])
def admin_cached_lessons():
    """List all cached lessons. Requires admin session."""
    if not _admin_authed():
        return jsonify({"ok": False, "error": "forbidden"}), 403
    if not db_enabled:
        return jsonify({"ok": False, "error": "DB not enabled"}), 503
    from lesson_platform.db import _pool, _POOL_TIMEOUT_S, ADMIN_TZ
    try:
        with _pool.connection(timeout=_POOL_TIMEOUT_S) as conn, conn.cursor() as cur:
            cur.execute(f"""
                SELECT question,
                       hit_count,
                       pinned,
                       to_char(created_at AT TIME ZONE '{ADMIN_TZ}', 'YYYY-MM-DD HH24:MI') AS created,
                       to_char(last_hit_at  AT TIME ZONE '{ADMIN_TZ}', 'YYYY-MM-DD HH24:MI') AS last_hit
                FROM cached_lessons
                ORDER BY hit_count DESC, created_at DESC
            """)
            rows = [{"question": r[0], "hits": r[1], "pinned": r[2],
                     "created": r[3], "last_hit": r[4]} for r in cur.fetchall()]
        return jsonify({"ok": True, "lessons": rows})
    except Exception as exc:
        logger.exception("Admin cached-lessons list failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Admin dashboard — read-only views over api_calls. Never touches lesson path.
# ---------------------------------------------------------------------------
def _admin_authed() -> bool:
    return bool(session.get("admin"))


def _admin_password() -> str:
    return os.environ.get("ADMIN_PASSWORD", "")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        provided = request.form.get("password", "")
        expected = _admin_password()
        if not expected:
            return render_template("admin_login.html",
                                   error="Admin not configured (ADMIN_PASSWORD missing)."), 503
        if secrets.compare_digest(provided.encode(), expected.encode()):
            session["admin"] = True
            session.permanent = True
            return redirect(url_for("admin_dashboard"))
        return render_template("admin_login.html", error="Wrong password."), 401
    if _admin_authed():
        return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html", error=None)


@app.route("/admin/logout", methods=["POST", "GET"])
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    if not _admin_authed():
        return redirect(url_for("admin_login"))
    if not db_enabled:
        return render_template("admin.html", db_enabled=False, data={})
    data = admin_load_all(
        days_lessons=14, days_cost=14, days_perf=14,
        limit_questions=15, limit_recent=50,
        limit_slowest=10, limit_errors=30,
    )
    from lesson_platform.db import ADMIN_TZ
    return render_template("admin.html", db_enabled=True, data=data,
                           daily_budget=settings.daily_budget_usd,
                           admin_tz=ADMIN_TZ)


@app.route("/admin/lessons-by-location", methods=["GET"])
def admin_lessons_by_location():
    """JSON: recent lessons filtered by country/region. Used by the location filter UI."""
    if not _admin_authed():
        return jsonify({"ok": False, "error": "forbidden"}), 403
    if not db_enabled:
        return jsonify({"ok": False, "lessons": []}), 503
    country = request.args.get("country") or None
    region = request.args.get("region") or None
    rows = admin_lessons_filtered(country=country, region=region, limit=100)
    return jsonify({"ok": True, "lessons": rows})


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.route("/healthz")
def healthz():
    return {
        "ok": True,
        "model": settings.anthropic_model,
        "db": db_enabled,
        "tts": bool(settings.openai_api_key),
        "image_provider": settings.image_provider,
        "flux": settings.image_provider == "flux" and bool(settings.fal_api_key),
        "today_spend_usd": round(today_spend_usd(), 4),
        "daily_budget_usd": settings.daily_budget_usd,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.port, debug=True)
