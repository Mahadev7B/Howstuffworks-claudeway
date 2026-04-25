"""HowStuffWorks - Claudeway

A Flask app that turns any 'How does X work?' question into a kid-friendly
4-slide illustrated lesson with voiceover.

Powered by Claude (single API call per lesson).
Images: Claude-generated SVG (free).
Voiceover: browser Web Speech API (free).
"""
import base64
import logging
import time

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from lesson_platform import (
    extract_and_lookup,
    generate_image,
    generate_lesson,
    get_cached_lesson,
    init_db,
    ip_calls_last_hour,
    load_settings,
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
db_enabled = init_db()


EXAMPLE_QUESTIONS = [
    "How do rockets fly?",
    "How do aeroplanes fly?",
    "How does rain happen?",
    "How does a volcano erupt?",
    "Why is the sky blue?",
    "How do plants eat sunlight?",
]

LESSON_ENDPOINTS = ("/lesson", "/api/lesson")


def _render_with_flux(slide: dict, ctx: dict) -> bool:
    """Generate image via Flux Schnell. Returns True on success. Records api_call."""
    prompt = (slide.get("image_prompt") or "").strip()
    if not prompt:
        return False
    started = time.time()
    try:
        img_bytes, mime, cost = generate_image(prompt, settings)
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
    """
    if ctx is None:
        ctx = {"ip_address": None, "city": None, "region": None,
               "country": None, "user_agent": None}
    use_flux = settings.image_provider == "flux" and bool(settings.fal_api_key)
    rendered_flux = rendered_mpl = 0
    for slide in lesson.get("slides", []):
        if slide.get("image_data_url"):
            continue  # already rendered (cache hit)
        ok = False
        if use_flux:
            ok = _render_with_flux(slide, ctx)
            if ok:
                rendered_flux += 1
        if not ok:
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


def _guardrail_check(ctx: dict) -> str | None:
    """Return an error message if a guardrail blocks the request, else None."""
    spent = today_spend_usd()
    if spent >= settings.daily_budget_usd:
        logger.warning("Daily budget exceeded: $%.4f >= $%.2f", spent, settings.daily_budget_usd)
        return (
            "Daily lesson budget reached. Cached lessons still work — try one of the examples."
        )
    hits = ip_calls_last_hour(ctx.get("ip_address"), LESSON_ENDPOINTS)
    if hits >= settings.per_ip_hourly_limit:
        logger.warning("IP rate limit hit: %s (%d calls/hour)", ctx.get("ip_address"), hits)
        return (
            f"You've asked {hits} questions in the last hour. "
            "Please wait a bit before asking another one."
        )
    return None


def _track_lesson(endpoint: str, question: str, ctx: dict) -> tuple[dict | None, str | None]:
    """Serve from cache if possible; otherwise generate, record, and cache."""
    cached = get_cached_lesson(question) if db_enabled else None
    if cached is not None:
        cached.setdefault("meta", {})["from_cache"] = True
        record_api_call(
            endpoint=endpoint,
            question=question,
            model=cached.get("meta", {}).get("model", "cache"),
            duration_ms=0,
            cost_usd=0.0,
            success=True,
            **ctx,
        )
        logger.info("Cache hit for question: %s", question)
        # Cache may or may not have rendered images embedded; ensure they exist.
        _attach_slide_images(cached, ctx)
        return cached, None

    blocked = _guardrail_check(ctx)
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
    # Render images first (Flux or matplotlib), THEN cache so cached lessons
    # don't re-pay for image generation on every hit.
    render_started = time.time()
    _attach_slide_images(data, ctx)
    logger.info("Image rendering took %dms", int((time.time() - render_started) * 1000))
    if db_enabled:
        save_cached_lesson(question, data)
    return data, None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", examples=EXAMPLE_QUESTIONS)


@app.route("/lesson", methods=["POST"])
def lesson():
    question = request.form.get("question", "").strip()
    if not question:
        return redirect(url_for("home"))

    ctx = _client_context()
    data, err = _track_lesson("/lesson", question, ctx)
    if err is not None:
        return render_template("error.html", question=question, error=err), 500
    return render_template("lesson.html", question=question, lesson=data)


@app.route("/api/lesson", methods=["POST"])
def api_lesson():
    """JSON endpoint for async / client-side fetches."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    ctx = _client_context()
    data, err = _track_lesson("/api/lesson", question, ctx)
    if err is not None:
        return jsonify({"ok": False, "error": err}), 500
    return jsonify({"ok": True, "lesson": data})


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
    try:
        save_feedback(question, rating, comment, **_client_context())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Feedback save failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
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
