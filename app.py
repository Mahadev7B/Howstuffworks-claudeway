"""HowStuffWorks - Claudeway

A Flask app that turns any 'How does X work?' question into a kid-friendly
4-slide illustrated lesson with voiceover.

Powered by Claude (single API call per lesson).
Images: Claude-generated SVG (free).
Voiceover: browser Web Speech API (free).
"""
import logging
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

from lesson_platform import (
    extract_and_lookup,
    generate_lesson,
    init_db,
    load_settings,
    record_api_call,
    save_feedback,
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


def _track_lesson(endpoint: str, question: str, ctx: dict) -> tuple[dict | None, str | None]:
    """Generate a lesson, record the API call. Returns (lesson_or_None, error_or_None)."""
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
    record_api_call(
        endpoint=endpoint,
        question=question,
        model=meta.get("model", settings.anthropic_model),
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        cost_usd=meta.get("estimated_cost_usd", 0.0),
        duration_ms=meta.get("generation_time_ms", 0),
        success=True,
        lesson=data,
        **ctx,
    )
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


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.route("/healthz")
def healthz():
    return {"ok": True, "model": settings.anthropic_model, "db": db_enabled}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.port, debug=True)
