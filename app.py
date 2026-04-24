"""HowStuffWorks - Claudeway

A Flask app that turns any 'How does X work?' question into a kid-friendly
4-slide illustrated lesson with voiceover.

Powered by Claude (single API call per lesson).
Images: Claude-generated SVG (free).
Voiceover: browser Web Speech API (free).
"""
import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

from lesson_platform import generate_lesson, init_db, load_settings, save_feedback

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = load_settings()
app.secret_key = settings.flask_secret_key
feedback_enabled = init_db()


EXAMPLE_QUESTIONS = [
    "How do rockets fly?",
    "How do aeroplanes fly?",
    "How does rain happen?",
    "How does a volcano erupt?",
    "Why is the sky blue?",
    "How do plants eat sunlight?",
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", examples=EXAMPLE_QUESTIONS)


@app.route("/lesson", methods=["POST"])
def lesson():
    question = request.form.get("question", "").strip()
    if not question:
        return redirect(url_for("home"))

    try:
        data = generate_lesson(question, settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lesson generation failed")
        return render_template("error.html", question=question, error=str(exc)), 500

    return render_template("lesson.html", question=question, lesson=data)


@app.route("/api/lesson", methods=["POST"])
def api_lesson():
    """JSON endpoint for async / client-side fetches."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    try:
        data = generate_lesson(question, settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lesson generation failed (API)")
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, "lesson": data})


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    if not feedback_enabled:
        return jsonify({"ok": False, "error": "feedback storage not configured"}), 503
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    rating = (payload.get("rating") or "").strip()
    comment = (payload.get("comment") or "").strip() or None
    if not question or rating not in ("up", "down"):
        return jsonify({"ok": False, "error": "question and rating ('up'|'down') are required"}), 400
    try:
        save_feedback(question, rating, comment)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Feedback save failed")
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True})


@app.route("/healthz")
def healthz():
    return {"ok": True, "model": settings.anthropic_model, "feedback": feedback_enabled}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=settings.port, debug=True)
