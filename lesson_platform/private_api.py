"""Dedicated bearer scope: insert a complete private lesson or read a safe receipt."""
import hmac
import importlib.util
import os
from pathlib import Path

from flask import jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from . import library, library_import
from .private_import_protocol import (
    MAX_PACKET_BYTES, METADATA_FIELDS, fingerprint_prepared, fingerprint_stored, preview_path,
    valid_slug, valid_token,
)


def _error(message, status):
    return jsonify(error=message), status


def _authorize():
    token = os.environ.get("ASKLILOWL_IMPORT_TOKEN")
    if not valid_token(token):
        return _error("Private import is not configured", 503)
    authorization = request.headers.get("Authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not valid_token(supplied):
        return _error("Dedicated import bearer required", 401)
    if not hmac.compare_digest(token.encode("ascii"), supplied.encode("ascii")):
        return _error("Dedicated import bearer required", 401)
    return None


def _receipt(slug, fingerprint, slides=4, audio_clips=4, **extra):
    return {"slug": slug, "preview_url": preview_path(slug), "fingerprint": fingerprint,
            "slides": slides, "audio_clips": audio_clips, "active": False,
            "approved_for_publish": False, **extra}


def _lookup(slug):
    # One SELECT gives a consistent metadata/audio snapshot, without a schema change or writes.
    with library._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT p.slug,p.source_question_hash,p.question,p.title,p.summary,p.category,
                   p.sort_order,p.featured_rank,p.cover_slide_index,p.lesson,p.thumbnail,p.thumbnail_mime,
                   p.active,p.approved_for_publish,a.slide_index,a.mp3,a.model,a.voice,a.source_text_hash
                   FROM published_lessons p LEFT JOIN published_lesson_audio a ON a.slug=p.slug
                   WHERE p.slug=%s ORDER BY a.slide_index""", (slug,))
        rows = cur.fetchall()
    if not rows:
        return None
    row = rows[0]
    if row[12] or row[13]:
        raise library_import.ImportConflict("Receipt is only available for an unapproved private lesson")
    clips = [{"slide_index": r[14], "mp3": bytes(r[15]), "model": r[16], "voice": r[17],
              "source_text_hash": r[18]} for r in rows if r[14] is not None]
    slides = row[9].get("slides", [])
    if len(slides) != 4 or [c["slide_index"] for c in clips] != [0, 1, 2, 3]:
        raise library_import.ImportConflict("Private lesson is incomplete")
    metadata = dict(zip(METADATA_FIELDS, row[:9]))
    return _receipt(slug, fingerprint_stored(metadata, row[9], row[10], row[11], clips))


def _kokoro_capabilities():
    try:
        installed = importlib.util.find_spec("kokoro_onnx") is not None
    except (ImportError, ValueError):
        installed = False
    result = {"package_installed": installed, "verified": False}
    for name in ("model", "voices"):
        configured = os.environ.get(f"KOKORO_{name.upper()}_PATH", "")
        try:
            present = bool(configured) and Path(configured).is_file()
        except (OSError, ValueError):
            present = False
        result[name + "_configured"] = bool(configured)
        result[name + "_file_present"] = present
    result["status"] = "configured_unverified" if installed and all(
        result[k + "_file_present"] for k in ("model", "voices")) else "unavailable"
    return result


def register_private_api(app):
    """Register only import/receipt/capability routes; token grants no session or review access."""
    @app.before_request
    def private_import_scope():
        if request.endpoint in ("private_lessons_import", "private_lessons_receipt", "private_lessons_capabilities"):
            return _authorize()

    @app.after_request
    def private_import_no_store(response):
        if request.endpoint in ("private_lessons_import", "private_lessons_receipt", "private_lessons_capabilities"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/admin/private-lessons", endpoint="private_lessons_import")
    def import_lesson():
        if request.mimetype != "application/json" or request.headers.get("Content-Encoding", "identity") != "identity":
            return _error("Use an uncompressed JSON package", 415)
        declared_length = request.environ.get("CONTENT_LENGTH")
        if declared_length in (None, ""):
            return _error("Content-Length is required", 411)
        if not declared_length.isascii() or not declared_length.isdigit():
            return _error("Invalid Content-Length", 400)
        size = int(declared_length)
        if size > MAX_PACKET_BYTES:
            return _error("Package exceeds 12 MiB", 413)
        try:
            raw = request.stream.read(MAX_PACKET_BYTES + 1)
            if len(raw) > MAX_PACKET_BYTES:
                return _error("Package exceeds 12 MiB", 413)
            ready = library_import.prepare_import(raw)
        except RequestEntityTooLarge:
            return _error("Package exceeds 12 MiB", 413)
        except (ValueError, BadRequest):
            return _error("Invalid private lesson package", 400)
        except library_import.ImportUnavailable:
            return _error("Local media validation is unavailable", 503)
        except Exception:
            return _error("Private package validation is unavailable", 503)
        try:
            fingerprint = fingerprint_prepared(ready)
            result = library_import.save_import(ready)
            receipt = _receipt(result["slug"], fingerprint, created=result["created"])
            return jsonify(receipt), 201 if result["created"] else 200
        except library_import.ImportConflict:
            return _error("Existing content or review state conflicts; nothing was changed", 409)
        except Exception:
            return _error("Storage outcome is unconfirmed; reconcile the private receipt before retrying", 503)

    @app.get("/api/admin/private-lessons/capabilities", endpoint="private_lessons_capabilities")
    def capabilities():
        return jsonify(protocol=1, max_packet_bytes=MAX_PACKET_BYTES,
                       server_tts={"provider": "openai", "source": "lesson_platform.tts.synthesize"},
                       kokoro=_kokoro_capabilities())

    @app.get("/api/admin/private-lessons/<slug>", endpoint="private_lessons_receipt")
    def receipt(slug):
        if not valid_slug(slug):
            return _error("Unknown private lesson", 404)
        try:
            result = _lookup(slug)
            return jsonify(result) if result else _error("Unknown private lesson", 404)
        except library_import.ImportConflict:
            return _error("Lesson is not a complete unapproved private import", 409)
        except Exception:
            return _error("Private receipt lookup is unavailable", 503)
