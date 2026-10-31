"""One held cube may remove narration emphasis without changing editorial content."""
import copy
import hashlib

from flask import jsonify, request
from psycopg.types.json import Jsonb
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from . import db, library, library_import
from .private_import_protocol import (
    CUBE_SLUG, MAX_PACKET_BYTES, METADATA_FIELDS, fingerprint_prepared,
    fingerprint_stored, valid_fingerprint,
)


ENDPOINT = "private_cube_formatting_revision"


def save_formatting_revision(ready, expected_fingerprint):
    """Compare a locked snapshot, then update its JSON and existing clips together."""
    if (ready.entry["slug"], ready.entry["sort_order"], len(ready.clips)) != (CUBE_SLUG, 28, 16):
        raise ValueError("Formatting revision is restricted to cube 28")
    if not valid_fingerprint(expected_fingerprint):
        raise ValueError("Expected fingerprint is required")
    target_fingerprint = fingerprint_prepared(ready)
    with library._connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("asklilowl-private-import",))
        cur.execute("""SELECT slug,source_question_hash,question,title,summary,category,sort_order,
                       featured_rank,cover_slide_index,lesson,thumbnail,thumbnail_mime,
                       active,approved_for_publish FROM published_lessons WHERE slug=%s FOR UPDATE""",
                    (CUBE_SLUG,))
        row = cur.fetchone()
        if row is None or row[12] or row[13] or row[6] != 28 or row[8] != 0:
            raise library_import.ImportConflict("Cube must already be a complete private draft")
        lesson = row[9]
        if not isinstance(lesson, dict) or not isinstance(lesson.get("slides"), list) or len(lesson["slides"]) != 16:
            raise library_import.ImportConflict("Cube must already contain all sixteen slides")
        cur.execute("""SELECT slide_index,mp3,model,voice,source_text_hash FROM published_lesson_audio
                       WHERE slug=%s ORDER BY slide_index FOR UPDATE""", (CUBE_SLUG,))
        audio_rows = cur.fetchall()
        if [clip[0] for clip in audio_rows] != list(range(16)):
            raise library_import.ImportConflict("Cube must already contain all sixteen clips")
        for index, clip in enumerate(audio_rows):
            if (not clip[1] or (clip[2], clip[3]) != (library_import.MODEL, library_import.VOICE)
                    or clip[4] != hashlib.sha256(library.publication_audio_text(lesson["slides"][index]).encode()).hexdigest()):
                raise library_import.ImportConflict("Existing cube narration is incomplete or stale")
        clips = [dict(zip(("slide_index", "mp3", "model", "voice", "source_text_hash"), clip)) for clip in audio_rows]
        metadata = dict(zip(METADATA_FIELDS, row[:9]))
        current_fingerprint = fingerprint_stored(metadata, lesson, row[10], row[11], clips)

        expected_lesson = copy.deepcopy(lesson)
        removed_asterisks = False
        for slide in expected_lesson["slides"]:
            for field in ("narration", "explanation"):
                removed_asterisks = removed_asterisks or "*" in slide[field]
                slide[field] = slide[field].replace("*", "")
        expected_lesson = library.with_publication_scripts(expected_lesson)
        expected_metadata = {**ready.entry, "source_question_hash": db.question_hash(ready.entry["question"]),
                             "cover_slide_index": 0}
        if (metadata != {field: expected_metadata[field] for field in METADATA_FIELDS}
                or ready.prepared.lesson != expected_lesson
                or ready.prepared.thumbnail != bytes(row[10]) or ready.prepared.thumbnail_mime != row[11]):
            raise library_import.ImportConflict("Only removal of narration and explanation asterisks is allowed")

        if current_fingerprint == target_fingerprint:
            # A retry after a committed response was lost is a read-only success.
            return {"slug": CUBE_SLUG, "fingerprint": current_fingerprint, "revised": False}
        if not removed_asterisks:
            raise library_import.ImportConflict("An already plain cube cannot be rewritten by this formatting action")
        if current_fingerprint != expected_fingerprint:
            raise library_import.ImportConflict("Private cube changed since its receipt was read")

        cur.execute("UPDATE published_lessons SET lesson=%s WHERE slug=%s", (Jsonb(ready.prepared.lesson), CUBE_SLUG))
        for index, clip in enumerate(ready.clips):
            cur.execute("""UPDATE published_lesson_audio SET mp3=%s,model=%s,voice=%s,source_text_hash=%s
                           WHERE slug=%s AND slide_index=%s RETURNING slide_index""",
                        (clip["mp3"], clip["model"], clip["voice"], clip["source_text_hash"], CUBE_SLUG, index))
            if cur.fetchone() != (index,):
                raise RuntimeError("Private cube audio replacement was incomplete")
        if library_import._existing_matches(cur, ready) is not True:
            raise RuntimeError("Private cube revision verification failed")
    return {"slug": CUBE_SLUG, "fingerprint": target_fingerprint, "previous_fingerprint": current_fingerprint,
            "revised": True}


def register_cube_revision(app, *, error, receipt):
    """Register a literal cube-only POST under the shared bearer and no-store hooks."""
    @app.post("/api/admin/private-lessons/solve-a-2x2-rubiks-cube/formatting-revision", endpoint=ENDPOINT)
    def revise():
        expected = request.headers.get("If-Match")
        if not expected:
            return error("If-Match private receipt fingerprint is required", 428)
        if not valid_fingerprint(expected):
            return error("If-Match must be one exact lowercase SHA-256 fingerprint", 400)
        if request.mimetype != "application/json" or request.headers.get("Content-Encoding", "identity") != "identity":
            return error("Use an uncompressed JSON package", 415)
        declared = request.environ.get("CONTENT_LENGTH")
        if declared in (None, ""):
            return error("Content-Length is required", 411)
        if not declared.isascii() or not declared.isdigit():
            return error("Invalid Content-Length", 400)
        if int(declared) > MAX_PACKET_BYTES:
            return error("Package exceeds 12 MiB", 413)
        try:
            raw = request.stream.read(MAX_PACKET_BYTES + 1)
            if len(raw) > MAX_PACKET_BYTES:
                return error("Package exceeds 12 MiB", 413)
            ready = library_import.prepare_import(raw)
            if (ready.entry["slug"], ready.entry["sort_order"], len(ready.clips)) != (CUBE_SLUG, 28, 16):
                raise ValueError("Only cube 28 can be revised")
        except RequestEntityTooLarge:
            return error("Package exceeds 12 MiB", 413)
        except (ValueError, BadRequest):
            return error("Invalid private cube package", 400)
        except library_import.ImportUnavailable:
            return error("Local media validation is unavailable", 503)
        except Exception:
            return error("Private package validation is unavailable", 503)
        try:
            result = save_formatting_revision(ready, expected)
            return jsonify(receipt(CUBE_SLUG, result.pop("fingerprint"), slides=16, audio_clips=16,
                                   sort_order=28, **{key: value for key, value in result.items() if key != "slug"}))
        except library_import.ImportConflict:
            return error("Existing content or review state conflicts; nothing was changed", 409)
        except Exception:
            return error("Storage outcome is unconfirmed; reconcile the private receipt before retrying", 503)
