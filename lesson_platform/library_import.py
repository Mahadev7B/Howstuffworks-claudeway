"""Bounded, provider-free imports of one prepared private lesson and four clips."""
import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import db, library
from .private_import_protocol import private_slide_count

MAX_PACKET_BYTES = 12 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_PACKET_BYTES + 64 * 1024
MAX_AUDIO_BYTES = 1024 * 1024
MODEL = "Kokoro-82M ONNX v1.0"
VOICE = "af_heart"


class ImportConflict(ValueError):
    """An existing lesson must never be replaced by this insert-only action."""


class ImportUnavailable(RuntimeError):
    """Required local media validation is unavailable; fail closed."""


@dataclass(frozen=True)
class PreparedImport:
    entry: dict
    prepared: library.PreparedPublication
    clips: tuple[dict, ...]


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Package contains duplicate JSON keys")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Package contains a non-finite number")


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Package fields are missing or unsupported")


def _validate_mp3(audio: bytes, index: int, *, ffmpeg_path=None) -> None:
    executable = str(Path(ffmpeg_path).resolve(strict=True)) if ffmpeg_path is not None else shutil.which("ffmpeg")
    if not executable:
        raise ImportUnavailable("Existing FFmpeg is required to validate saved MP3 clips")
    try:
        result = subprocess.run(
            [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
             "-f", "mp3", "-i", "pipe:0", "-map", "0:a:0", "-t", "91",
             "-ac", "1", "-ar", "24000", "-f", "s16le", "pipe:1"],
            input=audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ImportUnavailable("Saved audio validation could not finish") from exc
    seconds = len(result.stdout) / (24000 * 2)
    if result.returncode or not 0.25 <= seconds <= 90:
        raise ValueError(f"Slide {index + 1} needs a decodable MP3 between 0.25 and 90 seconds")


def prepare_import(raw: bytes, *, ffmpeg_path=None) -> PreparedImport:
    """Validate exact stored text/images/audio, without a DB or provider call."""
    if not raw or len(raw) > MAX_PACKET_BYTES:
        raise ValueError("Choose one JSON package no larger than 12 MiB")
    try:
        packet = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Choose a valid UTF-8 JSON lesson package") from exc
    _keys(packet, ("entry", "clips", "player_rate"))
    if isinstance(packet["player_rate"], bool) or packet["player_rate"] != 0.9:
        raise ValueError("Package playback rate must be 0.9; saved audio must not be slowed")
    entry = packet["entry"]
    _keys(entry, ("slug", "question", "title", "summary", "category", "sort_order",
                  "featured_rank", "lesson", "cover_data_url"))
    for key in ("slug", "question", "title", "summary", "category"):
        if not isinstance(entry[key], str) or entry[key] != entry[key].strip():
            raise ValueError("Metadata must be nonempty, trimmed text")
    if len(entry["question"]) > 200:
        raise ValueError("Question is too long")
    if type(entry["sort_order"]) is not int or not 1 <= entry["sort_order"] <= 10000:
        raise ValueError("Choose a valid positive sort order")
    if entry["featured_rank"] is not None and (type(entry["featured_rank"]) is not int or not 1 <= entry["featured_rank"] <= 10000):
        raise ValueError("Featured rank must be a positive integer or null")
    library._validate_metadata(*(entry[key] for key in (
        "question", "slug", "title", "summary", "category", "sort_order", "featured_rank")))
    lesson = entry["lesson"]
    slide_count = private_slide_count(entry['slug'], entry['sort_order'])
    _keys(lesson, ("title", "slides"))
    if lesson["title"] != entry["title"] or not isinstance(lesson["slides"], list) or len(lesson["slides"]) != slide_count:
        raise ValueError(f"Package needs one matching lesson title and exactly {slide_count} slides")
    for slide in lesson["slides"]:
        _keys(slide, ("title", "explanation", "fun_fact", "narration", "image_data_url"),
              ("subtitle", "publication_narration", "publication_fun_fact", "publication_script"))
        if any(not isinstance(slide[key], str) or not slide[key].strip() or len(slide[key]) > 2000
               for key in ("title", "explanation", "fun_fact", "narration")):
            raise ValueError("Every slide needs bounded complete text")
        expected = library.with_publication_scripts({"slides": [slide]})["slides"][0]
        if any(key in slide and slide[key] != expected[key]
               for key in ("publication_narration", "publication_fun_fact", "publication_script")):
            raise ValueError("Displayed and spoken slide text must match")
    prepared = library.prepare_publication(lesson, cover_data_url=entry["cover_data_url"])
    if not isinstance(packet["clips"], list) or len(packet["clips"]) != slide_count:
        raise ValueError(f"Package needs exactly {slide_count} saved narration clips")
    clips = []
    for index, clip in enumerate(packet["clips"]):
        _keys(clip, ("slide_index", "script", "source_text_hash", "mp3", "mp3_sha256", "model", "voice", "speed"))
        script = library.publication_audio_text(lesson["slides"][index])
        if type(clip["slide_index"]) is not int or clip["slide_index"] != index:
            raise ValueError("Audio indices must cover every slide exactly in order")
        if clip["script"] != script or clip["source_text_hash"] != hashlib.sha256(script.encode()).hexdigest():
            raise ValueError("Saved narration source must match the canonical slide text including Fun fact")
        if (clip["model"], clip["voice"]) != (MODEL, VOICE) or isinstance(clip["speed"], bool) or clip["speed"] != 1.0:
            raise ValueError("Use saved Kokoro Heart at natural speed 1.0, not double-slowed audio")
        encoded = clip["mp3"]
        if not isinstance(encoded, str) or not encoded or len(encoded) > ((MAX_AUDIO_BYTES + 2)//3)*4:
            raise ValueError("Each saved MP3 must be nonempty and no larger than 1 MiB")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Saved MP3 has invalid base64") from exc
        if not audio or len(audio) > MAX_AUDIO_BYTES or hashlib.sha256(audio).hexdigest() != clip["mp3_sha256"]:
            raise ValueError("Saved MP3 integrity check failed")
        if ffmpeg_path is None:
            _validate_mp3(audio, index)
        else:
            _validate_mp3(audio, index, ffmpeg_path=ffmpeg_path)
        clips.append({"mp3": audio, "model": MODEL, "voice": VOICE, "source_text_hash": clip["source_text_hash"]})
    return PreparedImport(entry, prepared, tuple(clips))


def _existing_matches(cur, ready: PreparedImport):
    entry = ready.entry
    cur.execute("""SELECT source_question_hash,question,title,summary,category,sort_order,featured_rank,
                   cover_slide_index,lesson,thumbnail,thumbnail_mime,active,approved_for_publish
                   FROM published_lessons WHERE slug=%s FOR UPDATE""", (entry["slug"],))
    row = cur.fetchone()
    if row is None:
        return None
    expected = (db.question_hash(entry["question"]), *(entry[key] for key in (
        "question", "title", "summary", "category", "sort_order", "featured_rank")),
        0, ready.prepared.lesson, ready.prepared.thumbnail, ready.prepared.thumbnail_mime, False, False)
    actual = (*row[:9], bytes(row[9]), *row[10:])
    if actual != expected:
        return False
    cur.execute("""SELECT slide_index,mp3,model,voice,source_text_hash FROM published_lesson_audio
                   WHERE slug=%s ORDER BY slide_index""", (entry["slug"],))
    audio = [(row[0], bytes(row[1]), *row[2:]) for row in cur.fetchall()]
    return audio == [(index, c["mp3"], c["model"], c["voice"], c["source_text_hash"])
                     for index, c in enumerate(ready.clips)]


def save_import(ready: PreparedImport) -> dict:
    """Insert one complete draft atomically, or return an identical private draft unchanged."""
    entry = ready.entry
    with library._connect() as conn, conn.transaction(), conn.cursor() as cur:
        # Transaction-scoped built-in lock serializes concurrent importer retries.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("asklilowl-private-import",))
        matches = _existing_matches(cur, ready)
        if matches is not None:
            if not matches:
                raise ImportConflict("This slug already exists with different content or review/publication state; nothing was changed")
            return {"slug": entry["slug"], "created": False}
        cur.execute("SELECT slug FROM published_lessons WHERE sort_order=%s", (entry["sort_order"],))
        if cur.fetchone():
            raise ImportConflict("That sort order already belongs to another lesson; nothing was changed")
        library._insert_publication(cur, ready.prepared, *(entry[key] for key in (
            "question", "slug", "title", "summary", "category", "sort_order", "featured_rank")), 0)
        for index, clip in enumerate(ready.clips):
            cur.execute("""INSERT INTO published_lesson_audio
                (slug,slide_index,mp3,model,voice,source_text_hash) VALUES (%s,%s,%s,%s,%s,%s)""",
                (entry["slug"], index, clip["mp3"], clip["model"], clip["voice"], clip["source_text_hash"]))
        if _existing_matches(cur, ready) is not True:
            raise RuntimeError("Private import verification failed; transaction rolled back")
    return {"slug": entry["slug"], "created": True}
