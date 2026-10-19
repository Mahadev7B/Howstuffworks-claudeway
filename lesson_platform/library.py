"""Curated public lesson snapshots; no generation occurs in this module."""

import base64
import binascii
import copy
import hashlib
import re
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError
from psycopg.types.json import Jsonb

from lesson_platform import db


_RASTER_MIMES = {"image/png", "image/jpeg", "image/webp"}
_MAX_IMAGE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class PreparedPublication:
    lesson: dict
    thumbnail: bytes
    thumbnail_mime: str


def publication_audio_text(slide: dict) -> str:
    """Return the complete spoken copy for one curated slide."""
    narration = slide["narration"].strip()
    fun_fact = slide.get("fun_fact", "").strip()
    return f"{narration} Fun fact: {fun_fact}" if fun_fact else narration


def with_publication_scripts(lesson: dict) -> dict:
    """Return a display snapshot whose script is exactly the saved TTS input."""
    result = copy.deepcopy(lesson)
    for slide in result.get("slides", []):
        slide["publication_narration"] = slide["narration"].strip()
        slide["publication_fun_fact"] = slide.get("fun_fact", "").strip()
        slide["publication_script"] = publication_audio_text(slide)
    return result


def _decode_raster(data_url: str) -> Image.Image:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("Every slide needs a raster image")
    header, separator, payload = data_url.partition(",")
    mime = header[5:].split(";", 1)[0].lower()
    if not separator or mime not in _RASTER_MIMES or not header.endswith(";base64"):
        raise ValueError("Every slide needs a supported raster image")
    if len(payload) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 8:
        raise ValueError("Slide image is too large")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Slide image has invalid base64") from exc
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError("Slide image is too large")
    try:
        with Image.open(BytesIO(raw)) as image:
            if Image.MIME.get(image.format) != mime:
                raise ValueError("Slide image type does not match its raster data")
            if image.width * image.height > 16_000_000:
                raise ValueError("Slide image dimensions are too large")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Slide image is not a valid raster image") from exc


def prepare_publication(lesson: dict, cover_slide_index: int = 0,
                        cover_data_url: str | None = None) -> PreparedPublication:
    """Validate approved content and derive a stable, independent card image."""
    if not isinstance(lesson, dict) or not isinstance(lesson.get("slides"), list) or not lesson["slides"]:
        raise ValueError("Lesson needs slides")
    if not isinstance(lesson.get("title"), str) or not lesson["title"].strip():
        raise ValueError("Lesson needs a title")
    if (not isinstance(cover_slide_index, int) or isinstance(cover_slide_index, bool)
            or not 0 <= cover_slide_index < len(lesson["slides"])):
        raise ValueError("Choose a valid cover slide")
    first_image = None
    for index, slide in enumerate(lesson["slides"]):
        if not isinstance(slide, dict):
            raise ValueError("Every slide needs complete text")
        for field in ("title", "explanation", "fun_fact", "narration"):
            if not isinstance(slide.get(field), str) or not slide[field].strip():
                raise ValueError(f"Every slide needs {field}")
        image = _decode_raster(slide.get("image_data_url"))
        if index == cover_slide_index:
            first_image = image
    output = BytesIO()
    cover_image = _decode_raster(cover_data_url) if cover_data_url is not None else first_image
    ImageOps.fit(cover_image, (480, 300), method=Image.Resampling.LANCZOS).save(output, "WEBP", quality=84)
    return PreparedPublication(lesson=with_publication_scripts(lesson),
                               thumbnail=output.getvalue(), thumbnail_mime="image/webp")


def _connect():
    conn = db._direct_connect()
    if conn is None:
        raise RuntimeError("Database is unavailable")
    return conn


def _validate_metadata(question: str, slug: str, title: str, summary: str,
                       category: str, sort_order: int,
                       featured_rank: int | None) -> None:
    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        raise ValueError("Slug must contain lowercase words and hyphens")
    for label, value in (("question", question), ("title", title),
                         ("summary", summary), ("category", category)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} is required")
    if len(title) > 120 or len(summary) > 280 or len(category) > 60:
        raise ValueError("Publication metadata is too long")
    if not isinstance(sort_order, int) or isinstance(sort_order, bool):
        raise ValueError("sort_order must be an integer")
    if featured_rank is not None and (not isinstance(featured_rank, int) or featured_rank < 1):
        raise ValueError("featured_rank must be positive")


def _insert_publication(cur, prepared: PreparedPublication, question: str,
                        slug: str, title: str, summary: str, category: str,
                        sort_order: int, featured_rank: int | None,
                        cover_slide_index: int) -> dict:
    cur.execute("SELECT slug FROM published_lessons WHERE slug = %s", (slug,))
    if cur.fetchone() is not None:
        raise ValueError("Publication slug already exists")
    cur.execute(
        """INSERT INTO published_lessons
           (slug, source_question_hash, question, title, summary, category,
            sort_order, featured_rank, cover_slide_index, lesson, thumbnail, thumbnail_mime)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (slug, db.question_hash(question), question.strip(), title.strip(), summary.strip(), category.strip(),
         sort_order, featured_rank, cover_slide_index, Jsonb(prepared.lesson), prepared.thumbnail,
         prepared.thumbnail_mime),
    )
    return {"slug": slug, "active": False, "thumbnail_mime": prepared.thumbnail_mime}


def stage_cached_lesson(question: str, slug: str, title: str, summary: str,
                        category: str, sort_order: int = 0,
                        featured_rank: int | None = None,
                        cover_slide_index: int = 0) -> dict:
    """Copy a fully illustrated cache entry into a private, stable draft."""
    _validate_metadata(question, slug, title, summary, category, sort_order, featured_rank)
    qhash = db.question_hash(question)
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT lesson FROM cached_lessons WHERE question_hash = %s", (qhash,))
        row = cur.fetchone()
        if row is None:
            raise ValueError("Cached lesson was not found")
        prepared = prepare_publication(row[0], cover_slide_index=cover_slide_index)
        return _insert_publication(cur, prepared, question, slug, title, summary,
                                   category, sort_order, featured_rank, cover_slide_index)


def stage_editorial_lesson(lesson: dict, question: str, slug: str, title: str,
                           summary: str, category: str, sort_order: int = 0,
                           featured_rank: int | None = None,
                           cover_data_url: str | None = None) -> dict:
    """Stage a locally authored lesson privately, without editing the ordinary cache."""
    _validate_metadata(question, slug, title, summary, category, sort_order, featured_rank)
    prepared = prepare_publication(lesson, cover_data_url=cover_data_url)
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        return _insert_publication(cur, prepared, question, slug, title, summary,
                                   category, sort_order, featured_rank, 0)


def set_publication_active(slug: str, active: bool) -> bool:
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        if active:
            cur.execute(
                "SELECT lesson, approved_for_publish FROM published_lessons WHERE slug = %s FOR UPDATE", (slug,))
            row = cur.fetchone()
            if row is None:
                return False
            if not row[1]:
                raise ValueError("Lesson must be approved before publication")
            slides = row[0].get("slides", [])
            cur.execute("SELECT slide_index, source_text_hash FROM published_lesson_audio WHERE slug = %s", (slug,))
            hashes = dict(cur.fetchall())
            if not slides or any(hashes.get(index) != hashlib.sha256(publication_audio_text(slide).encode("utf-8")).hexdigest()
                                 for index, slide in enumerate(slides)):
                raise ValueError("Every slide needs saved audio before activation")
        cur.execute(
            """UPDATE published_lessons
               SET active = %s, published_at = CASE WHEN %s THEN COALESCE(published_at, NOW()) ELSE published_at END
               WHERE slug = %s RETURNING slug""", (active, active, slug))
        return cur.fetchone() is not None


def set_publication_approved(slug: str, approved: bool) -> bool:
    """Record editorial approval without changing public visibility."""
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """UPDATE published_lessons
               SET approved_for_publish = %s
               WHERE slug = %s AND active = false
               RETURNING slug""",
            (approved, slug),
        )
        return cur.fetchone() is not None


def list_published_cards() -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT slug, title, summary, category, sort_order, featured_rank
               FROM published_lessons WHERE active = true
               ORDER BY sort_order, slug""")
        return [dict(zip(("slug", "title", "summary", "category", "sort_order", "featured_rank"), row))
                for row in cur.fetchall()]


def get_published_lesson(slug: str) -> dict | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT lesson, title, summary, category FROM published_lessons
               WHERE slug = %s AND active = true""", (slug,))
        row = cur.fetchone()
        return dict(zip(("lesson", "title", "summary", "category"), row)) if row else None


def get_published_thumbnail(slug: str) -> tuple[bytes, str] | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT thumbnail, thumbnail_mime FROM published_lessons
               WHERE slug = %s AND active = true""", (slug,))
        row = cur.fetchone()
        return (bytes(row[0]), row[1]) if row else None


def get_published_audio(slug: str, slide_index: int) -> bytes | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT a.mp3 FROM published_lesson_audio a
               JOIN published_lessons p ON p.slug = a.slug
               WHERE p.slug = %s AND p.active = true AND a.slide_index = %s""",
            (slug, slide_index))
        row = cur.fetchone()
        return bytes(row[0]) if row else None


def get_staged_lesson(slug: str) -> dict | None:
    """Editorial-only read; never expose this through a public route."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT lesson, active FROM published_lessons WHERE slug = %s", (slug,))
        row = cur.fetchone()
        return row[0] if row and not row[1] else None


def get_editorial_publication(slug: str) -> dict | None:
    """Return a private publication to an authenticated editorial route."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT lesson, title, summary, category, active
               FROM published_lessons WHERE slug = %s AND active = false""", (slug,))
        row = cur.fetchone()
        return dict(zip(("lesson", "title", "summary", "category", "active"), row)) if row else None


def get_audio_metadata(slug: str) -> dict[int, dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT slide_index, model, voice, source_text_hash
                       FROM published_lesson_audio WHERE slug = %s""", (slug,))
        return {row[0]: {"model": row[1], "voice": row[2], "source_text_hash": row[3]}
                for row in cur.fetchall()}


def save_publication_audio(slug: str, slide_index: int, mp3: bytes,
                           model: str, voice: str, source_text_hash: str) -> None:
    if not mp3:
        raise ValueError("Generated audio is empty")
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT active FROM published_lessons WHERE slug = %s FOR UPDATE", (slug,))
        row = cur.fetchone()
        if row is None:
            raise ValueError("Publication was not found")
        if row[0]:
            raise ValueError("Only a private draft's audio can be changed")
        cur.execute(
            """INSERT INTO published_lesson_audio
               (slug, slide_index, mp3, model, voice, source_text_hash)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (slug, slide_index) DO UPDATE
               SET mp3 = EXCLUDED.mp3, model = EXCLUDED.model, voice = EXCLUDED.voice,
                   source_text_hash = EXCLUDED.source_text_hash""",
            (slug, slide_index, mp3, model, voice, source_text_hash))


def list_editorial_publications() -> list[dict]:
    """Admin metadata only; no lesson artwork or audio blobs in the dashboard."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT p.slug, p.title, p.active, p.approved_for_publish,
                      jsonb_array_length(p.lesson->'slides'),
                      COUNT(a.slide_index) AS clips, p.cover_slide_index,
                      ARRAY(SELECT slide->>'fun_fact'
                            FROM jsonb_array_elements(p.lesson->'slides') WITH ORDINALITY AS item(slide, position)
                            ORDER BY position) AS fun_facts
               FROM published_lessons p
               LEFT JOIN published_lesson_audio a ON a.slug = p.slug
               GROUP BY p.slug ORDER BY p.sort_order, p.slug""")
        return [dict(zip(("slug", "title", "active", "approved_for_publish", "slide_count", "clip_count", "cover_slide_index", "fun_facts"), row))
                for row in cur.fetchall()]


def get_editorial_audio(slug: str, slide_index: int) -> bytes | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT mp3 FROM published_lesson_audio WHERE slug = %s AND slide_index = %s",
                    (slug, slide_index))
        row = cur.fetchone()
        return bytes(row[0]) if row else None


def get_editorial_thumbnail(slug: str) -> tuple[bytes, str] | None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT thumbnail, thumbnail_mime FROM published_lessons WHERE slug = %s", (slug,))
        row = cur.fetchone()
        return (bytes(row[0]), row[1]) if row else None


def update_publication_cover(slug: str, cover_slide_index: int) -> bool:
    """Let an editor improve a private draft's cover without restaging it."""
    with _connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT lesson, active FROM published_lessons WHERE slug = %s FOR UPDATE", (slug,))
        row = cur.fetchone()
        if row is None:
            return False
        if row[1]:
            raise ValueError("Only a private draft's cover can be changed")
        prepared = prepare_publication(row[0], cover_slide_index=cover_slide_index)
        cur.execute("""UPDATE published_lessons
                       SET cover_slide_index = %s, thumbnail = %s, thumbnail_mime = %s
                       WHERE slug = %s""",
                    (cover_slide_index, prepared.thumbnail, prepared.thumbnail_mime, slug))
        return True
