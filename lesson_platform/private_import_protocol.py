"""Shared fingerprints and receipt checks for the insert-only private protocol."""
import hashlib
import json
import re

from . import db


MAX_PACKET_BYTES = 12 * 1024 * 1024
CUBE_SLUG = 'solve-a-2x2-rubiks-cube'
_TOKEN = re.compile(r"[A-Za-z0-9._~-]{32,512}\Z", re.ASCII)
_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z", re.ASCII)
_HASH = re.compile(r"[a-f0-9]{64}\Z", re.ASCII)
METADATA_FIELDS = ("slug", "source_question_hash", "question", "title", "summary", "category",
                   "sort_order", "featured_rank", "cover_slide_index")


def valid_token(value):
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def valid_slug(value):
    return isinstance(value, str) and len(value) <= 200 and _SLUG.fullmatch(value) is not None


def valid_fingerprint(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def preview_path(slug):
    return f"/admin/published-lessons/{slug}/preview"


def private_slide_count(slug, sort_order):
    """The held cube keeps its full draft; all other private imports use four."""
    if slug == CUBE_SLUG:
        if type(sort_order) is not int or sort_order != 28:
            raise ValueError('The sixteen-slide cube must retain sort order 28')
        return 16
    return 4


def fingerprint_stored(metadata, lesson, thumbnail, thumbnail_mime, clips):
    """Hash every canonical imported field; never include timestamps or raw receipt copy."""
    content = {
        "protocol": 1,
        "metadata": {field: metadata[field] for field in METADATA_FIELDS},
        "lesson": lesson,
        "thumbnail": {"sha256": hashlib.sha256(bytes(thumbnail)).hexdigest(), "mime": thumbnail_mime},
        "clips": [{"slide_index": c["slide_index"], "mp3_sha256": hashlib.sha256(bytes(c["mp3"])).hexdigest(),
                   "model": c["model"], "voice": c["voice"], "source_text_hash": c["source_text_hash"]}
                  for c in clips],
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_prepared(ready):
    metadata = {**ready.entry, "source_question_hash": db.question_hash(ready.entry["question"]),
                "cover_slide_index": 0}
    clips = [{**clip, "slide_index": index} for index, clip in enumerate(ready.clips)]
    return fingerprint_stored(metadata, ready.prepared.lesson, ready.prepared.thumbnail,
                              ready.prepared.thumbnail_mime, clips)


def valid_receipt(receipt, slug, expected_fingerprint=None):
    if not isinstance(receipt, dict):
        return False
    try:
        count = private_slide_count(slug, receipt.get('sort_order'))
    except ValueError:
        return False
    return (isinstance(receipt, dict) and receipt.get("slug") == slug
            and valid_fingerprint(receipt.get("fingerprint"))
            and (expected_fingerprint is None or receipt["fingerprint"] == expected_fingerprint)
            and receipt.get("active") is False and receipt.get("approved_for_publish") is False
            and type(receipt.get("slides")) is int and receipt["slides"] == count
            and type(receipt.get("audio_clips")) is int and receipt["audio_clips"] == count
            and receipt.get("preview_url") == preview_path(slug))
