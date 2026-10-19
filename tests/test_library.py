"""Publication validation must reject incomplete or unsafe cached lessons."""

import base64
import hashlib
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from lesson_platform import library
from lesson_platform.library import prepare_publication


def _image_data_url(color="#345678") -> str:
    output = io.BytesIO()
    Image.new("RGB", (640, 400), color).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _lesson(image_url: str | None = None) -> dict:
    image_url = image_url if image_url is not None else _image_data_url()
    return {
        "title": "Why is the sky blue?",
        "slides": [
            {"title": "Sunlight", "explanation": "Sunlight contains colors.", "fun_fact": "Light is fast.",
             "narration": "Sunlight contains many colors.",
             "image_data_url": image_url},
            {"title": "Air", "explanation": "Air scatters light.", "fun_fact": "The sky changes.",
             "narration": "Air scatters blue light.",
             "image_data_url": image_url},
        ],
    }


def test_prepare_publication_produces_webp_card():
    prepared = prepare_publication(_lesson())
    assert prepared.thumbnail_mime == "image/webp"
    image = Image.open(io.BytesIO(prepared.thumbnail))
    assert image.format == "WEBP"
    assert image.size == (480, 300)
    assert prepared.lesson["slides"][0]["narration"] == "Sunlight contains many colors."


def test_canonical_publication_script_is_identical_for_display_and_tts():
    lesson = library.with_publication_scripts(_lesson())

    assert [slide["publication_script"] for slide in lesson["slides"]] == [
        "Sunlight contains many colors. Fun fact: Light is fast.",
        "Air scatters blue light. Fun fact: The sky changes.",
    ]
    assert all(slide["publication_script"] == library.publication_audio_text(slide)
               for slide in lesson["slides"])
    assert [(slide["publication_narration"], slide["publication_fun_fact"])
            for slide in lesson["slides"]] == [
        ("Sunlight contains many colors.", "Light is fast."),
        ("Air scatters blue light.", "The sky changes."),
    ]


def test_editor_can_choose_representative_cover_slide():
    lesson = _lesson()
    lesson["slides"][1]["image_data_url"] = _image_data_url("#ea4c3a")
    prepared = prepare_publication(lesson, cover_slide_index=1)
    pixel = Image.open(io.BytesIO(prepared.thumbnail)).getpixel((240, 150))
    assert pixel[0] > 200 and pixel[1] < 120
    with pytest.raises(ValueError, match="cover"):
        prepare_publication(lesson, cover_slide_index=2)


def test_editorial_cover_can_differ_from_slide_art():
    prepared = prepare_publication(_lesson(), cover_data_url=_image_data_url("#e94c32"))
    pixel = Image.open(io.BytesIO(prepared.thumbnail)).getpixel((240, 150))
    assert pixel[0] > 200 and pixel[1] < 120
    assert prepared.lesson["slides"][0]["image_data_url"] != _image_data_url("#e94c32")


def test_editorial_cover_rejects_non_raster():
    with pytest.raises(ValueError, match="raster"):
        prepare_publication(_lesson(), cover_data_url="data:image/svg+xml;base64,PHN2Zy8+")


@pytest.mark.parametrize("lesson", [
    {"title": "Empty", "slides": []},
    {"title": "No narration", "slides": [{"title": "x", "image_data_url": "x"}]},
    {"title": "No image", "slides": [{"title": "x", "explanation": "Text", "fun_fact": "Fact", "narration": "Hi"}]},
])
def test_prepare_publication_rejects_incomplete_lessons(lesson):
    with pytest.raises(ValueError):
        prepare_publication(lesson)


def test_prepare_publication_rejects_svg():
    image = "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode("ascii")
    with pytest.raises(ValueError, match="raster"):
        prepare_publication(_lesson(image))


def test_prepare_publication_rejects_image_over_2mb():
    image = "data:image/png;base64," + base64.b64encode(b"x" * (2 * 1024 * 1024 + 1)).decode("ascii")
    with pytest.raises(ValueError, match="large"):
        prepare_publication(_lesson(image))


def _fake_db(monkeypatch, rows):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = rows
    monkeypatch.setattr(library.db, "_direct_connect", lambda: conn)
    return conn, cur


def test_stage_copies_complete_cache_snapshot(monkeypatch):
    source = _lesson()
    conn, cur = _fake_db(monkeypatch, [(source,), None])
    staged = library.stage_cached_lesson("Why is the sky blue?", "sky-blue", "Why is the sky blue?",
                                         "Discover light and air.", "Space & Science", 1, 1)
    assert staged["slug"] == "sky-blue"
    assert staged["active"] is False
    assert staged["thumbnail_mime"] == "image/webp"
    assert any("INSERT INTO published_lessons" in call.args[0] for call in cur.execute.call_args_list)
    assert conn.transaction.called


def test_stage_rejects_missing_artwork(monkeypatch):
    source = _lesson()
    del source["slides"][0]["image_data_url"]
    _, cur = _fake_db(monkeypatch, [(source,)])
    with pytest.raises(ValueError, match="raster"):
        library.stage_cached_lesson("Why sky?", "sky", "Why sky?", "A short summary", "Science", 0, None)
    assert not any("INSERT INTO published_lessons" in call.args[0] for call in cur.execute.call_args_list)


def test_stage_rejects_duplicate_slug(monkeypatch):
    _, cur = _fake_db(monkeypatch, [(_lesson(),), ("sky-blue",)])
    with pytest.raises(ValueError, match="already exists"):
        library.stage_cached_lesson("Why sky?", "sky-blue", "Why sky?", "A short summary", "Science", 0, None)
    assert not any("INSERT INTO published_lessons" in call.args[0] for call in cur.execute.call_args_list)


def test_stage_editorial_lesson_without_mutating_cache(monkeypatch):
    _, cur = _fake_db(monkeypatch, [None])
    staged = library.stage_editorial_lesson(
        _lesson(), "Why is the sky blue?", "sky-blue", "Why is the sky blue?",
        "Discover light and air.", "Earth & Nature", 1, 1,
        cover_data_url=_image_data_url("#e94c32"),
    )
    assert staged["slug"] == "sky-blue"
    assert not any("cached_lessons" in call.args[0] for call in cur.execute.call_args_list)
    assert any("INSERT INTO published_lessons" in call.args[0] for call in cur.execute.call_args_list)


def test_public_queries_filter_active_records(monkeypatch):
    _, cur = _fake_db(monkeypatch, [({"slides": []}, "Sky", "Summary", "Science"),
                                   (b"webp", "image/webp"), (b"mp3",)])
    cur.fetchall.return_value = [("sky", "Sky", "Summary", "Science", 0, 1)]
    cards = library.list_published_cards()
    detail = library.get_published_lesson("sky")
    thumbnail = library.get_published_thumbnail("sky")
    audio = library.get_published_audio("sky", 0)
    assert cards[0]["slug"] == "sky"
    assert detail["lesson"] == {"slides": []}
    assert thumbnail == (b"webp", "image/webp")
    assert audio == b"mp3"
    assert all("active = true" in c.args[0] for c in cur.execute.call_args_list)


def test_activation_requires_every_slide_clip(monkeypatch):
    _, cur = _fake_db(monkeypatch, [(_lesson(), True)])
    with pytest.raises(ValueError, match="audio"):
        library.set_publication_active("sky", True)
    assert not any("UPDATE published_lessons" in c.args[0] for c in cur.execute.call_args_list)


def test_activation_accepts_clips_that_include_the_fun_fact(monkeypatch):
    lesson = _lesson()
    _, cur = _fake_db(monkeypatch, [(lesson, True), ("sky",)])
    expected_texts = [
        "Sunlight contains many colors. Fun fact: Light is fast.",
        "Air scatters blue light. Fun fact: The sky changes.",
    ]
    cur.fetchall.return_value = [
        (index, hashlib.sha256(text.encode("utf-8")).hexdigest())
        for index, text in enumerate(expected_texts)
    ]

    assert library.set_publication_active("sky", True) is True
    assert any("UPDATE published_lessons" in c.args[0] for c in cur.execute.call_args_list)


def test_activation_rejects_unapproved_draft(monkeypatch):
    _, cur = _fake_db(monkeypatch, [(_lesson(), False)])

    with pytest.raises(ValueError, match="approved"):
        library.set_publication_active("sky", True)

    assert not any("SET active" in c.args[0] for c in cur.execute.call_args_list)


def test_private_draft_can_be_approved_without_becoming_active(monkeypatch):
    _, cur = _fake_db(monkeypatch, [("sky",)])

    assert library.set_publication_approved("sky", True) is True

    update = next(c for c in cur.execute.call_args_list if "UPDATE published_lessons" in c.args[0])
    assert "approved_for_publish" in update.args[0]
    assert "active = false" in update.args[0]
    assert update.args[1] == (True, "sky")


def test_public_draft_cannot_have_approval_changed(monkeypatch):
    _, cur = _fake_db(monkeypatch, [None])

    assert library.set_publication_approved("sky", False) is False
    assert any("active = false" in c.args[0] for c in cur.execute.call_args_list)


def test_draft_cover_can_be_changed_without_touching_lesson(monkeypatch):
    lesson = _lesson()
    lesson["slides"][1]["image_data_url"] = _image_data_url("#ea4c3a")
    _, cur = _fake_db(monkeypatch, [(lesson, False)])
    assert library.update_publication_cover("sky", 1) is True
    update = next(c for c in cur.execute.call_args_list if "UPDATE published_lessons" in c.args[0])
    assert update.args[1][0] == 1
    assert update.args[1][-1] == "sky"


def test_public_cover_cannot_be_changed(monkeypatch):
    _, cur = _fake_db(monkeypatch, [(_lesson(), True)])
    with pytest.raises(ValueError, match="private"):
        library.update_publication_cover("sky", 1)
    assert not any("UPDATE published_lessons" in c.args[0] for c in cur.execute.call_args_list)


def test_active_lesson_is_not_a_voice_generation_draft(monkeypatch):
    _, cur = _fake_db(monkeypatch, [(_lesson(), True)])
    assert library.get_staged_lesson("sky") is None
    assert "active" in cur.execute.call_args[0][0]


def test_active_lesson_rejects_audio_writes(monkeypatch):
    _, cur = _fake_db(monkeypatch, [(True,)])
    with pytest.raises(ValueError, match="private"):
        library.save_publication_audio("sky", 0, b"mp3", "model", "voice", "hash")
    assert not any("INSERT INTO published_lesson_audio" in call.args[0]
                   for call in cur.execute.call_args_list)
