"""Local editorial assets must validate without a database or paid generation."""

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from lesson_platform.editorial_package import load_editorial_batch
from lesson_platform.library import prepare_publication, publication_audio_text


def _batch(tmp_path):
    Image.new("RGB", (640, 400), "#234567").save(tmp_path / "slide.png")
    Image.new("RGB", (640, 400), "#d44a30").save(tmp_path / "cover.png")
    batch = [{
        "slug": "a-sky-lesson", "question": "Why is the sky blue?",
        "title": "Why Is the Sky Blue?", "summary": "Discover light and air.",
        "category": "Earth & Nature", "sort_order": 1, "featured_rank": 1,
        "cover": "cover.png", "slides": [{
            "title": "Sunlight", "explanation": "Sunlight has many colors.",
            "fun_fact": "A prism can reveal them.", "narration": "Sunlight has many colors.",
            "image": "slide.png",
        }],
    }]
    (tmp_path / "batch.json").write_text(json.dumps(batch), encoding="utf-8")
    return tmp_path / "batch.json"


def test_local_package_loads_distinct_cover_without_db(tmp_path):
    entry = load_editorial_batch(_batch(tmp_path))[0]
    assert entry["slug"] == "a-sky-lesson"
    assert len(entry["lesson"]["slides"]) == 1
    prepared = prepare_publication(entry["lesson"], cover_data_url=entry["cover_data_url"])
    assert prepared.thumbnail_mime == "image/webp"
    assert entry["cover_data_url"] != entry["lesson"]["slides"][0]["image_data_url"]


def test_local_package_rejects_paths_outside_asset_folder(tmp_path):
    path = _batch(tmp_path)
    batch = json.loads(path.read_text(encoding="utf-8"))
    batch[0]["cover"] = "../secret.png"
    path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        load_editorial_batch(path)


def test_local_package_requires_complete_slide_copy(tmp_path):
    path = _batch(tmp_path)
    batch = json.loads(path.read_text(encoding="utf-8"))
    del batch[0]["slides"][0]["narration"]
    path.write_text(json.dumps(batch), encoding="utf-8")
    with pytest.raises(ValueError, match="narration"):
        load_editorial_batch(path)


def test_first_six_batch_uses_reviewed_assets_and_copy():
    manifest = Path(__file__).parents[1] / "assets" / "editorial" / "first-six-lessons.json"
    entries = load_editorial_batch(manifest)
    assert len(entries) == 6
    assert sum(len(entry["lesson"]["slides"]) for entry in entries) == 24

    source = json.loads(manifest.read_text(encoding="utf-8"))
    lessons = {entry["slug"]: entry for entry in source}
    assert lessons["why-is-the-sky-blue"]["slides"][2]["image"].endswith("blue-scattering-v3.png")
    assert lessons["how-does-a-volcano-erupt"]["slides"][0]["image"].endswith("underground-magma-v2.png")
    assert lessons["how-does-a-volcano-erupt"]["slides"][1]["image"].endswith("surface-lava-v2.png")
    assert lessons["how-the-internet-works"]["slides"][1]["image"].endswith("data-packets-v2.png")
    assert lessons["how-the-internet-works"]["slides"][3]["image"].endswith("built-from-data-v2.png")
    rocket_copy = lessons["how-do-rockets-fly"]["slides"]
    assert "coast" in rocket_copy[0]["narration"]
    assert "Rockets like these carry both fuel and an oxidizer" in rocket_copy[2]["narration"]
    diwali_slides = lessons["how-diwali-is-celebrated"]["slides"]
    expanded_diwali_opening = (
        "Diwali, also called Deepavali, is the festival of lights, celebrated in India and around the world. "
        "It celebrates the victory of light over darkness and good over evil. Glowing lamps brighten homes as "
        "families gather to share joy and welcome new beginnings."
    )
    assert diwali_slides[0]["narration"] == expanded_diwali_opening
    assert diwali_slides[0]["explanation"] == expanded_diwali_opening
    diwali_copy = diwali_slides[1]
    unified_diwali_copy = (
        "Diwali celebrations bring families together for prayers, glowing diyas, colorful rangoli, "
        "sweets, festive meals, and fireworks."
    )
    assert diwali_copy["narration"] == unified_diwali_copy
    assert diwali_copy["explanation"] == unified_diwali_copy
    assert "Some families" not in diwali_copy["narration"]
    assert "Others" not in diwali_copy["narration"]

    displayed_scripts = []
    spoken_scripts = []
    for entry in entries:
        prepared = prepare_publication(entry["lesson"], cover_data_url=entry["cover_data_url"])
        for slide in prepared.lesson["slides"]:
            displayed_scripts.append(" ".join((slide["publication_narration"],
                                                "Fun fact:",
                                                slide["publication_fun_fact"])))
            spoken_scripts.append(publication_audio_text(slide))
    assert len(displayed_scripts) == 24
    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    assert [normalize(text) for text in displayed_scripts] == [normalize(text) for text in spoken_scripts]
