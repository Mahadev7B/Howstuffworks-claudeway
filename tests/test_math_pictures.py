"""Math lesson pictures never ask the picture tool for numbers or sums; other lessons are unchanged."""

import importlib

app = importlib.import_module("app")
SUFFIX = ", " + app._FLUX_STYLE_SUFFIX


def clean(prompt, title="What Are Decimals?", math=True):
    return app._sanitize_flux_prompt(prompt + SUFFIX, title, "fractions", math=math).replace(SUFFIX, "")


def test_numbers_and_sums_are_dropped_but_the_scene_stays():
    out = clean("A chocolate bar divided in half, with 0.5 displayed clearly. One half is shaded, the other is plain. "
                "Warm, friendly chocolate bar illustration with a simple background")
    assert "0.5" not in out and "displayed" not in out and "chocolate bar divided in half" in out


def test_counts_become_words():
    assert clean("A round pizza cut into 8 equal slices, with 2 slices highlighted in a different color, warm pizza "
                 "parlor setting").startswith("A round pizza cut into eight equal slices, with two slices")


def test_a_number_line_prompt_becomes_the_safe_scene_not_the_slide_title():
    out = clean("A number line showing 0.5 marked in the middle between 0 and 1, soft blue background")
    assert out == app._SAFE_MATH_SCENE
    assert "decimal" not in out.lower()


def test_other_lessons_keep_the_old_behaviour():
    prompt = "A dignified woman in 1700s colonial American dress writing a letter by candlelight" + SUFFIX
    assert app._sanitize_flux_prompt(prompt, "Martha", "history") == prompt


def test_math_lessons_are_marked_for_the_picture_step(monkeypatch):
    monkeypatch.setattr(app.settings, "image_provider", "none")
    lesson = {"title": "Adding Fractions", "subject": "fractions", "slides": [{"title": "S", "image_prompt": "pie"}]}
    monkeypatch.setattr(app, "_render_with_matplotlib", lambda slide: True)
    app._attach_slide_images(lesson)
    assert lesson["slides"][0]["math_lesson"] is True


def test_prompt_tells_claude_math_pictures_are_objects_only():
    from lesson_platform.ai_generation import SYSTEM_PROMPT
    assert "Math image_prompt" in SYSTEM_PROMPT and "number lines" in SYSTEM_PROMPT
