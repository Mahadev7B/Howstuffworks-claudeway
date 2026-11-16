"""Lesson arithmetic is checked exactly, and a wrong lesson gets one more try (no real Claude calls)."""

from types import SimpleNamespace

import pytest

from lesson_platform import ai_generation
from lesson_platform.config import load_settings
from lesson_platform.math_check import check_lesson, problems_in_text

RIGHT = ["1/4 + 3/5 = 17/20", "3 times 4 is 12", "Five plus twelve equals seventeen", "1 ÷ 4 = 0.25",
         "1 ÷ 3 = 0.333...", "1/3 is about 0.33", "2 + 3 × 4 = 14", "1/4 × 2/8 = 1/16", "3 x 4 = 12",
         "1,000 + 1 = 1,001", "1/4 + 2/8 = 2/8 + 2/8 = 4/8", "Practice: 0.5 + 0.25 = 0.75.",
         "kids aged 6-10 is fun", "On 2026-11-16 we", "We multiply 4 and 5 together to get 20", "5 is 5"]
WRONG = ["1/4 + 3/5 = 4/9", "3 times 4 is 13", "five plus twelve equals eighteen", "1 ÷ 3 = 0.33",
         "2 + 3 × 4 = 20", "8 / 2 = 5", "3/4 = 0.7", "1/4 + 2/8 = 2/8 + 2/8 = 5/8"]


@pytest.mark.parametrize("text", RIGHT)
def test_correct_or_non_math_text_passes(text):
    assert problems_in_text(text) == []


@pytest.mark.parametrize("text", WRONG)
def test_wrong_sums_are_caught(text):
    assert len(problems_in_text(text)) == 1


def test_problems_name_the_slide_and_the_right_answer():
    lesson = {"title": "Fractions", "slides": [{"explanation": "Easy."}, {"narration": "So 1/2 + 1/4 = 2/6."}]}
    [problem] = check_lesson(lesson)
    assert problem.startswith("Slide 2:") and "3/4" in problem


def lesson(text):
    return {"title": "Adding fractions", "subject": "fractions", "slides": [
        {"number": n, "title": f"S{n}", "explanation": text if n == 3 else "Fractions are parts.",
         "fun_fact": "Fun.", "narration": "Listen.", "image_prompt": "A pie."} for n in range(1, 5)]}


class FakeClient:
    def __init__(self, drafts):
        self.drafts, self.requests = list(drafts), []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs["messages"][0]["content"])
        draft = self.drafts.pop(0)
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=50),
                               content=[SimpleNamespace(type="tool_use", input=draft)])


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return load_settings()


def run(monkeypatch, settings, drafts):
    client = FakeClient(drafts)
    monkeypatch.setattr(ai_generation, "Anthropic", lambda api_key: client)
    return ai_generation.generate_lesson("explain 1/4 + 1/2", settings), client


def test_correct_lesson_needs_only_one_call(monkeypatch, settings):
    result, client = run(monkeypatch, settings, [lesson("1/4 + 1/2 = 3/4")])
    assert len(client.requests) == 1
    assert "math_retried" not in result["meta"] and "math_problems" not in result["meta"]


def test_wrong_sum_is_retried_with_the_mistake_named(monkeypatch, settings):
    result, client = run(monkeypatch, settings, [lesson("1/4 + 1/2 = 2/6"), lesson("1/4 + 1/2 = 3/4")])
    assert len(client.requests) == 2
    assert "1/4 + 1/2 = 2/6" in client.requests[1] and "3/4" in client.requests[1]
    assert result["slides"][2]["explanation"] == "1/4 + 1/2 = 3/4"
    assert result["meta"]["math_retried"] is True and "math_problems" not in result["meta"]
    assert result["meta"]["input_tokens"] == 200


def test_still_wrong_after_retry_is_flagged(monkeypatch, settings):
    result, _ = run(monkeypatch, settings, [lesson("1/4 + 1/2 = 2/6"), lesson("1/4 + 1/2 = 2/8")])
    assert result["meta"]["math_problems"]


def test_prompt_has_math_rules():
    assert "Math questions" in ai_generation.SYSTEM_PROMPT
    assert "Never write \"1/3 = 0.33\"" in ai_generation.SYSTEM_PROMPT
    assert "math question" in ai_generation._QUIZ_SYSTEM
