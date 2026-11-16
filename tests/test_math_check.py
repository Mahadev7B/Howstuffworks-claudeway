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


@pytest.mark.parametrize("question", ["how to make fractions into decimals", "explain 1/4 + 3/5", "what is 7 times 8",
                                      "Teach me decimals and fractions", "what is a prime number", "how do you find the area of a square"])
def test_math_questions_are_recognised(question):
    from lesson_platform.math_check import is_math_question
    assert is_math_question(question)


@pytest.mark.parametrize("question", ["How to solve 2*2 Rubik's cube", "Why is the sky blue?", "Who is number three in Cars",
                                      "What are all India's prime ministers", "How do planes fly?", "Is the 14th Prime Minister alive?"])
def test_other_questions_are_not_math(question):
    from lesson_platform.math_check import is_math_question
    assert not is_math_question(question)


def test_math_questions_use_the_math_model(monkeypatch, settings):
    client = FakeClient([lesson("1/4 + 1/2 = 3/4")])
    used = []
    original = client.create
    client.messages.create = lambda **kw: used.append(kw["model"]) or original(**kw)
    monkeypatch.setattr(ai_generation, "Anthropic", lambda api_key: client)
    result = ai_generation.generate_lesson("explain 1/4 + 1/2", settings)
    assert used == ["claude-sonnet-4-6"] and result["meta"]["model"] == "claude-sonnet-4-6" and result["meta"]["math"]


def test_other_questions_keep_the_normal_model(monkeypatch, settings):
    client = FakeClient([lesson("Planes have wings.")])
    used = []
    original = client.create
    client.messages.create = lambda **kw: used.append(kw["model"]) or original(**kw)
    monkeypatch.setattr(ai_generation, "Anthropic", lambda api_key: client)
    result = ai_generation.generate_lesson("How do planes fly?", settings)
    assert used == [settings.anthropic_model] and "math" not in result["meta"]


def test_quiz_model_is_chosen_from_the_lesson_words_not_its_meta(monkeypatch, settings):
    used = []

    def create(**kw):
        used.append(kw["model"])
        quiz = {"questions": [{"question": "q", "options": ["a", "b", "c"], "correct_index": 0, "explanation": "e"}] * 3}
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                               content=[SimpleNamespace(type="tool_use", input=quiz)])

    monkeypatch.setattr(ai_generation, "Anthropic", lambda api_key: SimpleNamespace(messages=SimpleNamespace(create=create)))
    slides = [{"number": 1, "title": "t", "explanation": "e"}]
    ai_generation.generate_quiz({"title": "Adding Fractions", "slides": slides}, settings)
    ai_generation.generate_quiz({"title": "How Planes Fly", "meta": {"math": True}, "slides": slides}, settings)
    assert used == ["claude-sonnet-4-6", settings.anthropic_model]


def test_prompt_has_math_rules():
    assert "Math questions" in ai_generation.SYSTEM_PROMPT
    assert "Never write \"1/3 = 0.33\"" in ai_generation.SYSTEM_PROMPT
    assert "math question" in ai_generation._QUIZ_SYSTEM
