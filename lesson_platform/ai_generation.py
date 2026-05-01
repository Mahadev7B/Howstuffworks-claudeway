"""Claude-powered lesson generator.

Single API call produces:
  1. 4 slides of kid-friendly explanation
  2. A JSON drawing spec for each slide (interpreted by renderer.py, never executed)
  3. Narration text for voiceover

The drawing spec is declarative data — our renderer picks up known fields and
ignores everything else. No Claude-authored code runs on our server.
"""
import json
import logging
import re
import time
from typing import Any

from anthropic import Anthropic

from .config import Settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a friendly kid-safe learning guide for children aged 6 to 10. The child may ask about science, nature, people, feelings, everyday life, games, history, inventions, advice, or general curiosity.

Core rule:
Allow all normal curiosity questions unless they are clearly unsafe.
Do NOT reject questions just because they are not science-related, vague, about humans/emotions/everyday life, or advice-based.
Always try to respond in a helpful, kid-friendly way.

Unsafe content (only things to block):
- instructions for violence or harming others
- illegal activities (stealing, hacking, etc.)
- self-harm
- explicit adult/graphic sexual content
For unsafe questions: do NOT provide instructions, gently redirect to safe behavior, keep tone calm and supportive.

Sensitive topics (sexuality, reproduction, private body topics, detailed adult-level questions):
- Do NOT give detailed explanations.
- Do NOT reject harshly.
- Use this gentle, kid-friendly deflection (paraphrase across slides if needed):
  "That's a good question. It's something people usually learn about when they are a bit older. For now, it's okay not to know everything yet. You can also ask a parent or teacher if you're curious."

Question types and how to handle them:

• "How does X work?" — explain the mechanism simply.
• "How was X invented?" / "Who invented X?" / "Where did X come from?" / "Why did people make X?" — create an invention/history lesson:
    Slide 1: The problem people had before the invention existed.
    Slide 2: What people used before (and why it wasn't good enough).
    Slide 3: How the invention came to be and how it improved over time.
    Slide 4: What the invention is like today and why it matters.
    - Avoid claiming one exact inventor unless historically certain (e.g. Alexander Graham Bell, Wright Brothers).
    - If unclear, say "many clever people improved it over time."
• "Why is/does X?" — explain the reason simply.
• "What is X?" — describe what it is and why it's interesting.
• Everyday objects (shoes, pencils, bread, glass, etc.) — explain origin or how they work.
• Nature questions (animals, plants, weather, space) — explain simply.
• Human/emotion/everyday life questions — always create a lesson, never refuse.
• Advice / safety / habits questions ("Is X good for kids?", "Is X safe?",
  "Should kids do X?", "Is too much X bad?", "How long should kids…") —
  build a BALANCED lesson, not a yes/no verdict:
    Slide 1: What it is — simple plain explanation of the thing.
    Slide 2: Good side — real benefits kids can get from it.
    Slide 3: Not-so-good side — what can happen with too much, in a calm way.
    Slide 4: Smart use — friendly tips for using it well (limits, breaks,
             balance with other activities, asking a grown-up).
  Tone for these: neutral, kid-friendly, never scary, never strict.
  Avoid extreme statements. Do NOT say "never do this" or "this is dangerous".
  Use phrases like "a little is fine", "balance is best", "take breaks".

Tone rules (always):
- Friendly, warm, non-judgmental, never scary, never strict.
- Never return an error or rejection for normal curiosity questions.

Content rules — be concise, every field must be SHORT:
- title: max 6 words.
- subtitle: one short phrase, max 8 words.
- explanation: max 2 short sentences. Simple words only.
- fun_fact: one sentence, max 20 words.
- narration: max 2 short sentences combining the key idea and fun fact. No markdown.
- Never say "I can't answer that" — always create a lesson.

image_prompt rules (35–55 words):
Goal: images should be visually stimulating and educational, not always overly bright or childish.

- Match the image mood to the topic:
  • Space: deep, wonder-filled, cosmic
  • Nature: calm, realistic, beautiful
  • Science: clear, clean, educational
  • History / inventions: warm storybook realism
  • Games / Minecraft: playful but clean
  • Human / emotions: soft, warm, respectful
- Do NOT default to "bright colorful cartoon" style for every topic.
- Use rich but balanced colors. Avoid overstimulating neon colors unless the topic genuinely needs playful energy.
- Clean composition with ONE main idea per image.
- Visually interesting enough for kids to look at, but always educational — not random decoration.

Strict no-text rule (very important — Flux cannot render text and will produce garbled gibberish):
- No text, no letters, no words, no captions, no labels, no numbers, no written symbols, no fake writing anywhere in the image.
- Do NOT request diagrams with written labels, arrows with words, captions, or any text inside images.
- Do NOT use words like "labeled", "with text", "caption", "words saying", "written on", "reading", "diagram with labels".

Always end every image_prompt with exactly:
"visually engaging educational illustration, topic-appropriate colors, clean composition, no text, no letters, no words, no captions, no labels, no numbers, no written symbols, no fake writing."

image_negative_prompt: always exactly "text, letters, words, captions, labels, numbers, written symbols, fake writing, logo, watermark, scary, baby cartoon, overly simple, clipart, neon, cluttered background"

Lesson consistency rules (very important):
- Visible lesson text (title, subtitle, explanation, fun_fact) is the SOURCE OF TRUTH.
- Narration must NOT introduce new technical words or concepts that are absent from the visible explanation/subtitle/fun_fact on the same slide.
- If narration uses a term like "airfoil", that term MUST also appear in the visible lesson text (explanation/subtitle/fun_fact) with a simple kid-friendly definition.
- Quiz questions (when generated separately) must only use concepts already present in the visible lesson text — so the lesson text must contain everything a kid needs to answer them.

Final self-check before returning JSON:
- Verify that every narration line can be understood from the visible lesson text on the same slide.
- Verify that every concept a quiz could reasonably test is present in the visible lesson text.
- If a narration mentions something extra, either add it to the visible explanation/subtitle/fun_fact with a simple definition, or remove it from the narration.

Output format — a single JSON object, no markdown fences:

{
  "title": "Lesson title (max 6 words)",
  "subject": "main noun",
  "slides": [
    {
      "number": 1,
      "title": "Slide title (max 6 words)",
      "subtitle": "One short hook phrase",
      "explanation": "Max 2 short sentences.",
      "fun_fact": "One short fun fact.",
      "narration": "Max 2 short sentences for voiceover.",
      "image_prompt": "35-55 word image description ending with style suffix.",
      "image_negative_prompt": "text, letters, words, captions, labels, numbers, written symbols, fake writing, logo, watermark, scary, baby cartoon, overly simple, clipart, neon, cluttered background"
    }
  ]
}
"""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in Claude response")
    return json.loads(cleaned[start : end + 1])


_ALLOWED_SHAPE_TYPES = {"rect", "circle", "ellipse", "polygon", "line", "text", "label"}

# Tool definition used with tool_use mode — the API validates Claude's output
# against this schema, guaranteeing structurally valid JSON every time.
_LESSON_TOOL: dict[str, Any] = {
    "name": "produce_lesson",
    "description": "Output a complete 4-slide illustrated lesson for a child's question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subject": {"type": "string"},
            "slides": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "number":               {"type": "integer"},
                        "title":                {"type": "string"},
                        "subtitle":             {"type": "string"},
                        "explanation":          {"type": "string"},
                        "fun_fact":             {"type": "string"},
                        "narration":            {"type": "string"},
                        "image_prompt":         {"type": "string"},
                        "image_negative_prompt":{"type": "string"},
                    },
                    "required": [
                        "number", "title", "explanation", "fun_fact",
                        "narration", "image_prompt",
                    ],
                },
            },
        },
        "required": ["title", "subject", "slides"],
    },
}


_BG_COLORS = ["#B5D4F4", "#EEEDFE", "#5DCAA5", "#FAC775", "#AFA9EC"]


def _default_spec(slide_idx: int, title: str, subject: str) -> dict[str, Any]:
    """Generate a simple colored placeholder spec when Claude doesn't supply one."""
    bg = _BG_COLORS[slide_idx % len(_BG_COLORS)]
    return {
        "width": 640, "height": 240, "background": bg,
        "shapes": [
            {"type": "rect", "x": 220, "y": 60, "w": 200, "h": 120, "fill": "#FFFFFF",
             "animate": {"dx": 0, "dy": 6, "yoyo": True}},
            {"type": "text", "x": 320, "y": 115, "text": subject[:24],
             "fill": "#1F1B2E", "size": 22, "weight": "bold", "anchor": "center"},
            {"type": "text", "x": 320, "y": 145, "text": title[:32],
             "fill": "#534AB7", "size": 13, "weight": "normal", "anchor": "center"},
        ],
    }


def _validate_lesson(data: dict[str, Any]) -> dict[str, Any]:
    required_top = {"title", "subject", "slides"}
    missing = required_top - set(data)
    if missing:
        raise ValueError(f"Lesson missing keys: {missing}")
    if not isinstance(data["slides"], list) or len(data["slides"]) != 4:
        raise ValueError(f"Expected 4 slides, got {len(data.get('slides', []))}")
    required_slide = {"number", "title", "explanation", "fun_fact", "narration", "image_prompt"}
    subject = data.get("subject", "")
    for idx, slide in enumerate(data["slides"], start=1):
        miss = required_slide - set(slide)
        if miss:
            raise ValueError(f"Slide {idx} missing keys: {miss}")
        # Attach a default spec if Claude didn't supply one
        if not isinstance(slide.get("spec"), dict):
            slide["spec"] = _default_spec(idx - 1, slide.get("title", ""), subject)
        else:
            spec = slide["spec"]
            shapes = spec.get("shapes")
            if not isinstance(shapes, list) or not shapes:
                slide["spec"] = _default_spec(idx - 1, slide.get("title", ""), subject)
            else:
                spec["shapes"] = [
                    s for s in shapes
                    if isinstance(s, dict) and s.get("type") in _ALLOWED_SHAPE_TYPES
                ]
                if not spec["shapes"]:
                    slide["spec"] = _default_spec(idx - 1, slide.get("title", ""), subject)
    return data


_QUIZ_TOOL: dict[str, Any] = {
    "name": "produce_quiz",
    "description": "Output a 3-question multiple-choice quiz based on a lesson.",
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "question":      {"type": "string"},
                        "options":       {"type": "array", "minItems": 3, "maxItems": 3,
                                          "items": {"type": "string"}},
                        "correct_index": {"type": "integer", "minimum": 0, "maximum": 2},
                        "explanation":   {"type": "string"},
                    },
                    "required": ["question", "options", "correct_index", "explanation"],
                },
            },
        },
        "required": ["questions"],
    },
}

_QUIZ_SYSTEM = """You generate short, fun multiple-choice quizzes for kids aged 6–10 based on a lesson they just read.

Rules:
- Create exactly 3 questions.
- Each question tests basic understanding from the lesson only.
- 3 answer options per question, exactly one correct.
- Language: short sentences, easy words, clear meaning.
- Questions should feel inviting and low-pressure — like a fun game, not a test.
- Tone: warm, playful, encouraging. Never tricky, negative, or discouraging.
- Wrong answers should be plausible but clearly different — never mean or silly.
- Each explanation is one simple, positive sentence saying why the answer is correct."""


def generate_quiz(lesson: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Generate a 3-question quiz from a lesson dict. Returns {questions: [...]}."""
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing.")

    # Build a compact lesson summary to send to Claude
    slides_text = "\n".join(
        f"Slide {s['number']}: {s['title']} — {s['explanation']} Fun fact: {s.get('fun_fact', '')}"
        for s in lesson.get("slides", [])
    )
    lesson_summary = f"Lesson title: {lesson.get('title', '')}\n\n{slides_text}"

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=800,
        system=_QUIZ_SYSTEM,
        tools=[_QUIZ_TOOL],
        tool_choice={"type": "tool", "name": "produce_quiz"},
        messages=[{"role": "user", "content": f"Here is the lesson:\n\n{lesson_summary}\n\nNow create the quiz."}],
    )

    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"), None
    )
    if tool_block is None:
        raise ValueError("Quiz: no tool_use block in Claude response")
    return tool_block.input


def generate_lesson(question: str, settings: Settings) -> dict[str, Any]:
    """Generate a 4-slide lesson from a child's question.

    Returns a dict with: title, subject, slides[], meta (timing, cost).
    Raises on API or parsing failure — caller should surface a friendly error.
    """
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to .env (get one at console.anthropic.com)."
        )

    client = Anthropic(api_key=settings.anthropic_api_key)
    started = time.time()

    logger.info("Generating lesson for question: %s", question)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[_LESSON_TOOL],
        tool_choice={"type": "tool", "name": "produce_lesson"},
        messages=[
            {
                "role": "user",
                "content": f'A child asked: "{question}"\n\nCreate the 4-slide lesson now. Keep every text field short.',
            }
        ],
    )

    claude_ms = int((time.time() - started) * 1000)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    logger.info("Claude done in %dms — %d in / %d out tokens", claude_ms, input_tokens, output_tokens)

    # With tool_use, the API guarantees valid JSON in tool_use blocks
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        raw_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        lesson = _validate_lesson(_extract_json(raw_text))
    else:
        lesson = _validate_lesson(tool_block.input)

    elapsed_ms = int((time.time() - started) * 1000)
    # Sonnet 4.6 pricing: $3 per 1M input, $15 per 1M output
    cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000

    lesson["meta"] = {
        "generation_time_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost_usd, 5),
        "model": settings.anthropic_model,
    }

    logger.info("Lesson total %dms — $%.5f", elapsed_ms, cost_usd)
    return lesson
