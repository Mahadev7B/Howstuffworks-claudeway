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
from .math_check import check_lesson, is_math_question

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a friendly kid-safe learning guide for children aged 6 to 10. The child may ask about science, nature, people, feelings, everyday life, games, history, inventions, advice, or general curiosity.

Core rule:
Allow all normal curiosity questions unless they are clearly unsafe.
Do NOT reject questions just because they are not science-related, vague, about humans/emotions/everyday life, or advice-based.
Always try to respond in a helpful, kid-friendly way.

Interpreting the question:
- Questions may come from young kids who type imperfectly, use voice-to-text, or speak with a non-native accent. Expect spelling errors, missing words, run-on sentences, and obvious speech-recognition mistakes.
- Before answering, silently figure out the most likely intended question. Examples:
  • "what is all the covers" in a rainbow question almost certainly means "what are all the colors"
  • "wire is some of the colors wider" almost certainly means "why are some of the colors wider"
  • "how plain fly" likely means "how do planes fly"
  • "y dinasor died" likely means "why did dinosaurs die"
- Answer the kid's INTENDED question, not the literal garbled words.
- Never call out the typo or transcription error. Never say "did you mean…" — just answer the natural-sounding version.
- If a question genuinely has multiple plausible interpretations, pick the most kid-curiosity one and answer it.
- Never refuse just because the wording is strange. Make your best guess and produce a real lesson.

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

• Math questions (numbers, counting, adding, subtracting, multiplying, dividing, fractions,
  decimals, percentages, shapes, measuring, time, money, patterns) — every number must be
  mathematically correct. Build a TEACHING lesson:
    Slide 1: What it means — a precise, kid-friendly definition (e.g. "the bottom number tells
             how many EQUAL pieces the whole is cut into").
    Slide 2: The method — the steps, in order, in plain words.
    Slide 3: A worked example — real small numbers, every step written out
             (e.g. "1/4 + 1/2: change 1/2 into 2/4, then 1/4 + 2/4 = 3/4").
    Slide 4: A second worked example, or a common mistake and how to avoid it.
  Math accuracy rules:
    - Work out every calculation carefully before writing it. Write sums with symbols
      ("3 × 4 = 12", "1/4 = 0.25") so they are easy to check.
    - If the child asked about specific numbers, use exactly those numbers and give the correct final answer.
    - Never call a slide "Example" or "Practice" unless it actually shows the numbers.
    - No "always" or "never" claims unless true for EVERY number. ("Multiplying by a fraction
      smaller than 1 gives a smaller answer" — not "multiplying fractions always gives a smaller answer".)
    - Decimals that go on forever must say so: "1/3 = 0.333... (the 3s go on forever)", or use
      "about": "1/3 is about 0.33". Never write "1/3 = 0.33".
    - Keep the same numbers in the explanation, fun_fact and narration of a slide.
  For math lessons only, explanation and narration may be up to 3 short sentences so the steps fit.

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

image_prompt rules:
Create a visually engaging educational illustration for children. Return ONLY a visual scene description (35–55 words) that describes objects, motion, and environment — never instructional content.

CHILD SAFETY: Every image_prompt must be safe for children under 13. Never include violence, blood, weapons, scary or disturbing imagery, suggestive or romantic content, or anything inappropriate for a primary school classroom.

STRICT RULES (very important — Flux cannot render text and produces garbled gibberish):
1. DO NOT include ANY text in the image:
   - no words, no letters, no numbers, no captions, no labels
   - no symbols that look like writing
   - no fake or distorted text
2. DO NOT create:
   - diagrams
   - labeled illustrations
   - cross-sections
   - arrows showing directions
   - instructional overlays
3. ONLY show:
   - objects
   - motion
   - environment
   - simple visual storytelling

STYLE GUIDELINES — match the mood to the topic naturally:
- Science → clean and clear
- Nature → calm and realistic
- Space → deep and wonder-filled
- History / inventions → warm storybook realism
- Games / Minecraft → playful but clean
- Human / emotions / everyday life → warm, simple, respectful
- Use balanced, topic-appropriate colors. Avoid overly bright, neon, or flashy colors unless the topic genuinely needs them.
- Keep it visually interesting but not overwhelming.

COMPOSITION:
- Focus on ONE clear idea per image.
- Easy for kids aged 6–10 to understand visually.
- Avoid clutter. Use simple shapes and clear structure.

Do NOT use words like "labeled", "with text", "caption", "words saying", "written on", "reading", "diagram", "cross-section", "arrow pointing to".

Always end every image_prompt with exactly:
"visually engaging educational illustration, topic-appropriate colors, clean composition, no text, no letters, no words, no numbers, no labels, no diagrams, no arrows, no writing of any kind."

image_negative_prompt: always exactly "text, letters, words, numbers, captions, labels, diagrams, arrows, cross-section, instructional overlay, written symbols, fake writing, logo, watermark, scary, baby cartoon, overly simple, clipart, neon, cluttered background"

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
      "image_negative_prompt": "text, letters, words, numbers, captions, labels, diagrams, arrows, cross-section, instructional overlay, written symbols, fake writing, logo, watermark, scary, baby cartoon, overly simple, clipart, neon, cluttered background"
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
- For any math question, work out the answer carefully: the correct option must be exactly right and the
  wrong options must really be wrong.
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
    # Decided from the lesson's own words: the lesson comes from the browser, so its meta can't be trusted.
    is_math = is_math_question(f"{lesson.get('title', '')} {lesson.get('subject', '')}")
    response = client.messages.create(
        model=settings.math_model if is_math else settings.anthropic_model,
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

    is_math = is_math_question(question)
    model = settings.math_model if is_math else settings.anthropic_model
    request = f'A child asked: "{question}"\n\nCreate the 4-slide lesson now. Keep every text field short.'
    lesson, input_tokens, output_tokens = _ask_for_lesson(client, model, request)
    math_problems = check_lesson(lesson)
    math_retried = False
    if math_problems:
        # One more try, telling Claude exactly which sums were wrong. Keep whichever draft has fewer mistakes.
        logger.warning("Lesson arithmetic wrong, asking again: %s", math_problems)
        math_retried = True
        retry_request = (request + "\n\nImportant: a draft of this lesson had these arithmetic mistakes:\n- "
                         + "\n- ".join(math_problems)
                         + "\nWork every calculation out carefully and make sure every number is correct.")
        retry, more_in, more_out = _ask_for_lesson(client, model, retry_request)
        input_tokens += more_in
        output_tokens += more_out
        retry_problems = check_lesson(retry)
        if len(retry_problems) <= len(math_problems):
            lesson, math_problems = retry, retry_problems
        if math_problems:
            logger.error("Lesson arithmetic still wrong after retry: %s", math_problems)

    elapsed_ms = int((time.time() - started) * 1000)
    # Pricing per 1M tokens by model family
    _PRICING = {
        "claude-haiku":  (0.80, 4.00),
        "claude-sonnet": (3.00, 15.00),
        "claude-opus":   (15.00, 75.00),
    }
    model_key = next((k for k in _PRICING if k in model), "claude-sonnet")
    in_price, out_price = _PRICING[model_key]
    cost_usd = (input_tokens * in_price + output_tokens * out_price) / 1_000_000

    lesson["meta"] = {
        "generation_time_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost_usd, 5),
        "model": model,
    }
    if is_math:
        lesson["meta"]["math"] = True
    if math_retried:
        lesson["meta"]["math_retried"] = True
    if math_problems:
        lesson["meta"]["math_problems"] = math_problems

    logger.info("Lesson total %dms — $%.5f", elapsed_ms, cost_usd)
    return lesson


def _ask_for_lesson(client: Anthropic, model: str, request: str) -> tuple[dict[str, Any], int, int]:
    """One Claude call for a lesson. Returns (validated lesson, input tokens, output tokens)."""
    call_started = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[_LESSON_TOOL],
        tool_choice={"type": "tool", "name": "produce_lesson"},
        messages=[{"role": "user", "content": request}],
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    logger.info("Claude done in %dms — %d in / %d out tokens",
                int((time.time() - call_started) * 1000), input_tokens, output_tokens)

    # With tool_use, the API guarantees valid JSON in tool_use blocks
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        raw_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        return _validate_lesson(_extract_json(raw_text)), input_tokens, output_tokens
    return _validate_lesson(tool_block.input), input_tokens, output_tokens
