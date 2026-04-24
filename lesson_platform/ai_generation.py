"""Claude-powered lesson generator.

Single API call produces:
  1. 4 slides of kid-friendly explanation
  2. SVG illustration for each slide (recognizable, labeled shapes)
  3. Narration text for voiceover

No paid image API. No server-side audio. All free downstream.
"""
import json
import logging
import re
import time
from typing import Any

from anthropic import Anthropic

from .config import Settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a friendly science teacher creating illustrated slides for kids aged 6 to 10.

When given a "How does X work?" question, you produce a 4-slide micro-lesson.

Rules for the content:
- Use short, simple sentences. Prefer concrete words over abstract ones.
- Use analogies a child would understand (like toys, animals, food, weather).
- Each slide has ONE main idea.
- Include a playful fun fact per slide.

Rules for the SVG illustration on each slide:
- The illustration must depict the SUBJECT of that slide in a shape that a kid would INSTANTLY RECOGNIZE.
  * If the topic is a rocket, draw an actual rocket with nose cone, body, fins, and fire.
  * If it's a plane, draw fuselage, swept wings, tail, windows.
  * If it's a volcano, draw a cone mountain with lava.
  * If it's a process (like rain), draw the recognizable objects involved (clouds, droplets, sun).
- Add small labels with dashed leader lines to name the important parts.
- Use friendly, colourful fills. Avoid photorealism. Think: clean flat illustration.
- Set viewBox="0 0 640 240". Keep all content inside.
- Use only these colours: #7F77DD #534AB7 #AFA9EC #EEEDFE (purples), #EF9F27 #FAC775 #BA7517 (oranges),
  #5DCAA5 #0F6E56 #3B6D11 #639922 (greens), #B5D4F4 #185FA5 #0C447C (blues), #E24B4A #A32D2D (reds),
  #888780 for neutral grey labels.
- Do NOT use <style>, <defs> filters, gradients, or external references. Only plain shapes and text.
- No emoji, no external images.
- Label font-family should be "sans-serif", font-size 12 or 13 for labels, 15 for headings.

Output format — respond ONLY with a single JSON object (no markdown fences, no commentary):

{
  "title": "Plain-English title of the lesson",
  "subject": "the main noun (e.g. 'rocket', 'rain', 'volcano')",
  "slides": [
    {
      "number": 1,
      "title": "Short catchy title (max 4 words)",
      "subtitle": "One-line hook",
      "explanation": "2-3 sentences. Simple vocabulary.",
      "fun_fact": "One delightful fact.",
      "narration": "Full text for voiceover. Combines title + explanation + fun fact in spoken-friendly prose. No markdown.",
      "svg": "<svg width='100%' viewBox='0 0 640 240' xmlns='http://www.w3.org/2000/svg'>...</svg>"
    },
    ... 3 more slides
  ]
}

The first slide's SVG should focus on the SUBJECT itself (the recognizable shape with labelled parts).
Slides 2-4 can show specific mechanisms, forces, or sub-parts.
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Claude sometimes wraps JSON in fences even when asked not to. Strip them."""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    # Find first { and last } to be safe
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in Claude response")
    return json.loads(cleaned[start : end + 1])


def _validate_lesson(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure lesson has the expected structure. Raises on bad shape."""
    required_top = {"title", "subject", "slides"}
    missing = required_top - set(data)
    if missing:
        raise ValueError(f"Lesson missing keys: {missing}")
    if not isinstance(data["slides"], list) or len(data["slides"]) != 4:
        raise ValueError(f"Expected 4 slides, got {len(data.get('slides', []))}")
    required_slide = {"number", "title", "explanation", "fun_fact", "narration", "svg"}
    for idx, slide in enumerate(data["slides"], start=1):
        miss = required_slide - set(slide)
        if miss:
            raise ValueError(f"Slide {idx} missing keys: {miss}")
        if "<svg" not in slide["svg"]:
            raise ValueError(f"Slide {idx} has no <svg>")
    return data


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
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f'A child asked: "{question}"\n\nCreate the 4-slide lesson now.',
            }
        ],
    )

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    lesson = _validate_lesson(_extract_json(raw_text))

    elapsed_ms = int((time.time() - started) * 1000)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    # Sonnet 4.6 pricing: $3 per 1M input, $15 per 1M output
    cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000

    lesson["meta"] = {
        "generation_time_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost_usd, 5),
        "model": settings.anthropic_model,
    }

    logger.info(
        "Lesson generated in %dms — %d in / %d out tokens ($%.5f)",
        elapsed_ms,
        input_tokens,
        output_tokens,
        cost_usd,
    )
    return lesson
