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


SYSTEM_PROMPT = """You are a friendly teacher creating illustrated slides for kids aged 6 to 10.

You answer any question a curious child might ask about science, nature, history, inventions, technology, everyday objects, or how the world works. You are NOT limited to "how does X work?" questions.

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

Content rules:
- Short, simple sentences. Concrete words over abstract.
- Kid-friendly analogies (toys, animals, food, weather).
- One main idea per slide.
- A playful fun fact per slide.
- Never say "I can't answer that" — always create a lesson.

Illustration rules:
- Each slide has a `spec` field: a drawing description, NOT code.
- Canvas is 640 wide × 240 tall, origin top-left.
- Colors must be from this palette (no other hex values):
  purples  #7F77DD #534AB7 #AFA9EC #EEEDFE
  oranges  #EF9F27 #FAC775 #BA7517
  greens   #5DCAA5 #0F6E56 #3B6D11 #639922
  blues    #B5D4F4 #185FA5 #0C447C
  reds     #E24B4A #A32D2D
  neutrals #FFFFFF #FDF8F3 #888780 #1F1B2E

Allowed shape types (use ONLY these; any other type is ignored):
  {"type":"rect",    "x":N,"y":N,"w":N,"h":N, "fill":"#..."}
  {"type":"circle",  "cx":N,"cy":N,"r":N,     "fill":"#..."}
  {"type":"ellipse", "cx":N,"cy":N,"rx":N,"ry":N,"fill":"#..."}
  {"type":"polygon", "points":[[x,y],...],   "fill":"#..."}
  {"type":"line",    "x1":N,"y1":N,"x2":N,"y2":N,"stroke":"#...","stroke_width":N,"dash":"-" or "--"}
  {"type":"text",    "x":N,"y":N,"text":"...","fill":"#...","size":N,"weight":"bold"|"normal","anchor":"left"|"center"|"right"}
  {"type":"label",   "x":N,"y":N,"text":"...","pointTo":[x,y],"fill":"#...","size":N}  // dashed leader line from (x,y) to pointTo

Animation (use on EVERY slide — at least one animated shape per slide):
- Add an "animate" field to any shape except labels:
    "animate":{"dx":N,"dy":N,"yoyo":true|false}
- dx/dy are pixel offsets at the peak of the motion. Keep them subtle (usually 3 to 30).
- yoyo=true → bounces back and forth (floating clouds, flame flicker, bobbing rocket).
- yoyo=false → one-way motion (rain drops falling, smoke rising).
- Animate motion that makes physical sense for the subject. Don't animate labels or static ground.
- Keep total animated shapes per slide small (usually 3–8) so the render stays fast.

Drawing composition rules:
- The SUBJECT of the slide must be instantly recognizable as a flat, friendly illustration.
  * Rocket → nose cone (triangle), body (rect), fins (triangles), flame (triangle).
  * Volcano → cone mountain, crater (dark opening), lava fountain, lava stream.
  * Plane → fuselage, swept wings, tail, windows.
  * Rain → cloud (ellipses), falling drops (small ellipses with yoyo=false).
- Use small labels with dashed leader lines to name important parts.
- Keep shapes inside the canvas.

Image prompt rules for Flux image generation:
- Each slide must include an `image_prompt` field.
- The image must be a clear educational visual for children aged 6–10.
- Use a simple centered composition with one main subject.
- Make the subject instantly recognizable.
- Show the mechanism clearly using visual action, not text.
- Avoid clutter, extra characters, fantasy objects, random decorations, or too many background elements.
- Do NOT include text, labels, captions, letters, numbers, logos, or watermarks.
- Use bright, clean, kid-friendly colors.
- Style should be: children's educational storybook illustration, simple science poster, flat bright colors, soft rounded shapes.
- The image_prompt should be 35–55 words.
- Include the slide's key idea visually.
- Always end with: "children's educational storybook illustration, simple science poster, flat bright colors, soft rounded shapes, no text, no labels, no letters, no numbers."

Each slide must also include an `image_negative_prompt` field with exactly this value:
"text, labels, captions, letters, numbers, logo, watermark, scary, realistic photo, cluttered background, extra people, extra animals, confusing diagram"

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
      "narration": "Spoken-friendly prose that combines title + explanation + fun fact. No markdown.",
      "image_prompt": "Clear educational scene showing the slide's key idea. One main subject, simple composition, visual action. children's educational storybook illustration, simple science poster, flat bright colors, soft rounded shapes, no text, no labels, no letters, no numbers.",
      "image_negative_prompt": "text, labels, captions, letters, numbers, logo, watermark, scary, realistic photo, cluttered background, extra people, extra animals, confusing diagram",
      "spec": {
        "width": 640,
        "height": 240,
        "background": "#B5D4F4",
        "shapes": [ ...shape objects... ]
      }
    },
    ... 3 more slides
  ]
}

The first slide's spec/image_prompt should focus on the SUBJECT itself (the recognizable subject).
Slides 2-4 can show mechanisms, forces, or sub-parts.
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


def _repair_json(bad_text: str, client: "Anthropic", model: str) -> dict[str, Any]:
    """Ask Claude to fix its own malformed JSON output."""
    logger.warning("JSON parse failed — attempting repair via Claude")
    repair_response = client.messages.create(
        model=model,
        max_tokens=8000,
        system="You are a JSON repair assistant. Fix the JSON so it is valid. Output ONLY the corrected JSON, no markdown, no explanation.",
        messages=[
            {
                "role": "user",
                "content": (
                    "The following JSON is malformed. Fix every syntax error "
                    "(unescaped quotes, missing commas, trailing commas, etc.) "
                    "and return only the corrected JSON:\n\n" + bad_text
                ),
            }
        ],
    )
    repaired = "".join(
        block.text for block in repair_response.content if getattr(block, "type", None) == "text"
    )
    return _extract_json(repaired)


_ALLOWED_SHAPE_TYPES = {"rect", "circle", "ellipse", "polygon", "line", "text", "label"}


def _validate_lesson(data: dict[str, Any]) -> dict[str, Any]:
    required_top = {"title", "subject", "slides"}
    missing = required_top - set(data)
    if missing:
        raise ValueError(f"Lesson missing keys: {missing}")
    if not isinstance(data["slides"], list) or len(data["slides"]) != 4:
        raise ValueError(f"Expected 4 slides, got {len(data.get('slides', []))}")
    required_slide = {"number", "title", "explanation", "fun_fact", "narration", "spec", "image_prompt"}
    for idx, slide in enumerate(data["slides"], start=1):
        miss = required_slide - set(slide)
        if miss:
            raise ValueError(f"Slide {idx} missing keys: {miss}")
        spec = slide.get("spec")
        if not isinstance(spec, dict):
            raise ValueError(f"Slide {idx} spec is not a dict")
        shapes = spec.get("shapes")
        if not isinstance(shapes, list) or not shapes:
            raise ValueError(f"Slide {idx} spec has no shapes")
        # Drop unknown shape types defensively (renderer would skip them anyway)
        spec["shapes"] = [s for s in shapes if isinstance(s, dict) and s.get("type") in _ALLOWED_SHAPE_TYPES]
        if not spec["shapes"]:
            raise ValueError(f"Slide {idx} spec has no recognizable shapes")
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
    try:
        lesson = _validate_lesson(_extract_json(raw_text))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Initial parse failed (%s) — trying repair", exc)
        lesson = _validate_lesson(_repair_json(raw_text, client, settings.anthropic_model))

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
