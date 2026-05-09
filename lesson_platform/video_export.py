"""Lesson video export — compose slide images + TTS audio into an MP4 reel.

Requires ffmpeg in PATH. Pillow is used to render text overlays onto frames.
Returns a path to a temporary MP4 file that callers should serve then delete.
"""
import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Video dimensions — vertical (Reels/Shorts) or landscape
PRESETS = {
    "vertical":  (1080, 1920),
    "landscape": (1920, 1080),
}

# Colors matching the Lil Owl brand
BG_COLOR = (255, 248, 240)      # #FFF8F0
ACCENT = (255, 107, 53)         # #FF6B35
TEXT_DARK = (45, 49, 66)        # #2D3142
TEXT_PURPLE = (106, 76, 147)    # #6A4C93
FUNFACT_BG = (255, 243, 232)    # #FFF3E8


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _load_font(size: int):
    """Load a TrueType font or fall back to PIL default."""
    from PIL import ImageFont
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_slide_frame(slide: dict, width: int, height: int) -> bytes:
    """Render a single slide as a PNG image (bytes)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    pad = int(width * 0.07)
    usable_w = width - 2 * pad
    y = int(height * 0.04)

    # Slide illustration (top ~45% of frame)
    img_area_h = int(height * 0.42)
    if slide.get("image_data_url"):
        try:
            data_url = slide["image_data_url"]
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(b64)
            slide_img = Image.open(io.BytesIO(raw)).convert("RGB")
            # Fit the image into the image area, centered
            ratio = min(usable_w / slide_img.width, img_area_h / slide_img.height)
            new_w = int(slide_img.width * ratio)
            new_h = int(slide_img.height * ratio)
            slide_img = slide_img.resize((new_w, new_h), Image.LANCZOS)
            x_off = pad + (usable_w - new_w) // 2
            img.paste(slide_img, (x_off, y))
        except Exception as e:
            logger.warning("Could not render slide image: %s", e)
    y += img_area_h + int(height * 0.02)

    # Slide title
    title_font = _load_font(int(width * 0.055))
    title = slide.get("title", "")
    title_lines = _wrap_text(title, title_font, usable_w, draw)
    for line in title_lines:
        draw.text((pad, y), line, font=title_font, fill=ACCENT)
        bbox = draw.textbbox((pad, y), line, font=title_font)
        y += (bbox[3] - bbox[1]) + int(height * 0.008)

    y += int(height * 0.01)

    # Explanation
    exp_font = _load_font(int(width * 0.038))
    explanation = slide.get("explanation", "")
    exp_lines = _wrap_text(explanation, exp_font, usable_w, draw)
    for line in exp_lines[:6]:  # cap at 6 lines to avoid overflow
        draw.text((pad, y), line, font=exp_font, fill=TEXT_DARK)
        bbox = draw.textbbox((pad, y), line, font=exp_font)
        y += (bbox[3] - bbox[1]) + int(height * 0.005)

    y += int(height * 0.015)

    # Fun fact box
    fun_fact = slide.get("fun_fact", "")
    if fun_fact and y < height * 0.88:
        ff_font = _load_font(int(width * 0.033))
        ff_label_font = _load_font(int(width * 0.036))
        ff_lines = _wrap_text(fun_fact, ff_font, usable_w - int(pad * 0.4), draw)
        box_h = int(height * 0.03) + len(ff_lines) * int(height * 0.04) + int(height * 0.015)
        box_y = min(y, int(height * 0.87) - box_h)
        draw.rounded_rectangle(
            [pad, box_y, width - pad, box_y + box_h],
            radius=int(width * 0.025),
            fill=FUNFACT_BG,
            outline=ACCENT,
            width=3,
        )
        inner_x = pad + int(pad * 0.3)
        ty = box_y + int(height * 0.012)
        draw.text((inner_x, ty), "Fun fact:", font=ff_label_font, fill=ACCENT)
        bbox = draw.textbbox((inner_x, ty), "Fun fact:", font=ff_label_font)
        ty += (bbox[3] - bbox[1]) + int(height * 0.005)
        for line in ff_lines:
            draw.text((inner_x, ty), line, font=ff_font, fill=TEXT_PURPLE)
            bbox = draw.textbbox((inner_x, ty), line, font=ff_font)
            ty += (bbox[3] - bbox[1]) + int(height * 0.004)

    # Branding footer
    brand_font = _load_font(int(width * 0.032))
    brand_text = "Ask Lil Owl 🦉"
    brand_bbox = draw.textbbox((0, 0), brand_text, font=brand_font)
    brand_x = (width - (brand_bbox[2] - brand_bbox[0])) // 2
    draw.text((brand_x, height - int(height * 0.04)), brand_text, font=brand_font, fill=ACCENT)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def export_lesson_video(
    lesson: dict[str, Any],
    audio_clips: list[bytes],  # one MP3 per slide, in order
    preset: str = "vertical",
    fade_duration: float = 0.4,
) -> str:
    """Compose a lesson into an MP4 and return the path to a temp file.

    `audio_clips` must be the same length as `lesson['slides']`. Pass b'' for
    a slide if audio is unavailable (a silent segment will be used instead).

    Raises RuntimeError if ffmpeg is unavailable or composition fails.
    """
    if not _ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed. Install it with: apt-get install ffmpeg"
        )

    slides = lesson.get("slides", [])
    if not slides:
        raise ValueError("Lesson has no slides")

    width, height = PRESETS.get(preset, PRESETS["vertical"])
    tmpdir = tempfile.mkdtemp(prefix="lilowl_video_")
    try:
        segment_paths = []
        for i, (slide, audio_bytes) in enumerate(zip(slides, audio_clips)):
            # Render frame
            frame_png = _render_slide_frame(slide, width, height)
            frame_path = os.path.join(tmpdir, f"frame_{i}.png")
            with open(frame_path, "wb") as f:
                f.write(frame_png)

            # Write audio
            audio_path = os.path.join(tmpdir, f"audio_{i}.mp3")
            if audio_bytes:
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                # Get audio duration
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                    capture_output=True, text=True,
                )
                try:
                    duration = float(probe.stdout.strip()) + 0.5  # 0.5s pause after
                except (ValueError, AttributeError):
                    duration = 6.0
            else:
                duration = 4.0
                # Create silent audio
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                     "-t", str(duration), audio_path],
                    capture_output=True, check=True,
                )

            # Compose image + audio into a segment
            seg_path = os.path.join(tmpdir, f"seg_{i}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", frame_path,
                "-i", audio_path,
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-t", str(duration),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={BG_COLOR[0]:02x}{BG_COLOR[1]:02x}{BG_COLOR[2]:02x}",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg segment {i} failed: {result.stderr.decode()[:500]}")
            segment_paths.append(seg_path)

        # Concatenate all segments
        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for seg in segment_paths:
                f.write(f"file '{seg}'\n")

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="lilowl_export_")
        os.close(out_fd)

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy",
            out_path,
        ]
        result = subprocess.run(concat_cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()[:500]}")

        return out_path

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
