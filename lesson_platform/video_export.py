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
    "vertical":  (720, 1280),
    "landscape": (1280, 720),
}
RENDER_SCALE = 1.0

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
    """Render a single slide as a vertical reel frame (full-bleed image + text card)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    pad = int(width * 0.06)
    usable_w = width - 2 * pad

    # ── Top half: full-bleed illustration ────────────────────────────────────
    img_h = int(height * 0.48)
    if slide.get("image_data_url"):
        try:
            data_url = slide["image_data_url"]
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(b64)
            slide_img = Image.open(io.BytesIO(raw)).convert("RGB")
            # Scale to fill the image area — cover both width and height, then crop
            ratio_w = width / slide_img.width
            ratio_h = img_h / slide_img.height
            ratio = max(ratio_w, ratio_h)  # cover: whichever is larger fills the box
            new_w = int(slide_img.width * ratio)
            new_h = int(slide_img.height * ratio)
            slide_img = slide_img.resize((new_w, new_h), Image.LANCZOS)
            # Centre-crop to exact frame size
            left = max(0, (new_w - width) // 2)
            top = max(0, (new_h - img_h) // 2)
            slide_img = slide_img.crop((left, top, left + width, top + img_h))
            img.paste(slide_img, (0, 0))
        except Exception as e:
            logger.warning("Could not render slide image: %s", e)
            # Gradient placeholder
            for row in range(img_h):
                t = row / img_h
                r = int(255 * (1 - t) + 106 * t)
                g = int(248 * (1 - t) + 76 * t)
                b = int(240 * (1 - t) + 147 * t)
                draw.line([(0, row), (width, row)], fill=(r, g, b))
    else:
        for row in range(img_h):
            t = row / img_h
            r = int(255 * (1 - t) + 106 * t)
            g = int(248 * (1 - t) + 76 * t)
            b_val = int(240 * (1 - t) + 147 * t)
            draw.line([(0, row), (width, row)], fill=(r, g, b_val))

    # Gradient fade from image into card
    fade_h = int(height * 0.06)
    for row in range(fade_h):
        alpha = row / fade_h
        r = int(BG_COLOR[0] * alpha + (BG_COLOR[0] * 0.3) * (1 - alpha))
        g = int(BG_COLOR[1] * alpha + (BG_COLOR[1] * 0.3) * (1 - alpha))
        b_val = int(BG_COLOR[2] * alpha + (BG_COLOR[2] * 0.3) * (1 - alpha))
        draw.line([(0, img_h - fade_h + row), (width, img_h - fade_h + row)], fill=(r, g, b_val))

    # ── Watermark pill at top-left ────────────────────────────────────────────
    brand_font = _load_font(int(width * 0.032))
    brand_text = "🦉 asklilowl.com"
    pill_pad_x, pill_pad_y = int(width * 0.025), int(height * 0.009)
    pill_x, pill_y = int(width * 0.04), int(height * 0.025)
    bb = draw.textbbox((0, 0), brand_text, font=brand_font)
    pill_w = (bb[2] - bb[0]) + pill_pad_x * 2
    pill_h = (bb[3] - bb[1]) + pill_pad_y * 2
    # Semi-transparent dark background so it reads on any image
    draw.rounded_rectangle(
        [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
        radius=pill_h // 2,
        fill=(0, 0, 0, 160),
    )
    draw.text((pill_x + pill_pad_x, pill_y + pill_pad_y), brand_text, font=brand_font, fill=(255, 255, 255))

    # ── Bottom card: title + explanation + fun fact ───────────────────────────
    card_y = img_h
    card_h = height - card_y
    # Card background already BG_COLOR from Image.new

    y = card_y + int(height * 0.025)

    # Title
    title_font = _load_font(int(width * 0.065))
    title = slide.get("title", "")
    title_lines = _wrap_text(title, title_font, usable_w, draw)
    for line in title_lines[:2]:
        draw.text((pad, y), line, font=title_font, fill=ACCENT)
        bbox = draw.textbbox((pad, y), line, font=title_font)
        y += (bbox[3] - bbox[1]) + int(height * 0.006)

    y += int(height * 0.012)

    # Explanation
    exp_font = _load_font(int(width * 0.042))
    explanation = slide.get("explanation", "")
    exp_lines = _wrap_text(explanation, exp_font, usable_w, draw)
    for line in exp_lines[:5]:
        draw.text((pad, y), line, font=exp_font, fill=TEXT_DARK)
        bbox = draw.textbbox((pad, y), line, font=exp_font)
        y += (bbox[3] - bbox[1]) + int(height * 0.005)

    y += int(height * 0.015)

    # Fun fact box
    fun_fact = slide.get("fun_fact", "")
    remaining = height - int(height * 0.04) - y
    if fun_fact and remaining > int(height * 0.08):
        ff_font = _load_font(int(width * 0.036))
        ff_lines = _wrap_text(fun_fact, ff_font, usable_w - pad, draw)[:3]
        line_h = int(height * 0.042)
        box_h = int(height * 0.018) + len(ff_lines) * line_h + int(height * 0.015)
        box_y = min(y, height - int(height * 0.04) - box_h)
        draw.rounded_rectangle(
            [pad, box_y, width - pad, box_y + box_h],
            radius=int(width * 0.03),
            fill=FUNFACT_BG,
            outline=ACCENT,
            width=3,
        )
        ty = box_y + int(height * 0.012)
        for line in ff_lines:
            draw.text((pad + int(pad * 0.3), ty), line, font=ff_font, fill=TEXT_PURPLE)
            ty += line_h

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = buf.getvalue()
    # Explicitly free large objects to keep memory footprint low
    del img, draw, buf
    return result


def _mp3_duration(data: bytes) -> float:
    """Estimate MP3 duration from file size and bitrate header — no ffprobe needed.

    Reads the first valid MPEG frame header to get bitrate, then divides total
    size by bytes-per-second. Accurate to within ~0.1s for CBR MP3s (OpenAI TTS output).
    Falls back to 6.0s on any parse error.
    """
    try:
        # Scan for sync word 0xFF 0xEx or 0xFF 0xFx (MPEG frame sync)
        for offset in range(min(len(data) - 4, 4096)):
            b0, b1 = data[offset], data[offset + 1]
            if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
                continue
            # Extract bitrate index from bits 4-7 of byte 2
            bitrate_idx = (data[offset + 2] >> 4) & 0xF
            bitrate_kbps = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0][bitrate_idx]
            if bitrate_kbps == 0:
                continue
            return len(data) / (bitrate_kbps * 125)  # bytes / (kbps * 125) = seconds
    except Exception:
        pass
    return 6.0


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
    pad_color = f"{BG_COLOR[0]:02x}{BG_COLOR[1]:02x}{BG_COLOR[2]:02x}"
    tmpdir = tempfile.mkdtemp(prefix="lilowl_video_")
    try:
        n = len(slides)

        # ── Phase 1: render all Pillow frames sequentially (can't parallelise —
        # each frame is ~3MB and we free it immediately to stay under RAM limit)
        frame_paths = []
        for i, slide in enumerate(slides):
            frame_png = _render_slide_frame(slide, width, height)
            frame_path = os.path.join(tmpdir, f"frame_{i}.png")
            with open(frame_path, "wb") as f:
                f.write(frame_png)
            del frame_png
            frame_paths.append(frame_path)

        # ── Phase 2: write audio files + get durations (pure I/O, fast)
        audio_paths = []
        durations = []
        for i, audio_bytes in enumerate(audio_clips):
            audio_path = os.path.join(tmpdir, f"audio_{i}.mp3")
            if audio_bytes:
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                duration = _mp3_duration(audio_bytes) + 0.5
            else:
                duration = 4.0
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                     "-t", str(duration), audio_path],
                    capture_output=True, check=True,
                )
            audio_paths.append(audio_path)
            durations.append(duration)

        # ── Phase 3: encode segments sequentially to cap peak RAM at one
        # ffmpeg process at a time (720p ffmpeg uses ~100MB working set)
        segment_paths = []
        for i in range(n):
            seg_path = os.path.join(tmpdir, f"seg_{i}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", frame_paths[i],
                "-i", audio_paths[i],
                "-c:v", "libx264", "-tune", "stillimage", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-t", str(durations[i]),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={pad_color}",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg segment {i} failed: {result.stderr.decode()[:500]}")
            try: os.unlink(frame_paths[i])
            except Exception: pass
            segment_paths.append(seg_path)

        # ── Phase 4: concat segments (stream copy — near instant)
        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for seg in segment_paths:
                f.write(f"file '{seg}'\n")

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="lilowl_export_")
        os.close(out_fd)

        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c", "copy", out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()[:500]}")

        return out_path

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
