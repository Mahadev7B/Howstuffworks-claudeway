"""Load reviewed, local-only editorial lessons without touching the question cache."""

import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from lesson_platform.library import prepare_publication


def _required_text(record: dict, field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Editorial package needs {field}")
    return value.strip()


def _asset_data_url(folder: Path, relative_name: str) -> str:
    if not isinstance(relative_name, str) or not relative_name.strip():
        raise ValueError("Editorial package needs an image path")
    target = (folder / relative_name).resolve()
    if not target.is_relative_to(folder.resolve()):
        raise ValueError("Editorial images must stay inside the asset folder")
    if target.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Editorial images must be raster assets")
    try:
        with Image.open(target) as source:
            source.verify()
        with Image.open(target) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("Editorial image dimensions are too large")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1672, 1050), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, "WEBP", quality=85, method=6)
    except (FileNotFoundError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Editorial image is missing or invalid: {relative_name}") from exc
    return "data:image/webp;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def load_editorial_batch(manifest_path: str | Path) -> list[dict]:
    """Return publication-ready in-memory snapshots; never opens a DB connection."""
    path = Path(manifest_path).resolve()
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Editorial batch needs a non-empty lesson list")
    result = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Editorial batch entries must be objects")
        metadata = {field: _required_text(record, field) for field in
                    ("slug", "question", "title", "summary", "category")}
        if metadata["slug"] in seen:
            raise ValueError("Editorial batch has a duplicate slug")
        seen.add(metadata["slug"])
        slides = record.get("slides")
        if not isinstance(slides, list) or not slides:
            raise ValueError("Editorial lesson needs slides")
        lesson_slides = []
        for slide in slides:
            if not isinstance(slide, dict):
                raise ValueError("Editorial slides must be objects")
            copy = {field: _required_text(slide, field) for field in
                    ("title", "explanation", "fun_fact", "narration")}
            copy["image_data_url"] = _asset_data_url(path.parent, _required_text(slide, "image"))
            lesson_slides.append(copy)
        lesson = {"title": metadata["title"], "slides": lesson_slides}
        cover_data_url = _asset_data_url(path.parent, _required_text(record, "cover"))
        prepare_publication(lesson, cover_data_url=cover_data_url)
        result.append({**metadata,
                       "sort_order": record.get("sort_order", 0),
                       "featured_rank": record.get("featured_rank"),
                       "lesson": lesson, "cover_data_url": cover_data_url})
    return result
