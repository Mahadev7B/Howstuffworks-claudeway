"""Flux Schnell image generation via fal.ai.

Returns (image_bytes, mime, cost_usd) or raises on failure.
"""
import logging
import os
import urllib.request
from typing import Any

import fal_client

from .config import Settings

logger = logging.getLogger(__name__)

# Flux Schnell on fal.ai pricing as of late 2025: ~$0.003/megapixel.
# Approximate cost depending on image size keyword.
_SIZE_TO_MP = {
    "square": 1.0,         # 1024x1024
    "square_hd": 1.0,
    "portrait_4_3": 0.79,  # 768x1024
    "portrait_16_9": 0.59,
    "landscape_4_3": 0.79,
    "landscape_16_9": 0.59,  # 1024x576
}
_PRICE_PER_MP = 0.003


def _estimate_cost(size: str) -> float:
    return _SIZE_TO_MP.get(size, 0.6) * _PRICE_PER_MP


def generate_image(
    prompt: str,
    settings: Settings,
    negative_prompt: str | None = None,
) -> tuple[bytes, str, float]:
    """Call Flux via fal.ai and return (bytes, mime, cost_usd).

    Raises RuntimeError if the API key is missing or generation fails.
    """
    if not settings.fal_api_key:
        raise RuntimeError("FAL_KEY is not set")

    # fal_client picks up the key from FAL_KEY env var; ensure it's set in this process
    os.environ["FAL_KEY"] = settings.fal_api_key

    arguments: dict[str, Any] = {
        "prompt": prompt,
        "image_size": settings.flux_image_size,
        "num_inference_steps": 4,    # Schnell is tuned for ~4 steps
        "num_images": 1,
        "enable_safety_checker": True,
    }
    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt

    try:
        result: dict[str, Any] = fal_client.subscribe(
            settings.flux_model,
            arguments=arguments,
            with_logs=False,
        )
    except Exception as exc:
        logger.exception("Flux generation failed")
        raise RuntimeError(f"Flux generation failed: {exc}") from exc

    images = result.get("images") or []
    if not images:
        raise RuntimeError("Flux returned no images")
    url = images[0].get("url")
    if not url:
        raise RuntimeError("Flux image had no URL")

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "howstuffworks-claudeway/1.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
            data = resp.read()
    except Exception as exc:
        logger.exception("Flux image download failed")
        raise RuntimeError(f"Flux image download failed: {exc}") from exc

    if not data:
        raise RuntimeError("Flux image was empty")

    cost = _estimate_cost(settings.flux_image_size)
    return data, content_type or "image/png", cost
