"""Flux image generation via fal.ai (Schnell or Dev).

Returns (image_bytes, mime, cost_usd) or raises on failure.
"""
import logging
import os
import time
import urllib.request
from typing import Any

import fal_client

from .config import Settings

logger = logging.getLogger(__name__)

_SIZE_TO_MP = {
    "square": 1.0,
    "square_hd": 1.0,
    "portrait_4_3": 0.79,
    "portrait_16_9": 0.59,
    "landscape_4_3": 0.79,
    "landscape_16_9": 0.59,
}

# fal.ai pricing per megapixel (as of late 2025).
_PRICE_PER_MP_BY_MODEL = {
    "fal-ai/flux/schnell": 0.003,
    "fal-ai/flux/dev":     0.025,
    "fal-ai/flux-pro":     0.05,
}
_DEFAULT_PRICE_PER_MP = 0.025
_POLL_INTERVAL = 1.0   # seconds between status checks (fal default is 0.1)


def _estimate_cost(model: str, size: str) -> float:
    return _SIZE_TO_MP.get(size, 0.6) * _PRICE_PER_MP_BY_MODEL.get(model, _DEFAULT_PRICE_PER_MP)


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

    os.environ["FAL_KEY"] = settings.fal_api_key

    is_schnell = "schnell" in settings.flux_model
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "image_size": settings.flux_image_size,
        # Schnell is a 4-step distilled model; Dev/Pro need ~28 steps.
        "num_inference_steps": 4 if is_schnell else 28,
        "num_images": 1,
        "enable_safety_checker": True,
    }
    # Dev/Pro support guidance_scale (CFG) and respect negative prompts well.
    # Schnell ignores both — passing them is harmless but useless.
    if not is_schnell:
        arguments["guidance_scale"] = 3.5
    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt

    started = time.time()
    try:
        handle = fal_client.submit(settings.flux_model, arguments=arguments)
        # Poll at 1s intervals (fal default is 0.1s — too noisy in logs)
        for _ in handle.iter_events(with_logs=False, interval=_POLL_INTERVAL):
            pass
        resp = handle.client.get(handle.response_url)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
    except Exception as exc:
        logger.exception("Flux generation failed")
        raise RuntimeError(f"Flux generation failed: {exc}") from exc
    logger.info("Flux done in %.1fs", time.time() - started)

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

    cost = _estimate_cost(settings.flux_model, settings.flux_image_size)
    return data, content_type or "image/png", cost


def generate_images_batch(
    prompts: list[str],
    settings: Settings,
    negative_prompt: str | None = None,
) -> list[tuple[bytes, str, float]]:
    """Generate multiple images in a single Flux API call.

    Returns a list of (bytes, mime, cost_usd) in the same order as prompts.
    Flux supports multiple prompts via repeated submit but not truly batched
    multi-prompt — so we send num_images=len(prompts) with the first prompt
    and one call per unique prompt when they differ.

    For lessons all slides have distinct prompts, so we make one call per
    prompt but submit them all concurrently using fal_client.submit() and
    collect results together, keeping a single billing event in the caller.

    Actually: Flux only accepts one prompt per call but does support
    num_images>1 for the *same* prompt. Since each slide has a unique prompt
    we still need N submits — but we fire them all before polling any,
    minimising wall-clock time. This function encapsulates that pattern.
    """
    if not settings.fal_api_key:
        raise RuntimeError("FAL_KEY is not set")
    os.environ["FAL_KEY"] = settings.fal_api_key

    is_schnell = "schnell" in settings.flux_model
    cost_each = _estimate_cost(settings.flux_model, settings.flux_image_size)

    def _base_args(prompt: str) -> dict:
        args: dict = {
            "prompt": prompt,
            "image_size": settings.flux_image_size,
            "num_inference_steps": 4 if is_schnell else 28,
            "num_images": 1,
            "enable_safety_checker": True,
        }
        if not is_schnell:
            args["guidance_scale"] = 3.5
        if negative_prompt:
            args["negative_prompt"] = negative_prompt
        return args

    started = time.time()
    try:
        # Submit all at once, then poll all in parallel via threads
        handles = [
            fal_client.submit(settings.flux_model, arguments=_base_args(p))
            for p in prompts
        ]

        def _collect(handle):
            for _ in handle.iter_events(with_logs=False, interval=_POLL_INTERVAL):
                pass
            resp = handle.client.get(handle.response_url)
            resp.raise_for_status()
            return resp.json()

        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        with ThreadPoolExecutor(max_workers=len(handles)) as ex:
            future_to_idx = {ex.submit(_collect, h): i for i, h in enumerate(handles)}
            results = [None] * len(handles)
            for fut in _as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()
    except Exception as exc:
        logger.exception("Flux batch generation failed")
        raise RuntimeError(f"Flux batch generation failed: {exc}") from exc

    logger.info("Flux batch (%d images) done in %.1fs", len(prompts), time.time() - started)

    output = []
    for result in results:
        images = result.get("images") or []
        if not images or not images[0].get("url"):
            raise RuntimeError("Flux batch: missing image in result")
        url = images[0]["url"]
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "howstuffworks-claudeway/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                mime = r.headers.get("Content-Type", "image/png").split(";")[0].strip()
                data = r.read()
        except Exception as exc:
            raise RuntimeError(f"Flux batch image download failed: {exc}") from exc
        if not data:
            raise RuntimeError("Flux batch: empty image")
        output.append((data, mime or "image/png", cost_each))

    return output
