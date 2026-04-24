"""OpenAI TTS — turn narration text into MP3 audio bytes."""
import logging

from openai import OpenAI

from .config import Settings

logger = logging.getLogger(__name__)

# tts-1 is priced at $15 per 1M characters (input chars, not tokens)
_PRICE_PER_CHAR = 15.0 / 1_000_000


def synthesize(text: str, settings: Settings) -> tuple[bytes, float]:
    """Return (mp3_bytes, estimated_cost_usd). Raises on missing key or API error."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.audio.speech.create(
        model=settings.openai_tts_model,
        voice=settings.openai_tts_voice,
        input=text,
        response_format="mp3",
    )
    cost = len(text) * _PRICE_PER_CHAR
    return response.content, cost
