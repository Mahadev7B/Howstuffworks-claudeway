"""Load settings from environment."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    anthropic_api_key: str
    anthropic_model: str
    openai_api_key: str
    openai_tts_model: str
    openai_tts_voice: str
    flask_secret_key: str
    port: int


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "nova"),
        flask_secret_key=os.getenv("FLASK_SECRET_KEY", "dev-secret-key"),
        port=int(os.getenv("PORT", "5000")),
    )
